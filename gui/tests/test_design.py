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
from solprop.recipe import Recipe, parameters
from solprop.recipe.pins import TARGETS
from solprop.service import handlers
from solprop.service.explore import PIN_KEYS, PIN_SEARCH

from conftest import TINY
from solprop_gui.registry.routes import CARDS, EXPLORER_SETUP, IMPLIED, card_of

MULTIMODE = TINY.replace("mode_average = true", "modes = 2").replace(
    "n_save = 6",
    'n_save = 3\nlaunch = { "HE1,1" = 0.8, "HE1,2" = { share = 0.2, phase = 1.0 } }',
)
"""The tiny capillary in two modes, launched into both."""

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


def test_the_modes_of_a_setup_are_plain_arrays():
    reply = ask("modes", {"text": EXPLORER_SETUP})
    assert reply["pump"] == pytest.approx(800e-9) and reply["sweep"] is None
    (only,) = reply["sets"]
    assert only["value"] is None and only["z"] == 0.0
    labels = [mode["label"] for mode in only["modes"]]
    assert labels == [f"HE1,{m}:even" for m in range(1, 5)]
    first = only["modes"][0]
    names = ("neff_minus_one", "beta1", "beta2", "beta3", "alpha", "gamma")
    for name in (*names, "phase_mismatch"):
        assert first[name].shape == only["omega"].shape
    assert 4e-7 < first["zdw"] < 6e-7 and first["aeff"] > 0
    # Each mode's wave phase-matched to the soliton in the first: bluer in higher ones.
    rdw = [mode["rdw"][0] for mode in only["modes"][:3]]
    assert rdw == sorted(rdw, reverse=True) and 1.5e-7 < rdw[-1] < rdw[0] < 2.1e-7


def test_the_registry_agrees_with_solprops_pins_and_defaults():
    required = {key for _, keys, _ in TARGETS.values() for key in keys}
    assert required == set(PIN_KEYS)
    capillary = parameters("capillary")
    assert set(PIN_SEARCH) <= set(capillary)
    for name, value in IMPLIED.items():
        assert capillary[name].default == value


DESIGN_PINS = {
    "pressure": {
        "target": "zdw",
        "wavelength": "500 nm",
        "search": ["0.1 bar", "10 bar"],
    },
    "energy": {"target": "soliton_order", "order": 3},
}
"""The design tool's two targets: the ZDW at 500 nm, then a third-order soliton."""


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


def test_each_card_s_example_is_solprop_s_and_the_card_s():
    for entry in CARDS:
        if entry.example is None:
            continue
        recipe = Recipe.example(entry.example)
        assert recipe.route == entry.route
        build = recipe.to_dict()["build"]
        # A card without an example of its own rewrites another's.
        for name, value in entry.rewrite.items():
            if value is None:
                build.pop(name)
            else:
                build[name] = value
        assert card_of(recipe.route, build, enabled_only=False) is entry, entry.key


_RECT = '{ type = "RectMode", half_width_x = "125 um", half_width_y = "50 um" }'
_SLAB_MODES = '{ type = "ModeCollection.slab", half_gap = "63.5 um", orders = [1, 3] }'
SLAB = (
    Recipe.example("hollow_rectangular")
    .to_text()
    .replace(f"modes = [{_RECT}]", f"modes = {_SLAB_MODES}")
    .replace("n_theta = 12\n", "")  # a slab set is uniform in y: one y node
)
"""The waveguide example with a slab mode set in place of its rectangular mode."""


SCAN = TINY.replace("n_save = 6", "n_save = 2") + (
    '\n[scan]\nname = "pressure"\npressure = ["0.5 bar", "1 bar"]\n'
)
"""The tiny capillary at two pressures."""
