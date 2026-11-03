# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Recipes on the waveguide route: supplied modes, their fill and the export (SEP 41).

No gallery script on the route has a recipe yet (``test_examples.NO_RECIPE`` says why),
so the calls here are the user guide's: the planar guide's slab collection
(``docs/guide/waveguides/planar.md``), a rectangular guide with both polarisation
partners, and two capillary channels supplied as modes. Each recipe must build the
call's configuration hash, so the comparisons are of digests and exact.
"""

import json
import subprocess
import sys
import textwrap

import pytest

from solprop.driver import Simulation
from solprop.modes import MarcatiliMode, ModeCollection, RectMode
from solprop.recipe import Recipe, RecipeError, parameters, provenance

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.GrazingIncidenceWarning"),
]

HEADER = '[recipe]\nformat = 1\nroute = "waveguide"\n\n[build]\n'

PLANAR = HEADER + textwrap.dedent(
    """\
    modes = { type = "ModeCollection.slab", half_gap = "63.5 um", orders = [1, 3, 5] }
    length = "0.1 m"
    gas = "Ar"
    pressure = "1 bar"
    wavelength = "800 nm"
    wavelength_limits = ["650 nm", "1000 nm"]
    trange = "1.2 ps"
    fwhm = "100 fs"
    energy = 0.5  # J per metre of y
    envelope = true
    plasma = false
    raman = false
    n_save = 2
    backend = "numpy_ref"
    """
)
PLANAR_CALL = {
    "length": 0.1,
    "gas": "Ar",
    "pressure": 1.0,
    "wavelength": 800e-9,
    "wavelength_limits": (650e-9, 1000e-9),
    "trange": 1.2e-12,
    "fwhm": 100e-15,
    "energy": 0.5,
    "envelope": True,
    "plasma": False,
    "raman": False,
    "n_save": 2,
    "backend": "numpy_ref",
}

RECTANGULAR = HEADER + textwrap.dedent(
    """\
    length = "1 mm"
    gas = "Ar"
    pressure = "5 bar"
    wavelength = "800 nm"
    wavelength_limits = ["450 nm", "1200 nm"]
    trange = "100 fs"
    fwhm = "30 fs"
    energy = "10 nJ"
    envelope = true
    plasma = false
    raman = false
    n_r = 12
    n_theta = 12
    n_save = 2
    noise = "none"
    backend = "numpy_ref"

    # Both polarisation partners, each a row; a silica-like constant cladding.
    [[build.modes]]
    type = "RectMode"
    half_width_x = "50 um"
    half_width_y = "10 um"
    cladding = { re = 1.45, im = 0.0 }

    [[build.modes]]
    type = "RectMode"
    half_width_x = "50 um"
    half_width_y = "10 um"
    polarisation = "y"
    cladding = { re = 1.45, im = 0.0 }
    """
)
RECTANGULAR_CALL = {
    "length": 1e-3,
    "gas": "Ar",
    "pressure": 5.0,
    "wavelength": 800e-9,
    "wavelength_limits": (450e-9, 1200e-9),
    "trange": 100e-15,
    "fwhm": 30e-15,
    "energy": 1e-8,
    "envelope": True,
    "plasma": False,
    "raman": False,
    "n_r": 12,
    "n_theta": 12,
    "n_save": 2,
    "noise": None,
    "backend": "numpy_ref",
}

CHANNELS = HEADER + textwrap.dedent(
    """\
    modes = [
        { type = "MarcatiliMode", radius = "125 um", m = 1 },
        { type = "MarcatiliMode", radius = "125 um", m = 2 },
    ]
    length = "10 cm"
    gas = "He"
    pressure = "1.2 bar"
    wavelength = "800 nm"
    wavelength_limits = ["300 nm", "2 um"]
    trange = "200 fs"
    fwhm = "30 fs"
    energy = "1 uJ"
    n_save = 3
    seed = 20261101
    backend = "numpy_ref"
    """
)
CHANNELS_CALL = {
    "length": 0.1,
    "gas": "He",
    "pressure": 1.2,
    "wavelength": 800e-9,
    "wavelength_limits": (300e-9, 2e-6),
    "trange": 200e-15,
    "fwhm": 30e-15,
    "energy": 1e-6,
    "n_save": 3,
    "seed": 20261101,
    "backend": "numpy_ref",
}


def _hash(simulation):
    return simulation.metadata().configuration_hash()


def _cases():
    """Return each recipe with the call it stands for, its modes built with a fill."""
    rectangular = [
        RectMode(50e-6, 10e-6, "Ar", 5.0, polarisation=one, cladding=1.45 + 0j)
        for one in ("x", "y")
    ]
    channels = [MarcatiliMode(125e-6, "He", 1.2, m=m) for m in (1, 2)]
    planar = ModeCollection.slab(63.5e-6, (1, 3, 5), "Ar", 1.0).modes
    return {
        "planar": (PLANAR, planar, PLANAR_CALL),
        "rectangular": (RECTANGULAR, rectangular, RECTANGULAR_CALL),
        "channels": (CHANNELS, channels, CHANNELS_CALL),
    }


@pytest.mark.parametrize("name", ["planar", "rectangular", "channels"])
def test_a_waveguide_recipe_builds_the_call_it_stands_for(name):
    text, modes, call = _cases()[name]
    recipe = Recipe.loads(text)
    expected = Simulation.from_waveguide(modes, **call)
    assert _hash(recipe.build()) == _hash(expected)


def test_check_is_clean_and_names_a_mode_s_fill_at_its_path():
    assert Recipe.loads(PLANAR).check() == []
    # The fill is the route's: a mode that names one is refused where it says so.
    recipe = Recipe.loads(CHANNELS.replace("m = 2 }", 'm = 2, gas = "He" }'))
    (issue,) = recipe.check()
    assert issue.parameter == "modes[1]"
    assert "'gas'" in issue.message
    with pytest.raises(RecipeError, match="'gas'"):
        recipe.build()


def test_a_guide_from_a_provider_points_to_the_python_export():
    text = HEADER + 'modes = []\nprovider = { type = "ModalGuide" }\n'
    with pytest.raises(RecipeError, match="Python export") as caught:
        Recipe.loads(text).keywords()
    assert caught.value.parameter == "provider"


def test_the_slab_set_resolves_the_slab_row():
    resolver = parameters("waveguide")["noise"].resolver
    assert resolver is not None
    slab = Recipe.loads(PLANAR).to_dict()["build"]
    assert resolver(slab).value is None
    assert "unit width" in resolver(slab).rule
    channels = Recipe.loads(CHANNELS).to_dict()["build"]
    assert resolver(channels).value == "input"
    assert Recipe.loads(PLANAR).build().noise is None


def test_the_resolved_recipe_rebuilds_the_run_and_each_value_keeps_the_hash():
    recipe = Recipe.loads(CHANNELS)
    simulation = recipe.build()
    simulation.translate()
    document = provenance.resolved_document(recipe, simulation)
    assert _hash(Recipe.from_dict(document).build()) == _hash(simulation)
    authored = recipe.to_dict()["build"]
    written = [key for key in document["build"] if key not in authored]
    table = parameters("waveguide")
    # The record holds the fill's temperature and whether a Raman response was built.
    assert {"temperature", "raman", "rtol", "integrator"} <= set(written)
    for key in written:
        assert table[key].resolvable, key
        one = recipe.to_dict()
        one["build"][key] = document["build"][key]
        if key == "save_planes":
            one["build"].pop("n_save", None)
        assert _hash(Recipe.from_dict(one).build()) == _hash(simulation), key


_PROBE = textwrap.dedent(
    """\
    import json
    import sys
    import warnings

    from solprop.driver.simulation import Simulation

    warnings.simplefilter("ignore")


    def run(self, *args, **kwargs):
        print(json.dumps(self.metadata().configuration_hash()))
        sys.exit(0)


    Simulation.run = run
    """
)
"""Prepended to an exported script: print the hash of what would run, and stop."""


@pytest.mark.medium
@pytest.mark.parametrize("name", ["planar", "rectangular", "channels"])
def test_the_python_export_builds_the_same_run(name, tmp_path):
    recipe = Recipe.loads(_cases()[name][0])
    script = recipe.to_python()
    assert "prop_waveguide(" in script
    path = tmp_path / f"{name}.py"
    path.write_text(_PROBE + script, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == _hash(recipe.build())


def test_a_build_only_keyword_is_exported_through_the_constructor():
    # from_waveguide's memory_budget admits a guide's tables at build; prop_waveguide
    # does not take it.
    recipe = Recipe.loads(PLANAR + 'memory_budget = "8 GiB"\n')
    script = recipe.to_python()
    assert "Simulation.from_waveguide(" in script
    assert "memory_budget='8 GiB'" in script
