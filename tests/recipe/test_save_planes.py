# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A recipe's saved planes and length under SEP 51's one rule.

The refusals a recipe meets are the driver's, reported as a
:class:`~solprop.recipe.RecipeError` against the keyword; a count in an object route's
``[build.settings]`` is ``n_save``; and the resolved recipe writes the run's canonical
positions at the level the route reads them, so a later change of a default count
cannot change the rebuilt run.
"""

import warnings
from typing import Any, cast

import numpy as np
import pytest

from driver.test_tdse_entry import _LENGTH as TDSE_LENGTH
from driver.test_tdse_entry import _sae as tdse_response
from recipe.test_free_space import RADIAL
from recipe.test_recipe import GNLSE
from solprop.driver import Simulation
from solprop.driver import settings as settings_module
from solprop.exceptions import DriverError, MaterialRangeWarning
from solprop.output.hdf5 import open_file
from solprop.recipe import Recipe, RecipeError, provenance

pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

LISTED = 'save_planes = ["0.5 mm", "1 mm"]'
"""The radial recipe's own positions, which each case here replaces."""


def _hash(simulation) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return simulation.metadata().configuration_hash()


def radial(selector: str) -> Recipe:
    """Return the radial recipe with its settings' planes written as `selector`."""
    return Recipe.loads(RADIAL.replace(LISTED, selector))


# ------------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    ("selector", "match", "path"),
    [
        ("save_planes = 80", "write n_save = 80", "settings.save_planes"),
        (
            'save_planes = "0.25 mm"',
            r"pass \[0\.00025\], or n_save",
            "settings.save_planes",
        ),
        ("n_save = 1", "n_save is an integer of at least 2", "settings.n_save"),
        (
            'save_planes = ["1 mm"]\nn_save = 3',
            "give one of n_save and save_planes",
            "settings.save_planes",
        ),
        (
            'save_planes = ["2 mm"]',
            r"raise length \(or grid\.length\)",
            "settings.save_planes",
        ),
    ],
    ids=["count", "lone", "small", "both", "beyond"],
)
def test_an_object_route_recipe_refuses_what_the_driver_refuses(selector, match, path):
    recipe = radial(selector)
    with pytest.raises(RecipeError, match=match) as refused:
        recipe.build()
    assert refused.value.parameter == path
    errors = [issue for issue in recipe.check() if issue.severity == "error"]
    assert [issue.parameter for issue in errors] == [path]


def test_another_refusal_of_the_settings_keeps_its_own_type():
    # Only the refusals SEP 51 states become the recipe's; any other stays the driver's.
    recipe = radial(f"{LISTED}\nnode_budget = -1")
    with pytest.raises(DriverError, match="node_budget must be a positive") as refused:
        recipe.build()
    assert not isinstance(refused.value, RecipeError)


def test_a_keyword_route_recipe_refuses_an_integer_save_planes():
    with pytest.raises(RecipeError, match="write n_save = 80") as refused:
        Recipe.loads(GNLSE.replace("n_save = 3", "save_planes = 80")).build()
    assert refused.value.parameter == "save_planes"


def test_a_keyword_route_recipe_refuses_a_length_its_grid_disagrees_with():
    text = GNLSE.replace(
        'wavelength_limits = ["500 nm", "2 um"]\ntrange = "4 ps"\n',
        'grid = { type = "EnvelopeGrid.from_limits", length = "2 cm", '
        'reference_wavelength = "835 nm", wavelength_limits = ["500 nm", "2 um"], '
        'trange = "4 ps" }\n',
    )
    with pytest.raises(RecipeError, match=r"length = 0\.01 m and grid\.length = 0\.02"):
        Recipe.loads(text).build()


@pytest.mark.parametrize("length", ["", 'length = "none"\n'], ids=["omitted", "none"])
def test_a_keyword_route_recipe_takes_its_length_from_its_grid(length):
    # The route's length is required, None beside a grid: a recipe may leave it out or
    # write none, and the route and the exported script both pass None.
    text = GNLSE.replace('length = "1 cm"\n', length).replace(
        'wavelength_limits = ["500 nm", "2 um"]\ntrange = "4 ps"\n',
        'grid = { type = "EnvelopeGrid.from_limits", length = "2 cm", '
        'reference_wavelength = "835 nm", wavelength_limits = ["500 nm", "2 um"], '
        'trange = "4 ps" }\n',
    )
    assert "grid = " in text
    recipe = Recipe.loads(text)
    assert recipe.keywords()["length"] is None
    assert recipe.build().grid.length == 0.02
    assert "    length=None," in recipe.to_python()


# ------------------------------------------------------------------ the selectors


def test_an_object_route_recipe_takes_a_count_over_its_grid_length():
    simulation = radial("n_save = 81").build()
    planes = np.asarray(simulation.settings.save_planes)
    assert planes.tolist() == np.linspace(0.0, 1e-3, 81)[1:].tolist()
    assert planes[-1] == simulation.grid.length


def test_an_object_route_recipe_without_a_selector_takes_the_default_count():
    simulation = radial("").build()
    assert (
        np.asarray(simulation.settings.save_planes).size + 1
        == settings_module.DEFAULT_N_SAVE
    )


def test_the_count_and_its_positions_build_the_same_run():
    counted = radial("n_save = 3").build()
    listed = radial(LISTED).build()
    assert _hash(counted) == _hash(listed)


# ------------------------------------------------------------- the resolved recipe


@pytest.mark.parametrize(
    "selector", ["n_save = 5", LISTED, 'save_planes = ["0.5 mm"]', ""]
)
def test_the_resolved_recipe_writes_the_canonical_positions_in_the_settings(selector):
    recipe = radial(selector)
    simulation = recipe.build()
    document = provenance.resolved_document(recipe, simulation)
    table = cast("dict[str, object]", document["build"]["settings"])
    assert "n_save" not in table
    assert table["save_planes"] == np.asarray(simulation.settings.save_planes).tolist()
    assert "save_planes" not in document["build"]
    assert _hash(Recipe.from_dict(document).build()) == _hash(simulation)


def test_a_resolved_recipe_does_not_depend_on_the_default_count(monkeypatch):
    recipe = radial("")
    simulation = recipe.build()
    document = provenance.resolved_document(recipe, simulation)
    monkeypatch.setattr(settings_module, "DEFAULT_N_SAVE", 11)
    assert np.asarray(radial("").build().settings.save_planes).size == 10
    assert _hash(Recipe.from_dict(document).build()) == _hash(simulation)


def test_a_keyword_route_s_resolved_recipe_drops_its_count():
    recipe = Recipe.loads(GNLSE)
    simulation = recipe.build()
    document = provenance.resolved_document(recipe, simulation)
    assert "n_save" not in document["build"]
    assert document["build"]["save_planes"] == [0.005, 0.01]
    assert _hash(Recipe.from_dict(document).build()) == _hash(simulation)


# ------------------------------------------------- defaults written out resolved


def tiny_fibre() -> Recipe:
    """Return a ten-micrometre GNLSE recipe with no plane selector."""
    return Recipe.from_keywords(
        "gnlse",
        gamma=0.0,
        length=1e-5,
        betas=[0.0, 0.0, -1e-26],
        wavelength=800e-9,
        wavelength_limits=[500e-9, 1.6e-6],
        trange=200e-15,
        fwhm=20e-15,
        peak_power=1.0,
        backend="numpy_ref",
        noise=None,
        dz_max=1e-6,
        dz_init=1e-6,
    )


def test_a_rebuilt_run_keeps_its_hash_when_the_default_count_changes(
    tmp_path, monkeypatch
):
    path = tmp_path / "run.h5"
    tiny_fibre().build().run(path, progress="none")
    with open_file(path) as stored:
        assert stored.metadata is not None
        recorded = stored.metadata.configuration_hash()
    monkeypatch.setattr(settings_module, "DEFAULT_N_SAVE", 11)
    assert np.asarray(tiny_fibre().build().settings.save_planes).size == 10
    rebuilt = Recipe.from_output(path, resolved=True).build()
    assert _hash(rebuilt) == recorded


def test_the_tdse_capillary_s_defaults_are_written_out_resolved():
    """A recipe cannot hold a TDSE response (``tdse`` has no recipe form), so the
    resolved document is checked at construction: the route's two observations and
    every natural step are written as values, not left to the defaults."""
    keywords: dict[str, Any] = {
        "radius": 256e-6,
        "length": TDSE_LENGTH,
        "gas": "He",
        "pressure": 0.1,
        "wavelength": 800e-9,
        "wavelength_limits": [400e-9, 1.6e-6],
        "trange": 60e-15,
        "fwhm": 6e-15,
        "energy": 1e-12,
        "modes": 1,
        "n_r": 4,
        "noise": None,
        "dz_init": 2.0**-20,
        "dz_max": 2.0**-20,
    }
    recipe = Recipe.from_keywords("capillary", **keywords)
    # Python takes the window's limits as a pair, the recipe as a list.
    python: dict[str, Any] = keywords | {"wavelength_limits": (400e-9, 1.6e-6)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        simulation = Simulation.from_capillary(**python, tdse=tdse_response())
    document = provenance.resolved_document(recipe, simulation)
    build = document["build"]
    assert build["save_planes"] == [TDSE_LENGTH]
    assert build["step_planes"] == 1
    assert "n_save" not in build
