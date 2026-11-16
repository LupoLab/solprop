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
from typing import cast

import numpy as np
import pytest

from recipe.test_free_space import RADIAL
from recipe.test_recipe import GNLSE
from solprop.driver import settings as settings_module
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
