# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The route cards held to solprop's pins, defaults and examples, and a scan setup."""

from solprop.recipe import Recipe, parameters
from solprop.recipe.pins import TARGETS
from solprop.service.explore import PIN_KEYS, PIN_SEARCH

from conftest import TINY
from solprop_gui.registry.routes import CARDS, IMPLIED, card_of


def test_the_registry_agrees_with_solprops_pins_and_defaults():
    required = {key for _, keys, _ in TARGETS.values() for key in keys}
    assert required == set(PIN_KEYS)
    capillary = parameters("capillary")
    assert set(PIN_SEARCH) <= set(capillary)
    for name, value in IMPLIED.items():
        assert capillary[name].default == value


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
        assert card_of(recipe.route, build) is entry, entry.key


SCAN = TINY.replace("n_save = 6", "n_save = 2") + (
    '\n[scan]\nname = "pressure"\npressure = ["0.5 bar", "1 bar"]\n'
)
"""The tiny capillary at two pressures."""
