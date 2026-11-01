# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Previews: the mode set and the input pulse, before the run (SEP 41, S18)."""

import warnings

import numpy as np
import pytest

from solprop.modes.base import beta2, zdw
from solprop.recipe import Recipe, RecipeError


@pytest.fixture(autouse=True)
def _quiet():
    """Ignore the warnings the examples' builds give by design."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_the_mode_preview_is_the_run_s_mode_set_at_the_launch():
    recipe = Recipe.example("rdw")
    preview = recipe.modes()
    simulation = recipe.build()
    assert preview.z == 0.0
    assert [mode.label for mode in preview.modes] == [
        mode.channel_label.display_label for mode in simulation.modes
    ]
    first = simulation.modes[0]
    assert np.array_equal(preview.modes[0].beta2, beta2(first, preview.omega))
    limits = (
        2 * np.pi * 299_792_458.0 / preview.omega[-1],
        2 * np.pi * 299_792_458.0 / preview.omega[0],
    )
    assert preview.modes[0].zdw == pytest.approx(zdw(first, wavelength_limits=limits))
    assert np.all(np.diff(preview.omega) > 0)
    assert preview.modes[0].aeff > preview.modes[1].aeff


def test_the_mode_preview_is_taken_where_asked():
    recipe = Recipe.example("rdw")  # a pressure gradient: the modes change along z
    entrance, exit_ = recipe.modes(), recipe.modes(z=1.5)
    assert exit_.z == 1.5
    assert not np.array_equal(entrance.modes[0].beta, exit_.modes[0].beta)


def test_a_route_without_modes_has_no_mode_preview():
    with pytest.raises(RecipeError, match="builds no mode set"):
        Recipe.example("soliton").modes()


@pytest.mark.parametrize(
    ("name", "energy", "peak_power"),
    [("rdw", 275e-6, None), ("soliton", None, 96.79090909090908)],
)
def test_the_input_preview_carries_the_recipe_s_pulse(name, energy, peak_power):
    preview = Recipe.example(name).preview_input()
    if energy is not None:
        assert preview.energy == pytest.approx(energy, rel=1e-4)
    if peak_power is not None:
        assert preview.peak_power == pytest.approx(peak_power, rel=1e-6)
    assert preview.time.shape == preview.power.shape


def test_the_input_preview_has_no_noise_and_draws_nothing():
    recipe = Recipe.example("rdw")
    first, second = recipe.preview_input(), recipe.preview_input()
    assert np.array_equal(first.power, second.power)
    document = recipe.to_dict()
    document["build"]["noise"] = "none"
    document["build"].pop("seed", None)
    quiet = Recipe.from_dict(document).preview_input()
    assert np.array_equal(first.spectrum, quiet.spectrum)


def test_a_preview_solves_the_pins_first():
    document = Recipe.example("soliton_self_compression").to_dict()
    del document["build"]["energy"]
    document["pins"] = {"energy": {"target": "soliton_order", "order": 4}}
    recipe = Recipe.from_dict(document)
    solved = recipe.resolve().get("energy")
    assert isinstance(solved, float)
    assert recipe.preview_input().energy == pytest.approx(solved, rel=1e-4)
