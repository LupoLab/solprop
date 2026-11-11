# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep device and backend options in recipe execution, outside scientific hashes."""

import dataclasses
import json
import multiprocessing
import pickle
import subprocess
import sys

import pytest

from recipe.test_free_space import LINE
from recipe.test_recipe import GNLSE
from solprop.recipe import Recipe, RecipeError, parameters, provenance
from solprop.recipe.document import parse_text


@pytest.mark.parametrize("text", [GNLSE, LINE], ids=["gnlse", "line"])
@pytest.mark.parametrize("device", ["none", "cpu", "cuda", "rocm", "metal"])
def test_execution_choices_round_trip_without_changing_scientific_hash(text, device):
    original = Recipe.loads(text).build()
    document = parse_text(text)
    options = {"wide": "df32", "setup_id": "published-identity"}
    document["execution"] = {"device": device, "backend_options": options}
    recipe = Recipe.from_dict(document)
    rebuilt = Recipe.loads(recipe.to_text()).build()
    expected = None if device == "none" else device
    assert rebuilt.settings.device == expected
    assert rebuilt.settings.execution_options().device == expected
    assert rebuilt.settings.backend_options == options
    options["setup_id"] = "changed-origin"
    assert rebuilt.settings.backend_options["setup_id"] == "published-identity"
    assert rebuilt.metadata().configuration_hash() == (
        original.metadata().configuration_hash()
    )
    translated = rebuilt.translate()
    request = rebuilt._run_request()
    rebuilt.settings = dataclasses.replace(
        rebuilt.settings, device="cpu", backend_options={"wide": "binary64"}
    )
    assert rebuilt.translate() is translated
    assert rebuilt._run_request() is request
    resolved = provenance.resolved_document(recipe, rebuilt)
    assert resolved["execution"]["device"] == "cpu"
    assert resolved["execution"]["backend_options"] == {"wide": "binary64"}
    restored = Recipe.from_dict(resolved).build()
    assert restored.settings.device == "cpu"
    assert restored.settings.backend_options == {"wide": "binary64"}
    assert restored.metadata().configuration_hash() == (
        original.metadata().configuration_hash()
    )


@pytest.mark.parametrize("device", ["none", "cuda"])
def test_python_export_builds_the_same_execution_options(device):
    document = parse_text(GNLSE)
    document["execution"] = {
        "device": device,
        "backend_options": {"setup_id": "kept", "wide": "df32"},
        "resume_history": "none",
    }
    recipe = Recipe.from_dict(document)
    # Run the actual exported constructor in a cold Python process, just as a user
    # would; capture settings instead of starting the deliberately unqualified GPU.
    prefix = """import json
from solprop.driver import Simulation
def capture(simulation, _output):
    settings = simulation.settings
    print(json.dumps([settings.device, dict(settings.backend_options),
                      settings.resume_history]))
Simulation.run = capture
"""
    process = subprocess.run(
        [sys.executable, "-c", prefix + recipe.to_python()],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    settings = recipe.build().settings
    assert json.loads(process.stdout) == [
        settings.device,
        settings.backend_options,
        None,
    ]


def test_legacy_mapping_remains_owned_and_resolves_to_one_canonical_location():
    document = parse_text(LINE)
    old = {"setup_id": "legacy"}
    settings = document["build"]["settings"]
    assert isinstance(settings, dict)
    settings["backend_options"] = old
    document["execution"] = {"device": "cpu", "backend_options": dict(old)}
    recipe = Recipe.from_dict(document)
    simulation = recipe.build()
    assert simulation.settings.backend_options == old
    old["setup_id"] = "mutated"
    assert simulation.settings.backend_options == {"setup_id": "legacy"}
    simulation.settings = dataclasses.replace(
        simulation.settings, backend_options={"setup_id": "current"}, device=None
    )
    resolved = provenance.resolved_document(recipe, simulation)
    resolved_settings = resolved["build"]["settings"]
    assert isinstance(resolved_settings, dict)
    assert "backend_options" not in resolved_settings
    assert "device" not in resolved["execution"]
    assert resolved["execution"]["backend_options"] == {"setup_id": "current"}
    assert Recipe.from_dict(resolved).build().settings.backend_options == {
        "setup_id": "current"
    }
    simulation.settings = dataclasses.replace(simulation.settings, backend_options={})
    empty = provenance.resolved_document(recipe, simulation)
    assert "backend_options" not in empty["execution"]
    assert Recipe.from_dict(empty).build().settings.backend_options == {}


def test_conflicting_legacy_and_canonical_options_are_refused_by_name():
    document = parse_text(LINE)
    settings = document["build"]["settings"]
    assert isinstance(settings, dict)
    settings["backend_options"] = {"wide": "df32"}
    document["execution"] = {"backend_options": {"wide": "binary64"}}
    with pytest.raises(
        RecipeError, match=r"conflicts with build\.settings\.backend_options"
    ):
        Recipe.from_dict(document)
    document["execution"] = {"backend_options": {"wide": "df32"}}
    recipe = Recipe.from_dict(document)
    recipe.set("settings.backend_options", {"wide": "binary64"})
    with pytest.raises(
        RecipeError, match=r"conflicts with build\.settings\.backend_options"
    ):
        recipe.build()


def test_device_belongs_in_execution_not_object_route_settings():
    document = parse_text(LINE)
    settings = document["build"]["settings"]
    assert isinstance(settings, dict)
    settings["device"] = "cpu"
    with pytest.raises(RecipeError, match=r"\[execution\]"):
        Recipe.from_dict(document).build()
    assert parameters("execution")["device"].default is None
    first = parameters("execution")["backend_options"].default
    second = parameters("execution")["backend_options"].default
    assert first == second == {}
    assert first is not second


def _spawn_recipe_point(connection, builder, point):
    simulation = builder(point)
    options = simulation.settings.execution_options()
    connection.send((options, "juliacall" in sys.modules))
    connection.close()


def test_spawned_recipe_scan_builder_receives_owned_execution_choices():
    document = parse_text(GNLSE)
    document["execution"] = {
        "device": "metal",
        "backend_options": {"wide": "df32", "setup_id": "kept"},
    }
    document["scan"] = {"name": "power", "peak_power": [5.0, 10.0]}
    scan = Recipe.from_dict(document).scan()
    assert "device" not in scan.keywords
    assert "backend_options" not in scan.keywords
    point = scan.scan.point(1)
    builder = pickle.loads(pickle.dumps(scan.build))
    expected = builder(point).settings.execution_options()
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_spawn_recipe_point, args=(sender, builder, point))
    process.start()
    sender.close()
    try:
        assert receiver.poll(30), "the spawned recipe builder did not return"
        received, imported = receiver.recv()
        assert received == expected
        assert received.backend_options is not expected.backend_options
        assert not imported
        process.join(timeout=20)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        receiver.close()
