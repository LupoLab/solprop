# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The problem classes, held to solprop: a sheet cannot drift from a recipe."""

import math
import warnings
from pathlib import Path
from typing import Any

import pytest
from solprop.recipe import Recipe
from solprop.stats import CATALOGUE

from conftest import TINY
from solprop_gui.classes.catalog import CLASSES, FILAMENT, class_named, class_of
from solprop_gui.classes.problem import (
    AnalysisPage,
    Derived,
    Field,
    ProblemClass,
    Target,
    derived_text,
    derived_value,
    even_planes,
    field_changes,
    field_meta,
    field_value,
    linked_paths,
    negated_value,
    other_keywords,
    page_refusal,
    pin_entry,
    sheet_paths,
    target_text,
)
from solprop_gui.classes.workflow import Progress, Stage, primary_action, stage_open
from solprop_gui.model.recipe import MISSING
from solprop_gui.model.values import inline
from solprop_gui.service.handlers import HANDLERS, describe, describe_output


@pytest.mark.parametrize("problem", CLASSES, ids=lambda p: p.key)
def test_every_row_is_a_value_of_the_class_example(problem: ProblemClass) -> None:
    recipe = Recipe.example(problem.example)
    assert recipe.route == problem.route
    build = recipe.to_dict()["build"]
    description = describe({"route": problem.route})
    for path in sheet_paths(problem):
        meta = field_meta(path, description, build)
        assert meta is not None, path
        assert meta.get("forms"), path


@pytest.mark.parametrize("problem", CLASSES, ids=lambda p: p.key)
def test_the_example_is_recognised_as_its_class(problem: ProblemClass) -> None:
    recipe = Recipe.example(problem.example)
    assert class_of(recipe.route, recipe.to_dict()["build"]) is problem


@pytest.mark.parametrize("problem", CLASSES, ids=lambda p: p.key)
def test_advanced_holds_what_the_sheet_does_not(problem: ProblemClass) -> None:
    description = describe({"route": problem.route})
    others = {entry["name"] for entry in other_keywords(problem, description)}
    own = {
        row.path
        for section in problem.sections
        for row in section.fields
        if row.member is None
    }
    assert others.isdisjoint(own)
    everything = {entry["name"] for entry in description["parameters"]}  # pyright: ignore[reportGeneralTypeIssues]
    tops = {path.split(".")[0] for path in sheet_paths(problem)}
    assert everything - others <= tops


@pytest.mark.parametrize("problem", CLASSES, ids=lambda p: p.key)
def test_statistics_are_solprop_s(problem: ProblemClass) -> None:
    assert set(problem.statistics) <= set(CATALOGUE)


@pytest.mark.parametrize("problem", CLASSES, ids=lambda p: p.key)
def test_derived_numbers_are_in_what_the_kernel_says(problem: ProblemClass) -> None:
    recipe = Recipe.example(problem.example)
    # solprop warns where a band leaves a gas's qualified range; it is not asked here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reply: Any = HANDLERS[problem.numbers]({"text": recipe.to_text()})
        known = {*reply, *reply.get("design", {})}
        if problem.curve is not None:
            curve: Any = HANDLERS[problem.curve](
                {"text": reply["resolved"], "numbers": reply}
            )
            known |= set(curve["numbers"])
    assert {number.key for number in problem.derived} <= known


def test_derived_numbers_read_from_the_summary() -> None:
    assert derived_value({"design": {"zdw": 1.0}}, "zdw") == 1.0
    assert derived_value({"design": {"zdw": "x"}}, "zdw") is None
    assert derived_value(None, "zdw") is None
    assert derived_text(Derived("zdw", "ZDW", "m"), 4.23e-7) == "423 nm"
    assert derived_text(Derived("zdw", "ZDW", "m"), None) == "—"
    assert derived_text(Derived("zdw", "ZDW", "m"), math.inf) == "none"
    assert derived_text(Derived("n", "N", prefixed=False), 3.74) == "3.74"


def test_class_lookup() -> None:
    assert class_named("capillary").route == "capillary"
    with pytest.raises(KeyError):
        class_named("nothing")
    assert class_of("free", {}) is None


def test_page_refusals_say_why() -> None:
    fields = AnalysisPage("a", "A", ("fields",))
    statistics = AnalysisPage("b", "B", ("statistics",))
    modes = AnalysisPage("c", "C", ("modes",))
    ensemble = AnalysisPage("d", "D", ("ensemble",))
    compress = AnalysisPage("e", "E", (), "compression")
    assert page_refusal(fields, {"field_planes": [0]}) is None
    assert "fields" in str(page_refusal(fields, {"field_planes": []}))
    assert page_refusal(statistics, {"z": [0.0]}) is None
    assert "statistics" in str(page_refusal(statistics, {}))
    assert page_refusal(modes, {"labels": ["a", "b"]}) is None
    assert "one mode" in str(page_refusal(modes, {"labels": ["a"]}))
    assert page_refusal(ensemble, {"ensemble": True}) is None
    assert "ensemble" in str(page_refusal(ensemble, {}))
    assert page_refusal(compress, {"refused": {"compression": "No."}}) == "No."
    assert page_refusal(compress, {"refused": {}}) is None
    assert page_refusal(compress, {}) is None


def test_a_typed_target_is_a_pin_solprop_reads() -> None:
    order = Target("soliton_order", "order", None, ("energy",))
    zdw = Target("zdw", "wavelength", "nm", ("pressure",))
    text = TINY.replace('energy = "100 uJ"\n', "")
    text += f"\n[pins]\nenergy = {inline(pin_entry(order, '3'))}\n"
    solved = Recipe.loads(text).resolve()
    assert solved.get("energy") is not None
    text = TINY.replace('pressure = "1 bar"\n', "")
    entry = pin_entry(zdw, "450", ["0.01 bar", "20 bar"])
    text += f"\n[pins]\npressure = {inline(entry)}\n"
    assert float(Recipe.loads(text).resolve().get("pressure")) > 0  # pyright: ignore[reportArgumentType]
    assert target_text(order, {"order": 3.0}) == "3"
    assert target_text(zdw, entry) == "450 nm"
    with pytest.raises(ValueError, match="could not convert"):
        pin_entry(order, "x")


def test_a_finished_run_offers_its_pages(finished_run: Path) -> None:
    capabilities = describe_output({"path": str(finished_run)})
    refused = {page.key: page_refusal(page, capabilities) for page in CLASSES[0].pages}
    assert refused.pop("modes") is not None  # one mode: the recipe is mode-averaged
    assert not any(refused.values()), refused


def test_field_meta_refuses_what_the_route_lacks() -> None:
    described = {"parameters": [{"name": "a"}], "types": {"T": [{"name": "b"}]}}
    assert field_meta("a.b", described, {"a": {"type": "T"}}) == {"name": "b"}
    assert field_meta("a.c", described, {"a": {"type": "T"}}) is None
    assert field_meta("a.b", described, {"a": 1}) is None
    assert field_meta("a[", described, {}) is None
    assert field_meta("a[0]", described, {"a": []}) is None


def test_stages_open_as_the_simulation_goes_on() -> None:
    assert stage_open(Stage.HOME, Progress())
    assert not stage_open(Stage.SETUP, Progress())
    assert stage_open(Stage.SETUP, Progress(has_setup=True))
    assert not stage_open(Stage.RUN, Progress(has_setup=True))
    assert stage_open(Stage.RUN, Progress(running=True))
    assert stage_open(Stage.RUN, Progress(has_run=True))
    assert not stage_open(Stage.ANALYSE, Progress(has_run=True))
    assert stage_open(Stage.ANALYSE, Progress(has_results=True))


def test_the_primary_action_is_what_comes_next() -> None:
    fresh = Progress(has_setup=True, can_run=True)
    assert primary_action(fresh).text == "Run"
    assert primary_action(fresh).enabled
    assert not primary_action(Progress(has_setup=True)).enabled
    # Results opened from Home have no setup to run yet.
    assert not primary_action(Progress(has_results=True)).enabled
    going = Progress(has_setup=True, can_run=True, running=True, has_run=True)
    assert primary_action(going).action == "stop"
    done = Progress(has_setup=True, can_run=True, has_run=True, has_results=True)
    assert primary_action(done).text == "Rerun"


def test_every_link_of_a_class_is_in_its_example() -> None:
    for problem in CLASSES:
        build = Recipe.example(problem.example).to_dict()["build"]
        for section in problem.sections:
            for row in section.fields:
                for pattern in row.also:
                    assert linked_paths(build, pattern), (problem.key, pattern)
                for path in (row.planes_over, row.moves_planes, *row.goes_with):
                    if path is not None:
                        assert linked_paths(build, path), (problem.key, path)


def test_a_switch_takes_an_entry_out_and_brings_it_back_with_today_s_gas() -> None:
    example: Any = Recipe.example("filament").to_dict()["build"]
    row = next(
        row
        for section in FILAMENT.sections
        for row in section.fields
        if row.member == "EnvelopePlasmaResponse"
    )
    assert field_value(row, example) is True
    off: dict[str, Any] = dict(field_changes(FILAMENT, row, False, example, example))
    assert [e["type"] for e in off["responses"]] == ["KerrResponse"]
    assert off["settings.ionisation_field_range"] is MISSING
    build = {**example, "responses": off["responses"], "medium": {"material": "Kr"}}
    assert field_value(row, build) is False
    on: dict[str, Any] = dict(field_changes(FILAMENT, row, True, build, example))
    assert [e["type"] for e in on["responses"]] == [
        "KerrResponse",
        "EnvelopePlasmaResponse",
    ]
    plasma = on["responses"][1]
    assert plasma["gas"] == "Kr"
    assert plasma["rate"]["source"] == "Kr"
    assert (
        on["settings.ionisation_field_range"]
        == example["settings"]["ionisation_field_range"]
    )
    # With no example at hand, nothing comes back; a list absent leaves nothing to keep.
    assert dict(field_changes(FILAMENT, row, True, {}, {}))["responses"] == []
    assert field_value(row, {}) is False


def test_the_filament_s_linked_rows_write_a_recipe_solprop_builds() -> None:
    build = Recipe.example("filament").to_dict()["build"]
    gas = FILAMENT.sections[0].fields[0]
    changes = dict(field_changes(FILAMENT, gas, "Kr", build))
    assert set(changes) == {
        "medium.material",
        "responses[0].chi3.gas",
        "responses[1].gas",
        "responses[1].rate.source",
    }
    length = FILAMENT.sections[0].fields[2]
    moved: dict[str, Any] = dict(field_changes(FILAMENT, length, "20 cm", build))
    assert moved["settings.save_planes"][-1] == "20 cm"
    assert len(moved["settings.save_planes"]) == 80
    assert field_changes(FILAMENT, length, MISSING, build) == [("grid.length", MISSING)]
    assert field_changes(FILAMENT, length, "20 cm", {"grid": {"length": "25 cm"}}) == [
        ("grid.length", "20 cm")
    ]
    planes = Field("settings.save_planes", planes_over="grid.length")
    assert field_value(planes, {}) is MISSING
    assert negated_value(True) is True
    assert even_planes(2, 4.0) == [2.0, 4.0]
