# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The problem classes, held to solprop: a sheet cannot drift from a recipe."""

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from solprop.recipe import Recipe
from solprop.recipe.paths import set_path
from solprop.service.handlers import HANDLERS, describe, describe_output
from solprop.stats import CATALOGUE

from conftest import TINY
from solprop_gui.classes.catalog import (
    CLASSES,
    FILAMENT,
    class_named,
    class_of,
    other_class,
)
from solprop_gui.classes.problem import (
    AnalysisPage,
    Derived,
    Field,
    ProblemClass,
    Target,
    derived_text,
    derived_value,
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
                for path in row.goes_with:
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
    assert negated_value(True) is True


def _planes_row(problem: ProblemClass) -> Field:
    return next(
        row
        for section in problem.sections
        for row in section.fields
        if row.positions is not None
    )


def _applied(
    build: dict[str, Any], changes: list[tuple[str, object]]
) -> dict[str, Any]:
    for path, value in changes:
        if value is MISSING:
            head, _, name = path.rpartition(".")
            (build[head] if head else build).pop(name, None)
        else:
            build = set_path(build, path, value)
    return build


@pytest.mark.parametrize("problem", CLASSES, ids=lambda p: p.key)
def test_every_class_offers_both_forms_of_the_saved_planes(
    problem: ProblemClass,
) -> None:
    # SEP 51: the count and the explicit positions, at the route's level.
    row = _planes_row(problem)
    nested = "settings." if row.path.startswith("settings.") else ""
    assert (row.path, row.positions) == (f"{nested}n_save", f"{nested}save_planes")


def test_a_loaded_list_is_shown_as_the_list_and_never_as_its_count() -> None:
    row = _planes_row(FILAMENT)
    example: Any = Recipe.example("filament").to_dict()["build"]
    uniform = example["settings"]["save_planes"]
    assert len(uniform) > 2
    assert field_value(row, example) == uniform
    uneven = {"settings": {"save_planes": ["1 cm", "4 cm", "4.5 cm"]}}
    assert field_value(row, uneven) == ["1 cm", "4 cm", "4.5 cm"]
    assert field_value(row, {"settings": {"save_planes": []}}) == []
    assert field_value(row, {"settings": {"n_save": 81}}) == 81
    assert field_value(row, {}) is MISSING


def test_each_form_of_the_saved_planes_replaces_the_other() -> None:
    row = _planes_row(FILAMENT)
    build: dict[str, Any] = Recipe.example("filament").to_dict()["build"]
    count = field_changes(FILAMENT, row, 81, build)
    assert count == [("settings.n_save", 81), ("settings.save_planes", MISSING)]
    build = _applied(build, count)
    assert build["settings"]["n_save"] == 81
    assert "save_planes" not in build["settings"]
    listed = field_changes(FILAMENT, row, ["5 cm", "8 cm"], build)
    assert listed == [
        ("settings.save_planes", ["5 cm", "8 cm"]),
        ("settings.n_save", MISSING),
    ]
    build = _applied(build, listed)
    assert "n_save" not in build["settings"]
    # Unset, neither is held, and the route's default count applies.
    cleared = field_changes(FILAMENT, row, MISSING, build)
    assert dict(cleared) == {
        "settings.n_save": MISSING,
        "settings.save_planes": MISSING,
    }


def test_a_length_moves_the_length_alone() -> None:
    # A count follows the length by itself; explicit positions stay as entered.
    length = FILAMENT.sections[0].fields[2]
    assert length.path == "grid.length"
    build = Recipe.example("filament").to_dict()["build"]
    assert field_changes(FILAMENT, length, "20 cm", build) == [("grid.length", "20 cm")]


def test_solprop_takes_a_count_and_refuses_a_position_beyond_a_shortened_length() -> (
    None
):
    row = _planes_row(FILAMENT)
    length = FILAMENT.sections[0].fields[2]
    document: Any = Recipe.example("filament").to_dict()
    build = document["build"]
    build = _applied(build, field_changes(FILAMENT, row, 11, build))
    build = _applied(build, field_changes(FILAMENT, length, "10 cm", build))
    simulation = Recipe.from_dict({**document, "build": build}).build()
    planes = np.asarray(simulation.settings.save_planes)
    # Eleven observations over [0, 10 cm], so ten planes after the entrance.
    assert planes.shape == (10,)
    # The end is the length: one rounding of "10 cm" to metres at most.
    assert planes[-1] == pytest.approx(0.1, rel=1e-15)
    build = _applied(build, field_changes(FILAMENT, row, ["5 cm", "12 cm"], build))
    errors = [
        issue
        for issue in Recipe.from_dict({**document, "build": build}).check()
        if issue.severity == "error"
    ]
    assert [issue.parameter for issue in errors] == ["settings.save_planes"]


def test_other_geometry_s_form_comes_from_the_metadata() -> None:
    described = Recipe.example("hollow_rectangular")
    from solprop.service.handlers import describe  # noqa: PLC0415 - only here

    parameters: Any = describe({"route": described.route})["parameters"]
    made = other_class(
        described.route, "Rectangular", described.to_dict()["build"], parameters
    )
    titles = [section.title for section in made.sections]
    assert titles[:2] == ["Geometry", "Medium"]
    shown = {row.path for section in made.sections for row in section.fields}
    assert {"gas", "pressure", "energy"} <= shown
    assert "rtol" not in shown
    assert [page.key for page in made.pages] == ["overview", "plane", "along", "health"]
    assert class_of(described.route, described.to_dict()["build"]) is None
    assert other_class("free", "Free", {}).sections == ()


def test_each_class_lists_examples_of_its_own_starting_from_its_own() -> None:
    shipped = set(Recipe.examples())
    for problem in CLASSES:
        listed = problem.examples or (problem.example,)
        assert listed[0] == problem.example, problem.key
        assert len(set(listed)) == len(listed), problem.key
        for name in listed:
            assert name in shipped, name
            document = Recipe.example(name).to_dict()
            found = class_of(document["recipe"]["route"], document.get("build", {}))
            assert found is problem, (problem.key, name)


def test_two_beams_are_the_cartesian_class_and_one_beam_its_own() -> None:
    for name, key in (
        ("crossing_beams", "cartesian"),
        ("cartesian_beam", "cartesian-beam"),
    ):
        build = Recipe.example(name).to_dict()["build"]
        found = class_of("free", build)
        assert found is not None and found.key == key
    assert [p.key for p in CLASSES if not p.home] == ["cartesian-beam"]
