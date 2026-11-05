# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The problem classes, held to solprop: a sheet cannot drift from a recipe."""

import warnings

import pytest
from solprop.recipe import Recipe
from solprop.stats import CATALOGUE

from solprop_gui.classes.catalog import CLASSES, class_named, class_of
from solprop_gui.classes.problem import (
    AnalysisPage,
    ProblemClass,
    field_meta,
    other_keywords,
    page_refusal,
    sheet_paths,
)
from solprop_gui.classes.workflow import Progress, Stage, primary_action, stage_open
from solprop_gui.service.handlers import describe, summarise


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
    assert others.isdisjoint(sheet_paths(problem))
    everything = {entry["name"] for entry in description["parameters"]}  # pyright: ignore[reportGeneralTypeIssues]
    tops = {path.split(".")[0] for path in sheet_paths(problem)}
    assert everything - others <= tops


@pytest.mark.parametrize("problem", CLASSES, ids=lambda p: p.key)
def test_statistics_are_solprop_s(problem: ProblemClass) -> None:
    assert set(problem.statistics) <= set(CATALOGUE)


@pytest.mark.parametrize("problem", CLASSES, ids=lambda p: p.key)
def test_derived_numbers_are_in_the_summary(problem: ProblemClass) -> None:
    recipe = Recipe.example(problem.example)
    # solprop warns where a band leaves a gas's qualified range; it is not asked here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = summarise({"text": recipe.to_text()})
    known = {*summary, *summary["design"]}  # pyright: ignore[reportGeneralTypeIssues]
    assert {number.key for number in problem.derived} <= known


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
    assert page_refusal(statistics, {"statistics_planes": [0]}) is None
    assert "statistics" in str(page_refusal(statistics, {}))
    assert page_refusal(modes, {"labels": ["a", "b"]}) is None
    assert "one mode" in str(page_refusal(modes, {"labels": ["a"]}))
    assert page_refusal(ensemble, {"ensemble": True}) is None
    assert "ensemble" in str(page_refusal(ensemble, {}))
    assert page_refusal(compress, {"refused": {"compression": "No."}}) == "No."
    assert page_refusal(compress, {"refused": {}}) is None
    assert page_refusal(compress, {}) is None


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
