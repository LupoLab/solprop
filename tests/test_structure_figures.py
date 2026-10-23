# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The structure figures name things that exist, and the committed SVGs are current.

`docs/dev/structure.md` shows six SVGs drawn by `docs/dev/figures/build.py` and
committed beside it. Nothing in the documentation build regenerates them, so two things
could rot in silence: a box could name a module, a contract field or a seam method the
code no longer has, and a committed SVG could stop being what the script draws. Both are
checked here, in the fast lane.

The script keeps the names it draws as data, so these tests read that data rather than
the SVG text. The redraw comparison is skipped when the drawing library is absent -- it
is a dev-group dependency, so an install without that group can still run the suite --
and the name checks are not, because they need only the package and the committed files.

The helper is loaded by path, the pattern `tests/test_accuracy_figures.py` uses for
`docs/dev/accuracy_figures.py`.
"""

import dataclasses
import importlib.util
import re
import sys
from pathlib import Path

import pytest

from solprop.backends.driver import StepDriver
from solprop.backends.numpy_ref import passes
from solprop.backends.protocol import Backend
from solprop.output import schema
from solprop.stats import CATALOGUE
from solprop.stepper import schedule as schedule_module
from solprop.stepper.controller import PIController
from solprop.translate.problem import KernelProblem

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "dev" / "figures"
PACKAGE = ROOT / "src" / "solprop"
PAGE = ROOT / "docs" / "dev" / "structure.md"

spec = importlib.util.spec_from_file_location("structure_figures", FIGURES / "build.py")
assert spec is not None and spec.loader is not None
build = importlib.util.module_from_spec(spec)
# NamedTuple resolves its deferred annotations through sys.modules, so register first.
sys.modules[spec.name] = build
spec.loader.exec_module(build)

DRAWSVG_MISSING = importlib.util.find_spec("drawsvg") is None
BUILT = [
    path
    for group in build.MODULES.values()
    for module in group
    for path in module.files()
]
PLANNED = [
    path
    for group in build.PLANNED.values()
    for module in group
    for path in module.files()
]


@pytest.mark.parametrize("path", [*BUILT, *(m.path for m in build.PACKAGE_WIDE)])
def test_every_module_a_figure_names_exists(path):
    """A box in the layers figure names a module of the package, or it is a lie."""
    assert (PACKAGE / path).exists(), f"{path} is drawn solid but is not in src/solprop"


@pytest.mark.parametrize("path", PLANNED)
def test_every_planned_module_is_still_unwritten(path):
    """A dashed box promises code that is not there; when it lands, make it solid."""
    assert not (PACKAGE / path).exists(), (
        f"{path} now exists: move it from PLANNED to MODULES and redraw"
    )


def test_the_layers_figure_draws_the_whole_package():
    """No module of the package is missing from the figure that claims to show it.

    Private modules are excluded: the figure shows the public surface, and
    ``processing/_examples.py`` is a fixture builder for the doctests.
    """
    drawn = {*BUILT, *(module.path for module in build.PACKAGE_WIDE)}
    on_disk = {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
        if not path.name.startswith("_") or path.name == "__init__.py"
    }
    on_disk -= {
        path
        for path in on_disk
        if path.endswith("__init__.py") and path != "__init__.py"
    }
    assert on_disk - drawn == set()


@pytest.mark.parametrize("section", build.CONTRACT_SECTIONS)
def test_every_contract_section_names_fields_of_the_problem(section):
    """Each numbered section of figure 3 names fields `KernelProblem` really has."""
    fields = {field.name for field in dataclasses.fields(KernelProblem)}
    assert set(section.fields) <= fields


def test_the_contract_figure_covers_every_field():
    """Between them the eight sections account for all eleven fields."""
    drawn = {name for section in build.CONTRACT_SECTIONS for name in section.fields}
    assert drawn == {field.name for field in dataclasses.fields(KernelProblem)}


def test_the_representations_are_the_ones_the_contract_offers():
    """The inset's four representation names are the literal the contract uses."""
    assert set(build.PHI_REPRESENTATIONS) == {
        "constant",
        "expansion",
        "table",
        "separable",
    }
    problem_source = (PACKAGE / "translate" / "linop.py").read_text()
    for kind in build.PHI_REPRESENTATIONS:
        assert f'"{kind}"' in problem_source


@pytest.mark.parametrize("name", [name for name, _ in build.SEAM_METHODS])
def test_every_seam_method_is_on_the_protocol(name):
    """Figure 5's seam draws the methods `Backend` declares, and all of them."""
    assert hasattr(Backend, name)


def test_the_seam_figure_draws_the_whole_protocol():
    """A method added to the protocol has to reach the figure."""
    declared = {
        name
        for name in vars(Backend)
        if not name.startswith("_") and callable(vars(Backend)[name])
    }
    assert declared == {name for name, _ in build.SEAM_METHODS}


@pytest.mark.parametrize(
    "name",
    [name for step in build.STAGE_PASSES for name in step.functions]
    + [name for _, _, funcs in build.RESPONSE_COLUMN for name in funcs]
    + [name for name, _ in build.STEP_PASSES],
)
def test_every_pass_a_figure_names_is_a_reference_pass(name):
    """Figure 4 labels each pass with a function of `numpy_ref.passes`."""
    assert name in passes.__all__


def test_the_stage_figure_draws_every_reference_pass():
    """A pass added to the oracle has to reach the figure; there is nowhere to hide."""
    drawn = (
        {name for step in build.STAGE_PASSES for name in step.functions}
        | {name for _, _, funcs in build.RESPONSE_COLUMN for name in funcs}
        | {name for name, _ in build.STEP_PASSES}
    )
    assert set(passes.__all__) == drawn


@pytest.mark.parametrize(
    "name", [name for _, members in build.DRIVER_PARTS for name in members]
)
def test_every_driver_part_is_a_member_of_the_step_driver(name):
    """Figure 5 breaks `StepDriver` into parts, and each part names real members."""
    assert hasattr(StepDriver, name)


@pytest.mark.parametrize("name", build.CONTROLLER_PARTS)
def test_every_controller_method_drawn_exists(name):
    """The controller box names methods `PIController` has."""
    assert hasattr(PIController, name)


@pytest.mark.parametrize("name", build.SCHEDULE_PARTS)
def test_every_schedule_function_drawn_exists(name):
    """The schedule box names functions `stepper.schedule` exports."""
    assert name in schedule_module.__all__


@pytest.mark.parametrize("name", build.OUTPUT_STATS)
def test_every_statistic_drawn_is_in_the_catalogue(name):
    """The ``stats/`` box of figure 6 names statistics a run can actually save."""
    assert name in CATALOGUE


def test_the_output_figure_names_the_history_columns_the_schema_defines():
    """The ``step_history/`` box lists exactly `schema.HISTORY_FIELDS`."""
    line = next(
        line
        for group in build.OUTPUT_GROUPS
        if group.name == "step_history/"
        for line in group.lines
    )
    for column in schema.HISTORY_FIELDS:
        assert column in line


def test_the_output_figure_names_the_datasets_the_writer_creates():
    """Every box of the file is a key `output/hdf5.py` really writes."""
    source = (PACKAGE / "output" / "hdf5.py").read_text()
    for group in build.OUTPUT_GROUPS:
        key = group.name.rstrip("/")
        assert f'"{key}"' in source, f"{key} is drawn but hdf5.py never creates it"


def test_the_resume_key_is_the_method_the_schema_defines():
    """Figure 6 names the digest by the method that computes it."""
    _, method = build.RESUME_KEY.rsplit(".", 1)
    assert hasattr(schema.RunMetadata, method)


def test_the_page_shows_exactly_the_figures_the_script_draws():
    """No orphan SVG, and no figure drawn and never shown."""
    shown = set(re.findall(r"\(figures/([a-z_]+)\.svg\)", PAGE.read_text()))
    assert shown == set(build.FIGURES)


def test_every_figure_the_script_draws_is_committed():
    """The page includes the files, so a missing one is a docs build failure too."""
    for name in build.FIGURES:
        assert (FIGURES / f"{name}.svg").exists()


@pytest.mark.skipif(
    DRAWSVG_MISSING,
    reason="drawsvg is a dev-group dependency, so a --no-dev install cannot redraw",
)
@pytest.mark.parametrize("name", list(build.FIGURES))
def test_every_committed_svg_is_what_the_script_draws(name):
    """A figure edited without `poe figures` is a failure, not a surprise in review."""
    committed = (FIGURES / f"{name}.svg").read_text(encoding="utf-8")
    assert build.render(name) == committed, (
        f"docs/dev/figures/{name}.svg is stale: run `uv run poe figures`"
    )


@pytest.mark.skipif(DRAWSVG_MISSING, reason="nothing to draw without the library")
def test_every_figure_is_the_standard_width_and_carries_both_palettes():
    """One width so the page renders them at one scale, and one style in each."""
    for name in build.FIGURES:
        svg = (FIGURES / f"{name}.svg").read_text(encoding="utf-8")
        assert f'viewBox="0 0 {build.WIDTH} ' in svg
        assert "prefers-color-scheme: dark" in svg
        assert build.FONT_STACK in svg


def test_the_two_palettes_define_the_same_colours():
    """A colour named in one theme and not the other would drop out of the figure."""
    assert set(build.LIGHT) == set(build.DARK)
    for letter in build.LAYER_LETTERS:
        assert f"{letter.lower()}-fill" in build.LIGHT
        assert f"{letter.lower()}-edge" in build.LIGHT
