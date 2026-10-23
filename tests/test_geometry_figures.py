# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The cross-section figures are current, and every section of their page shows its own.

`docs/guide/setup/spatial-decompositions.md` describes each spatial decomposition.
Each section shows an SVG drawn by `docs/guide/figures/geometries.py` and committed
beside it.
Nothing in the documentation build regenerates them, so three things could rot in
silence: a committed SVG could stop being what the script draws from the package; a
section could lose its figure, or show another's; and a figure could be drawn and never
shown. All three are checked here, in the fast lane.

The script keeps the page's section headings beside the figures as data, together with
the sections a later slice will fill (``PENDING``) and the ones that are about every
decomposition (``UNFIGURED``), so a new section on the page has to be declared in one of
the three. The redraw comparison is skipped when the drawing library is absent -- it is
a dev-group dependency -- and the page checks are not, because they need only the
committed files and the script's data.

The script is loaded by path, as `tests/test_structure_figures.py` loads the structure
figures' script.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "guide" / "figures"
PAGE = ROOT / "docs" / "guide" / "setup" / "spatial-decompositions.md"

spec = importlib.util.spec_from_file_location(
    "geometry_figures", FIGURES / "geometries.py"
)
assert spec is not None and spec.loader is not None
geometries = importlib.util.module_from_spec(spec)
# NamedTuple resolves its deferred annotations through sys.modules, so register first.
sys.modules[spec.name] = geometries
spec.loader.exec_module(geometries)

DRAWSVG_MISSING = importlib.util.find_spec("drawsvg") is None
SHOWN = re.compile(r"figures/([a-z_]+)\.svg")


def page_sections() -> dict[str, str]:
    """Return every second-level section of the page, heading to body."""
    parts = re.split(r"^## (.+)$", PAGE.read_text(encoding="utf-8"), flags=re.MULTILINE)
    return {
        heading.strip(): body
        for heading, body in zip(parts[1::2], parts[2::2], strict=True)
    }


def test_every_section_of_the_page_is_declared_by_the_script():
    """A section is a figure's, a later slice's, or about every decomposition."""
    declared = {figure.section for figure in geometries.FIGURES.values()}
    declared |= set(geometries.PENDING) | set(geometries.UNFIGURED)
    assert set(page_sections()) == declared


@pytest.mark.parametrize("name", list(geometries.FIGURES))
def test_every_figure_is_shown_in_its_own_section_and_only_there(name):
    """Each decomposition's section shows exactly its figure; no other section does."""
    sections = page_sections()
    home = geometries.FIGURES[name].section
    assert SHOWN.findall(sections[home]) == [name]
    elsewhere = [
        heading
        for heading, body in sections.items()
        if heading != home and name in SHOWN.findall(body)
    ]
    assert elsewhere == []


@pytest.mark.parametrize("heading", [*geometries.PENDING, *geometries.UNFIGURED])
def test_sections_without_a_figure_show_none(heading):
    """A placeholder gains its figure with its slice, and moves into FIGURES then."""
    assert SHOWN.findall(page_sections()[heading]) == []


def test_the_placeholders_name_the_slice_that_fills_them():
    """A placeholder says which slice replaces it, as the script declares."""
    sections = page_sections()
    for heading, slice_name in geometries.PENDING.items():
        assert slice_name in sections[heading]


def test_every_committed_svg_is_a_figure_the_script_draws():
    """No orphan SVG beside the script, and none missing."""
    committed = {path.stem for path in FIGURES.glob("*.svg")}
    assert committed == set(geometries.FIGURES)


def test_the_two_palettes_define_the_same_colours():
    """A colour named in one theme and not the other would drop out of the figure."""
    assert set(geometries.LIGHT) == set(geometries.DARK)


@pytest.mark.skipif(
    DRAWSVG_MISSING,
    reason="drawsvg is a dev-group dependency, so a --no-dev install cannot redraw",
)
@pytest.mark.parametrize("name", list(geometries.FIGURES))
def test_every_committed_svg_is_what_the_script_draws(name):
    """A figure edited by hand, or a package change it draws, fails until redrawn."""
    committed = (FIGURES / f"{name}.svg").read_text(encoding="utf-8")
    assert geometries.render(name) == committed, (
        f"docs/guide/figures/{name}.svg is stale: run `uv run poe figures`"
    )


@pytest.mark.skipif(DRAWSVG_MISSING, reason="nothing to draw without the library")
def test_every_figure_is_the_standard_width_and_carries_both_palettes():
    """One width, both colour schemes, the shared font stack and nothing in <defs>."""
    for name in geometries.FIGURES:
        svg = (FIGURES / f"{name}.svg").read_text(encoding="utf-8")
        assert f'viewBox="0 0 {geometries.WIDTH} ' in svg
        assert "prefers-color-scheme: dark" in svg
        assert geometries.FONT_STACK in svg
        # drawsvg always writes a <defs> element; anything in it would carry a
        # generated identifier and make the redraw comparison flaky.
        assert svg.count("<defs") == 1
        assert re.search(r"<defs>\s*</defs>", svg)
        assert " id=" not in svg
