# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The user guide's chapter as the window shows it."""

from pathlib import Path

from solprop_gui.help.guide import MANUAL
from solprop_gui.help.manual import chapter_folder, page_title, pages, to_markdown

CHAPTER = Path(__file__).resolve().parents[2] / "docs" / "guide" / "front-end"


def test_the_chapter_is_found_packaged_or_in_a_checkout(tmp_path: Path) -> None:
    assert chapter_folder() is not None  # this checkout's, or a wheel's
    packaged, checkout = tmp_path / "packaged", tmp_path / "checkout"
    assert chapter_folder(packaged, checkout) is None
    checkout.mkdir()
    (checkout / "index.md").write_text("# Index\n")
    assert chapter_folder(packaged, checkout) == checkout
    packaged.mkdir()
    (packaged / "index.md").write_text("# Index\n")
    assert chapter_folder(packaged, checkout) == packaged


def test_the_pages_follow_the_chapter_s_contents() -> None:
    found = pages(CHAPTER)
    assert found == [
        "index",
        "first-run",
        "classes",
        "analysis",
        "scans",
        "design",
        "machines",
    ]
    assert all((CHAPTER / f"{name}.md").is_file() for name in found)
    titles = [page_title((CHAPTER / f"{name}.md").read_text()) for name in found]
    assert titles[0] == "The desktop front end" and all(titles)


def test_every_page_reads_as_plain_markdown_with_working_links() -> None:
    for name in pages(CHAPTER):
        shown = to_markdown((CHAPTER / f"{name}.md").read_text(), name)
        assert "{py:" not in shown and "{toctree}" not in shown, name
        # A link leaves the chapter only for the web.
        assert "](../" not in shown, name
    first = to_markdown((CHAPTER / "first-run.md").read_text(), "first-run")
    assert f"]({MANUAL}guide/setup/recipes.html#pins)" in first
    assert "](analysis.md)" in first  # within the chapter: the window follows it


def test_references_and_links_of_each_kind() -> None:
    text = (
        "{py:mod}`solprop.plotting` and {py:func}`~solprop.tools.capillary_params`; "
        "[web](https://example.org/a.md), [here](#later), [mail](mailto:a@b.c), "
        "[a page](scans.md#its-results), [up](../../introduction/installation.md)"
    )
    shown = to_markdown(text, "index", "https://manual.example/")
    assert shown.startswith("`solprop.plotting` and `capillary_params`; ")
    assert "[web](https://example.org/a.md)" in shown
    assert "[here](#later)" in shown and "[mail](mailto:a@b.c)" in shown
    assert "[a page](scans.md#its-results)" in shown
    assert "[up](https://manual.example/introduction/installation.html)" in shown
    picture = "![Home](images/home.png)"
    assert to_markdown(picture) == picture  # an image keeps its own path
