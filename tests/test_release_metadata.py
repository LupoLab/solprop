# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The release records agree: authors, the citation's version and date, typing markers.

``CITATION.cff``, ``.zenodo.json`` and both distributions' ``pyproject.toml`` each name
the authors, and a release commit moves the changelog's ``Unreleased`` section under
its version and date (``docs/dev/releasing.md``); the citation must then carry the same
two. The files are read as text: the citation's few keys need no YAML parser.
"""

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RELEASED = re.compile(r"^## \[?v?(\d+\.\d+\.\d+)\]? - (\d{4}-\d{2}-\d{2})\s*$", re.M)
"""A released changelog section, Keep a Changelog style: ``## [0.1.0] - 2026-11-02``."""


def _citation() -> str:
    return (ROOT / "CITATION.cff").read_text(encoding="utf-8")


def _scalar(text: str, key: str) -> str | None:
    """Return a top-level scalar of the citation, unquoted, or ``None``."""
    found = re.search(rf"^{key}: *(.+?) *$", text, re.M)
    return None if found is None else found[1].strip("\"'")


def _project_authors(path: Path) -> list[str]:
    with path.open("rb") as handle:
        return [author["name"] for author in tomllib.load(handle)["project"]["authors"]]


def test_every_record_names_the_same_authors():
    """The citation, the archive record and both distributions agree, in order."""
    citation = [
        f"{given} {family}"
        for family, given in re.findall(
            r"- family-names: *(.+?)\n +given-names: *(.+?)\n", _citation()
        )
    ]
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    archive = [
        " ".join(reversed(creator["name"].split(", ")))
        for creator in zenodo["creators"]
    ]
    assert citation
    assert archive == citation
    assert _project_authors(ROOT / "pyproject.toml") == citation
    assert _project_authors(ROOT / "native" / "pyproject.toml") == citation


def test_the_citation_names_the_latest_release():
    """Before the first release it names none; afterwards the changelog's newest."""
    citation = _citation()
    released = RELEASED.findall((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    expected = released[0] if released else (None, None)
    assert (_scalar(citation, "version"), _scalar(citation, "date-released")) == (
        expected
    )


def test_both_packages_declare_their_inline_types():
    """Both packages ship ``py.typed`` (PEP 561), so checkers read their annotations."""
    assert (ROOT / "src" / "solprop" / "py.typed").is_file()
    assert (ROOT / "native" / "src" / "solprop_native" / "py.typed").is_file()
