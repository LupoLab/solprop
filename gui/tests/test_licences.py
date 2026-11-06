# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The licence texts the package carries are the repository's own."""

import tomllib
from pathlib import Path

GUI = Path(__file__).resolve().parents[1]
ROOT = GUI.parent


def test_the_carried_licences_are_copies_of_the_repository_s() -> None:
    assert (GUI / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    carried = (GUI / "LICENSE-CC-BY-4.0.txt").read_bytes()
    assert carried == (ROOT / "LICENSES" / "CC-BY-4.0.txt").read_bytes()


def test_every_licence_named_has_its_text_carried() -> None:
    project = tomllib.loads((GUI / "pyproject.toml").read_text())["project"]
    named = set(project["license"].split(" AND "))
    assert named == {"GPL-3.0-or-later", "OFL-1.1", "CC-BY-4.0"}
    assert all((GUI / path).is_file() for path in project["license-files"])
