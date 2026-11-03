# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Every distribution takes its one version from the same release tags."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _describe_command(path: str, *table: str) -> list[str]:
    tool = tomllib.loads((ROOT / path).read_text(encoding="utf-8"))["tool"]
    for key in table:
        tool = tool[key]
    return tool["scm"]["git"]["describe_command"]


def test_each_distribution_describes_only_release_tags_the_same_way() -> None:
    raw_options = ("hatch", "version", "raw-options")
    solprop = _describe_command("pyproject.toml", *raw_options)
    assert solprop[-2:] == ["--match", "v*"]
    assert _describe_command("gui/pyproject.toml", *raw_options) == solprop
    assert _describe_command("native/pyproject.toml", "setuptools_scm") == solprop
