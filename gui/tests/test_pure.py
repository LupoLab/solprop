# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The layers without solprop calls: the registry, the workspace and the help."""

from datetime import date
from pathlib import Path
from typing import Any

import pytest
from solprop.service.handlers import HANDLERS

from solprop_gui.help.guide import guide_url
from solprop_gui.model.workspace import new_run_folder, slug
from solprop_gui.registry.pages import LABELS, label
from solprop_gui.registry.routes import card_of

# ---------------------------------------------------------------------- messages


# ---------------------------------------------------------------------- states


# ---------------------------------------------------------------------- workspace


def test_run_folders_are_named_by_date_and_title(tmp_path: Path):
    day = date(2026, 11, 2)
    first = new_run_folder(tmp_path / "ws", "RDW in He!", day)
    second = new_run_folder(tmp_path / "ws", "RDW in He!", day)
    assert first.name == "2026-11-02-rdw-in-he"
    assert second.name == "2026-11-02-rdw-in-he-2"
    assert slug(None) == "run" and slug("***") == "run"
    assert len(slug("x" * 100)) == 40


# ---------------------------------------------------------------------- registry


def _card_key(route: str, build: dict[str, object]) -> str | None:
    found = card_of(route, build)
    return None if found is None else found.key


def test_a_recipe_is_named_by_its_route_s_card():
    assert _card_key("capillary", {"mode_average": True}) == "capillary-averaged"
    assert _card_key("capillary", {}) == "capillary-modal"  # solprop's default
    assert _card_key("gnlse", {}) == "gnlse"
    assert _card_key("nope", {}) is None
    assert _card_key("free", {}) is None  # no space: neither Cartesian nor a line


def test_guide_addresses():
    assert guide_url("docs/guide/a.md").endswith("/guide/a.html")
    assert guide_url("guide/a#b", manual="https://x/") == "https://x/guide/a.html#b"


# ---------------------------------------------------------------------- the explorer


def test_plot_options_read_typed_limits():
    from solprop_gui.model import plot_options as po  # noqa: PLC0415 - only here

    with pytest.raises(ValueError):
        po.parse_limit("fast")
    options = po.PlotOptions(axis="f", spectral_limits=(3e14, None))
    assert options.shown_limits("spectral") == (pytest.approx(300.0), None)
    assert options.label() == "Frequency (THz)" and options.describe() == ""


def test_every_label_names_a_keyword_of_a_route():
    # The registry's labels must not drift from solprop's keywords (design 12.1).
    keywords: set[str] = set()
    for route in ("capillary", "waveguide", "gnlse"):
        described: Any = HANDLERS["describe"]({"route": route})
        keywords |= {entry["name"] for entry in described["parameters"]}
        for entries in described["types"].values():
            keywords |= {entry["name"] for entry in entries}
    assert set(LABELS) <= keywords
    assert label("energy") == LABELS["energy"]
    assert label("pressure_floor") == "Pressure floor"


def test_a_remote_run_folder_keeps_a_workspace_given_in_full():
    from datetime import datetime  # noqa: PLC0415 - only here

    from solprop_gui.model.workspace import remote_run_folder  # noqa: PLC0415

    when = datetime(2026, 11, 7, 9, 0, 1)
    found = remote_run_folder("/scratch/runs", "/home/me", None, when)
    assert found.startswith("/scratch/runs/2026-11-07-") and found.endswith("-090001")
    assert remote_run_folder("~", "/home/me/", "t", when).startswith("/home/me/2026")
