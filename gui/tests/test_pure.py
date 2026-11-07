# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The layers without solprop calls: messages, outbox, reduction, states, registry."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest
from solprop.service.handlers import HANDLERS
from solprop.service.reduce import stats_row

from solprop_gui.help.guide import guide_url
from solprop_gui.model.workspace import list_runs, new_run_folder, slug
from solprop_gui.registry.pages import LABELS, PAGES, label, page_of
from solprop_gui.registry.routes import CARDS, card, card_of

# ---------------------------------------------------------------------- messages


# ---------------------------------------------------------------------- outbox


# ---------------------------------------------------------------------- reduction


def test_stats_row_totals_numbers_and_skips_the_rest():
    row = stats_row(
        {
            "energy": np.array([[1.0, 2.0]]),
            "zdw": float("nan"),
            "label": object(),
            "count": 3,
        }
    )
    assert row == {"energy": 3.0, "count": 3.0}


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


def test_the_workspace_lists_runs_newest_first(tmp_path: Path):
    assert list_runs(tmp_path / "missing") == []
    old = tmp_path / "a"
    old.mkdir()
    (old / "run.h5").write_bytes(b"")
    new = tmp_path / "b"
    new.mkdir()
    (new / "recipe.toml").write_text("")
    (tmp_path / "empty").mkdir()
    (tmp_path / "loose.txt").write_text("")
    import os  # noqa: PLC0415 - only here

    os.utime(old / "run.h5", (1, 1))
    runs = list_runs(tmp_path)
    assert [run.name for run in runs] == ["b", "a"]
    assert runs[0].output is None and runs[0].recipe == new / "recipe.toml"
    assert runs[1].output == old / "run.h5" and runs[1].recipe is None


# ---------------------------------------------------------------------- registry


def test_the_cards_are_presentation_over_routes():
    averaged = card("capillary-averaged")
    assert averaged.enabled and averaged.route == "capillary"
    assert "mode_average = true" in averaged.blank('A "quoted" title')
    assert 'title = "A \\"quoted\\" title"' in averaged.blank('A "quoted" title')
    assert "title" not in averaged.blank()
    modal = card("capillary-modal")
    assert modal.enabled and "mode_average" not in modal.blank()
    assert "modes = 4" in modal.blank()
    enabled = {entry.key for entry in CARDS if entry.enabled}
    assert enabled == {
        "capillary-averaged",
        "capillary-modal",
        "antiresonant",
        "gnlse",
        "cartesian",
        "line",
        "radial",
    }
    for key in ("planar", "rectangular", "slab"):
        assert card(key).hidden and not card(key).enabled  # not offered for now
    for key in ("step-index", "grin"):  # shown, until recipes can say them
        assert not card(key).hidden and "#47" in str(card(key).reason)
    with pytest.raises(KeyError):
        card("nope")
    assert card_of("capillary", {"mode_average": True}) is averaged
    assert card_of("capillary", {}) is modal  # solprop's default: multimode
    assert card_of("gnlse", {}) is card("gnlse")
    assert card_of("nope", {}) is None
    assert card_of("free", {}) is None  # no space: neither Cartesian nor a line


def test_a_cards_preset_values_are_written_as_toml():
    from solprop_gui.registry.routes import RouteCard  # noqa: PLC0415 - only here

    text = RouteCard("k", "t", "s", "capillary", {"a": True, "b": 2, "c": "x"}).blank()
    assert "a = true" in text and "b = 2" in text and 'c = "x"' in text
    assert "a = false" in RouteCard("k", "t", "s", "r", {"a": False}).blank()


def test_pages_and_labels():
    assert [page.key for page in PAGES][-2:] == ["results", "postprocess"]
    numerics = page_of("numerics")
    assert numerics is not None and numerics.key == "propagate"
    assert page_of("nothing") is None
    assert label("energy") == LABELS["energy"]
    assert label("pressure_floor") == "Pressure floor"


def test_guide_addresses():
    assert guide_url("docs/guide/a.md").endswith("/guide/a.html")
    assert guide_url("guide/a#b", manual="https://x/") == "https://x/guide/a.html#b"


# ---------------------------------------------------------------------- the explorer


def test_the_explorers_axes_and_quantities():
    from solprop.service import explore  # noqa: PLC0415 - only here

    omega = 2 * np.pi * 299_792_458.0 / np.array([1e-6, 5e-7])
    assert explore.axis_values(omega, "frequency") == pytest.approx(
        [299.792458, 599.584916]
    )
    with pytest.raises(KeyError):
        explore.axis_values(omega, "time")
    mode = {
        "neff_minus_one": np.array([1e-5, 2e-5]),
        "beta1": np.array([3.3e-9, 3.4e-9]),
        "beta2": np.array([-1e-29, 1e-29]),
        "beta3": np.array([1e-44, 1e-44]),
        "alpha": np.array([0.1, 0.2]),
        "aeff": 1e-8,
        "gamma": None,
        "phase_mismatch": None,
    }
    for quantity in explore.QUANTITIES:
        values = explore.quantity_values(mode, omega, quantity)
        if quantity.key in ("gamma", "mismatch"):
            assert values is None
        else:
            assert values is not None and values.shape == (2,)
    sets = [
        {"value": "x", "error": "no"},
        {"value": None, "omega": omega, "modes": [{"label": "HE1,1", **mode}]},
    ]
    (row,) = explore.table_rows(sets, 7e-7)
    assert row["label"] == "HE1,1" and row["gamma"] is None
    assert row["neff"] == pytest.approx(1.6e-5)
    assert explore.table_rows([{"error": "no"}], 7e-7) == []
    assert explore.table_rows(sets, 2e-6)[0]["neff"] is None  # outside the band
    assert explore.sweep_values("He, 2 bar") == ["He", "2 bar"]


def test_plot_options_read_typed_limits_and_saved_values():
    from solprop_gui.model import plot_options as po  # noqa: PLC0415 - only here

    with pytest.raises(ValueError):
        po.parse_limit("fast")
    options = po.PlotOptions(axis="f", spectral_limits=(3e14, None))
    assert options.shown_limits("spectral") == (pytest.approx(300.0), None)
    assert options.label() == "Frequency (THz)" and options.describe() == ""
    assert po.to_display("f", po.from_display("f", 2.5)) == pytest.approx(2.5)
    saved = {
        "axis": "f",
        "spectral_limits": [1e14, None],
        "time_limits": "bad",
        "scale": "linear",
        "floor": None,
        "bandpass": [7e-7, 9e-7],
    }
    restored = po.from_mapping(saved)
    assert restored.spectral_limits == (1e14, None)
    assert restored.time_limits == (None, None) and restored.floor == -60.0
    assert restored.bandpass == (7e-7, 9e-7) and restored.scale == "linear"
    assert po.from_mapping({"axis": "omega"}).axis == "wavelength"


def test_every_label_names_a_keyword_of_a_route():
    # The registry's labels must not drift from solprop's keywords (design 12.1).
    keywords: set[str] = set()
    for route in ("capillary", "waveguide", "gnlse"):
        described = HANDLERS["describe"]({"route": route})
        keywords |= {entry["name"] for entry in described["parameters"]}
        for entries in described["types"].values():
            keywords |= {entry["name"] for entry in entries}
    assert set(LABELS) <= keywords
