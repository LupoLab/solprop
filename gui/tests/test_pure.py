# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The layers without solprop calls: messages, outbox, reduction, states, registry."""

import threading
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from solprop_gui.help.guide import guide_url
from solprop_gui.model.workspace import list_runs, new_run_folder, slug
from solprop_gui.registry.pages import LABELS, PAGES, label, page_of
from solprop_gui.registry.routes import CARDS, card, card_of
from solprop_gui.service.messages import PROTOCOL, Control, Event, Reply, Request
from solprop_gui.service.outbox import Outbox
from solprop_gui.service.reduce import Display, Reducer, bin_max, stats_row
from solprop_gui.service.states import ACTIVE, ExecutionState, advance

# ---------------------------------------------------------------------- messages


def test_messages_carry_their_identity():
    request = Request("check", {"text": "x"}, request_id=4, revision=2, key="k")
    assert request.protocol == PROTOCOL
    assert Reply("check", 4, 2, value=[]).ok
    assert not Reply("check", 4, 2, error="no").ok
    assert not Reply("check", 4, 2, superseded=True).ok
    assert Event("r", "progress").replaceable
    assert not Event("r", "row").replaceable
    assert all(
        Event("r", kind).terminal for kind in ("finished", "stopped", "failed", "lost")
    )
    assert Control("stop").from_plane == 0


# ---------------------------------------------------------------------- outbox


def test_outbox_coalesces_progress_and_keeps_order():
    box = Outbox(capacity=4)
    box.put(Event("r", "row", {"index": 0}))
    box.put(Event("r", "progress", {"z": 1}))
    box.put(Event("r", "progress", {"z": 2}))
    box.put(Event("other", "progress", {"z": 3}))
    assert len(box) == 3
    taken = [box.get(timeout=0) for _ in range(3)]
    assert [(e.run_id, e.kind, e.data) for e in taken if e is not None] == [
        ("r", "row", {"index": 0}),
        ("r", "progress", {"z": 2}),
        ("other", "progress", {"z": 3}),
    ]
    assert box.get(timeout=0) is None


def test_outbox_drops_rows_when_full_but_never_a_terminal_event():
    box = Outbox(capacity=2)
    assert box.put(Event("r", "row", {"index": 0}))
    assert box.put(Event("r", "row", {"index": 1}))
    assert not box.put(Event("r", "row", {"index": 2}))
    assert box.dropped == 1
    assert box.put(Event("r", "finished"))
    box.close()
    assert box.closed
    assert not box.put(Event("r", "row", {"index": 3}))
    kinds = []
    while (event := box.get(timeout=0)) is not None:
        kinds.append(event.kind)
    assert kinds == ["row", "row", "finished"]


def test_outbox_wakes_a_waiting_reader():
    box = Outbox()
    got: list[Event | None] = []
    reader = threading.Thread(target=lambda: got.append(box.get()))
    reader.start()
    box.put(Event("r", "row"))
    reader.join(5)
    assert got and got[0] is not None and got[0].kind == "row"


def test_outbox_refuses_a_capacity_below_one():
    with pytest.raises(ValueError, match="at least 1"):
        Outbox(capacity=0)


# ---------------------------------------------------------------------- reduction


def test_bin_max_keeps_each_bins_peak_and_fills_empty_bins():
    edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    x = np.array([0.5, 0.6, 0.7, 3.5, 4.0])  # bins 1 and 2 are empty
    y = np.array([1.0, 9.0, 2.0, 4.0, 6.0])
    out = bin_max(x, y, edges)
    assert out[0] == 9.0  # the narrow peak survives
    assert out[3] == 6.0  # the upper edge belongs to the last bin
    assert out[1] == pytest.approx(np.interp(1.5, x, y))
    # Outside the samples' span an empty bin is zero.
    assert bin_max(np.array([0.1]), np.array([5.0]), edges)[2] == 0.0


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


def test_the_reducer_fixes_its_bins_on_the_first_plane():
    reducer = Reducer(Display(n_wavelength=4, n_time=3))
    with pytest.raises(RuntimeError, match="first plane"):
        reducer.axes()
    lam = np.array([np.inf, -1.0, 1e-6, 2e-6, 3e-6])
    density = np.array([7.0, 7.0, 1.0, 2.0, 3.0])
    t = np.linspace(-1e-13, 1e-13, 11)
    reduced, power = reducer.reduce(lam, density, t, np.ones(11))
    assert reduced.dtype == np.float32 and reduced.shape == (4,)
    assert power.shape == (3,)
    assert float(reduced.max()) == 3.0  # the non-physical wavelengths are left out
    axes = reducer.axes()
    assert axes["wavelength"][0] > 1e-6 and axes["time"].size == 3
    # The second plane reuses the bins.
    reducer.reduce(lam * 2, density, t * 2, np.ones(11))
    assert np.array_equal(reducer.axes()["wavelength"], axes["wavelength"])


def test_a_display_takes_the_runs_window_only_where_it_has_none():
    keywords = {"wavelength_limits": (2e-6, 2e-7)}
    assert Display().within(keywords).wavelength_range == (2e-7, 2e-6)
    fixed = Display(wavelength_range=(1e-7, 1e-6))
    assert fixed.within(keywords) is fixed
    assert Display().within({}).wavelength_range is None


# ---------------------------------------------------------------------- states


def test_the_state_machine_follows_a_run():
    state = ExecutionState.QUEUED
    for event, expected in [
        ("launch", ExecutionState.STARTING),
        ("progress", ExecutionState.STARTING),
        ("started", ExecutionState.RUNNING),
        ("stop", ExecutionState.STOPPING),
        ("stopped", ExecutionState.STOPPED),
        ("started", ExecutionState.STOPPED),  # an ended run stays ended
    ]:
        state = advance(state, event)
        assert state == expected
    assert advance(ExecutionState.STOPPING, "finished") == ExecutionState.COMPLETE
    assert advance(ExecutionState.RUNNING, "lost") == ExecutionState.LOST
    assert advance(ExecutionState.STARTING, "failed") == ExecutionState.FAILED
    assert ExecutionState.COMPLETE not in ACTIVE


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
        "waveguide",
        "gnlse",
    }
    with pytest.raises(KeyError):
        card("nope")
    assert card_of("capillary", {"mode_average": True}) is averaged
    assert card_of("capillary", {}) is modal  # solprop's default: multimode
    assert card_of("gnlse", {}) is card("gnlse")
    assert card_of("free", {}) is None


def test_a_cards_preset_values_are_written_as_toml():
    from solprop_gui.registry.routes import RouteCard  # noqa: PLC0415 - only here

    text = RouteCard("k", "t", "s", "capillary", {"a": True, "b": 2, "c": "x"}).blank()
    assert "a = true" in text and "b = 2" in text and 'c = "x"' in text
    assert "a = false" in RouteCard("k", "t", "s", "r", {"a": False}).blank()


def test_pages_and_labels():
    assert [page.key for page in PAGES][-1] == "results"
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
    from solprop_gui.registry import explore  # noqa: PLC0415 - only here

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
