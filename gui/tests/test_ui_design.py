# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The explorer, pins and the launch table, offscreen (smoke and screenshot tests)."""

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QTableWidgetItem
from pytestqt.qtbot import QtBot

from solprop_gui.model.recipe import MISSING, RecipeModel
from solprop_gui.service.messages import Reply
from solprop_gui.ui.design_tool import DesignDialog
from solprop_gui.ui.fields import ParameterRow
from solprop_gui.ui.launch import LaunchField, launch_rows
from solprop_gui.ui.main_window import MainWindow
from solprop_gui.ui.pin import PinField
from solprop_gui.views.explorer import ExplorerView
from solprop_gui.views.zdw_chart import ZdwChartView
from test_design import MULTIMODE, PINNED
from test_ui import WAIT_MS, leave, private_settings, shot  # noqa: F401 - a fixture

OMEGA = 2 * np.pi * 299_792_458.0 / np.linspace(2e-6, 3e-7, 64)

PRESSURE_PINS = [
    {"target": "zdw", "stage": "guide", "required": ["wavelength"]},
    {"target": "rdw", "stage": "guide", "required": ["rdw"]},
]


def mode(label: str, scale: float, *, gamma: bool = True) -> dict[str, object]:
    return {
        "label": label,
        "neff_minus_one": scale * 1e-5 * np.ones_like(OMEGA),
        "beta1": 3.3e-9 * np.ones_like(OMEGA),
        "beta2": scale * 1e-29 * np.linspace(1, -1, OMEGA.size),
        "beta3": 1e-44 * np.ones_like(OMEGA),
        "alpha": 0.1 * np.ones_like(OMEGA),
        "gamma": 1e-4 * OMEGA / OMEGA[0] if gamma else None,
        "aeff": 1e-8,
        "zdw": 6e-7,
    }


def test_the_explorer_draws_modes_sweeps_and_a_table(qtbot: QtBot, tmp_path: Path):
    view = ExplorerView()
    qtbot.addWidget(view)
    view.resize(900, 600)
    view.set_route("capillary")
    assert view.sweep() is None
    view.sweep_name.setCurrentIndex(view.sweep_name.findData("pressure"))
    view.sweep_text.setText("1, 2 bar")
    assert view.sweep() == {"name": "pressure", "values": ["1 bar", "2 bar"]}
    view.set_route("capillary")  # the sweep's keyword is kept
    assert view.sweep_name.currentData() == "pressure"
    reply = {
        "pump": 8e-7,
        "sweep": None,
        "sets": [
            {
                "value": None,
                "omega": OMEGA,
                "modes": [mode("HE1,1", 1), mode("HE1,2", 2, gamma=False)],
            }
        ],
    }
    view.set_sets(reply)
    assert len(view.curves) == 2 and view.table.rowCount() == 2
    view.set_rdw(2.5e-7)
    view.set_rdw(2.5e-7)  # the same: not drawn again
    assert view.rdw == 2.5e-7
    assert view.labels() == ["HE1,1", "HE1,2"]
    shot(view, tmp_path, "explorer")
    hit = view.lookup(1000.0, 0.0)
    assert hit is not None and "HE1,1" in hit[2]
    assert ExplorerView().lookup(1.0, 1.0) is None
    view.modes_menu.actions()[1].setChecked(False)
    assert len(view.curves) == 1
    view.modes_menu.actions()[1].setChecked(True)
    for index in range(view.quantity.count()):
        view.quantity.setCurrentIndex(index)
    view.axis.setCurrentIndex(2)
    view.quantity.setCurrentIndex(2)
    swept = {
        "pump": None,
        "sweep": "pressure",
        "sets": [
            {"value": "1 bar", "omega": OMEGA, "modes": [mode("HE1,1", 1)]},
            {"value": "2 bar", "omega": OMEGA, "modes": [mode("HE1,1", 2)]},
            {"value": "x", "error": "not a quantity"},
        ],
    }
    view.set_sets(swept)
    assert len(view.curves) == 2 and "not a quantity" in view.status.text()
    assert view.table.rowCount() == 0
    view.export(tmp_path / "curves.csv")
    lines = (tmp_path / "curves.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + OMEGA.size and "HE1,1, 1 bar" in lines[0]
    view.busy("Solving")
    view.show_error("no modes")
    assert view.curves == [] and view.status.text() == "no modes"


def test_a_pin_is_a_target_a_search_and_a_root(qtbot: QtBot):
    field = PinField("pressure", PRESSURE_PINS, "bar", None)
    qtbot.addWidget(field)
    seen = []
    field.changed.connect(seen.append)
    assert field.search.read() == ["0.01 bar", "20 bar"]  # a starting range
    leave(field.fields["zdw", "wavelength"].number, "500")
    assert seen == [
        {"target": "zdw", "wavelength": "500 nm", "search": ["0.01 bar", "20 bar"]}
    ]
    field.set_roots({"roots": [0.4, 3.0], "quantity": "q", "attainable": None}, "bar")
    field.root.setCurrentIndex(field.root.findData(1))
    assert seen[-1]["root"] == 1 and "2 root" in field.solved.text()
    field.set_roots({"roots": [], "quantity": "ZDW in m", "attainable": [1, 2]}, "bar")
    assert "reaches" in field.solved.text()
    field.set_roots({"roots": [], "quantity": "q", "attainable": None}, "bar")
    assert "not reached" in field.solved.text()
    field.set_value({"target": "rdw", "rdw": "200 nm", "search": [1, 5], "root": 2})
    assert field.target.currentData() == "rdw" and field.root.currentData() == 2
    field.set_solved(0.84, "bar")
    assert field.solved.text() == "Solved: 0.84 bar"
    field.set_solved(None, "bar")
    assert field.solved.text() == ""
    leave(field.fields["rdw", "rdw"].number, "x y")  # refused, not announced
    energy = PinField(
        "energy",
        [{"target": "soliton_order", "stage": "pulse", "required": ["order"]}],
        "J",
        "uJ",
    )
    qtbot.addWidget(energy)
    assert energy.search.isHidden()
    energy.changed.connect(seen.append)
    leave(energy.fields["soliton_order", "order"].number, "3")
    assert seen[-1] == {"target": "soliton_order", "order": 3}


def test_a_pinnable_row_pins_and_resets(qtbot: QtBot):
    meta = {
        "name": "pressure",
        "forms": [{"kind": "number", "unit": "bar"}],
        "summary": "S.",
        "level": "basic",
        "pins": PRESSURE_PINS,
    }
    row = ParameterRow(meta)
    qtbot.addWidget(row)
    assert row.modes == ["number", "pin"] and not row._fits("pin", 1)
    seen = []
    row.edited.connect(lambda *args: seen.append(args))
    row.mode.setCurrentIndex(1)
    leave(row.pin.fields["zdw", "wavelength"].number, "500")
    assert seen[-1][:2] == ("pressure", "pins") and row.is_set
    row.set_pin(seen[-1][2])
    assert row.stack.currentWidget() is row.pin
    row.reset.click()
    assert seen[-1] == ("pressure", "pins", MISSING)
    row.set_value("2 bar")
    assert row.stack.currentWidget() is row.number
    row.reset.click()
    assert seen[-1] == ("pressure", "build", MISSING)


def test_the_launch_table(qtbot: QtBot):
    field = LaunchField()
    qtbot.addWidget(field)
    seen = []
    field.changed.connect(seen.append)
    field.refused.connect(seen.append)
    field.set_labels(["HE1,1:even", "HE1,2:even"])
    field.add_row()
    field.add_row()
    assert seen[-1] == {"HE1,1": 1.0, "HE1,2": 0.0}
    cell(field, 1, 2).setText("1.5")
    assert seen[-1] == {"HE1,1": 1.0, "HE1,2": {"share": 0.0, "phase": 1.5}}
    cell(field, 1, 1).setText("x")
    assert "numbers" in str(seen[-1])
    field.table.setCurrentCell(1, 0)
    field.remove_row()
    field.table.setCurrentCell(-1, -1)
    field.remove_row()
    assert seen[-1] is None
    field.remove_row()  # empty: nothing to remove
    assert field.set_value({"HE1,1": 0.5}) and field.rows() == [("HE1,1", 0.5, 0.0)]
    assert not field.set_value([{"HE1,1": 1.0}])
    assert field.set_value(None) and field.rows() == []
    field.add_row()
    cell(field, 0, 0).setText("")
    assert seen[-1] is None  # a row with no mode is not launched
    assert launch_rows({"HE1,1": "x"}) is None
    row = ParameterRow(
        {"name": "launch", "forms": [{"kind": "mapping"}], "summary": "S."}
    )
    qtbot.addWidget(row)
    assert row.modes == ["table", "text"]
    row.set_value([{"HE1,1": 1.0}])  # a list per pulse: as text
    assert row.stack.currentWidget() is row.text
    row.set_value({"HE1,1": 1.0})
    assert row.stack.currentWidget() is row.launch


def cell(field: LaunchField, row: int, column: int) -> QTableWidgetItem:
    item = field.table.item(row, column)
    assert item is not None
    return item


GUIDE = [
    {
        "name": "pressure",
        "forms": [{"kind": "number", "unit": "bar"}],
        "pins": PRESSURE_PINS,
    },
    {
        "name": "radius",
        "forms": [{"kind": "number", "unit": "m"}],
        "display": "um",
        "pins": PRESSURE_PINS[:1],
    },
]
PULSE = {
    "name": "energy",
    "forms": [{"kind": "number", "unit": "J"}],
    "display": "uJ",
    "pins": [{"target": "soliton_order", "stage": "pulse", "required": ["order"]}],
}


def test_the_design_dialog_solves_and_writes_two_values(qtbot: QtBot, tmp_path: Path):
    pins = {"pressure": {"target": "rdw", "rdw": "200 nm", "search": [1, 5]}}
    dialog = DesignDialog(GUIDE, PULSE, pins)
    qtbot.addWidget(dialog)
    assert dialog.chosen() == ("pressure", "energy")
    assert dialog.fields["pressure"].target.currentData() == "rdw"  # the setup's pin
    asked, written = [], []
    dialog.solve_wanted.connect(asked.append)
    dialog.apply_wanted.connect(written.append)
    dialog.solve()  # the order is missing
    assert asked == [] and "every target" in dialog.outcome.text()
    leave(dialog.fields["energy"].fields["soliton_order", "order"].number, "3")
    dialog.solve()
    assert asked[-1] == {
        "pressure": {"target": "rdw", "rdw": "200 nm", "search": ["1 bar", "5 bar"]},
        "energy": {"target": "soliton_order", "order": 3},
    }
    dialog.show_solution(
        {
            "values": {"pressure": 1.2779114785, "energy": 1.231760842e-4},
            "design": {"soliton_order": 3.0},
            "design_error": "a note",
        }
    )
    assert dialog.apply_button.isEnabled() and "a note" in dialog.outcome.text()
    shot(dialog, tmp_path, "design")
    dialog.apply()
    assert written[-1] == [("pressure", "1.27791 bar"), ("energy", "123.176 uJ")]
    leave(dialog.fields["energy"].fields["soliton_order", "order"].number, "4")
    assert not dialog.apply_button.isEnabled()  # a changed target: solve again
    dialog.apply()
    assert len(written) == 1
    dialog.guide.setCurrentIndex(dialog.guide.findData("radius"))
    assert dialog.chosen()[0] == "radius"
    leave(dialog.fields["radius"].fields["zdw", "wavelength"].number, "x y")
    assert "x y" in dialog.outcome.text()
    dialog.show_error("no root")
    assert dialog.outcome.text() == "no root"


def test_the_zdw_chart_draws_its_contours(qtbot: QtBot, tmp_path: Path):
    view = ZdwChartView()
    qtbot.addWidget(view)
    view.resize(800, 500)
    asked = []
    view.chart_wanted.connect(asked.append)
    view.count.setValue(3)
    view.ask()
    assert asked[-1]["radii"] == pytest.approx([25e-6, 137.5e-6, 250e-6])
    assert asked[-1]["wavelengths"] == [
        "300 nm",
        "400 nm",
        "500 nm",
        "600 nm",
        "800 nm",
    ]
    chart = {
        "radius": [25e-6, 137.5e-6, 250e-6],
        "curves": [
            {"zdw": 4e-7, "pressure": [3.0, 0.2, 0.05]},
            {"zdw": 6e-7, "pressure": [None, 2.0, 0.5]},
        ],
        "setup": {"radius": 125e-6, "pressure": 1.0},
        "error": "no pressure gives 600 nm",
    }
    view.show_chart(chart)
    assert len(view.curves) == 2 and "600 nm" in view.status.text()
    shot(view, tmp_path, "zdw-chart")
    hit = view.lookup(137.5, np.log10(1.5))
    assert hit is not None and "600 nm" in hit[2]
    assert view.lookup(25.0, 0.0) is not None  # the gap is passed over
    view.show_error("nothing")
    assert view.lookup(1.0, 1.0) is None and view.status.text() == "nothing"
    view.wavelengths.setText("")
    assert view.request() is None and "Give" in view.status.text()
    view.wavelengths.setText("400 kg")
    assert view.request() is None
    view.radii.set_value(None)
    assert view.request() is None and "range" in view.status.text()


def _ready(window: MainWindow) -> bool:
    return window.explorer.status.text().startswith("From solprop")


def test_the_window_explores_pins_and_launches(qtbot: QtBot, tmp_path: Path):
    window = MainWindow(workspace=tmp_path / "workspace")
    qtbot.addWidget(window)
    window.show()
    window.confirm = lambda _title, _text: "discard"

    # The standalone explorer: its modes, then a sweep.
    window.open_explorer()
    assert window.views.currentWidget() is window.explorer
    qtbot.waitUntil(lambda: _ready(window), timeout=WAIT_MS)
    assert len(window.explorer.curves) == 4
    qtbot.wait(200)
    shot(window, tmp_path, "explorer")
    window.explorer.sweep_name.setCurrentIndex(
        window.explorer.sweep_name.findData("pressure")
    )
    window.explorer.sweep_text.setText("0.5, 2 bar")
    window.explorer.apply.click()
    qtbot.waitUntil(lambda: len(window.explorer.curves) == 8, timeout=WAIT_MS)
    # The launch table is offered the setup's modes.
    window.show_step("input")
    launch = window.pages["input"].rows["launch", "build"]
    assert launch.launch.labels[:2] == ["HE1,1", "HE1,2"]

    # Pin the pressure: the value goes, the pin comes, and undo brings it back.
    window.set_model(RecipeModel(PINNED), base=None)
    qtbot.waitUntil(
        lambda: window.descriptions.get("capillary") is not None, timeout=WAIT_MS
    )
    row = window.pages["medium"].rows["pressure", "build"]
    qtbot.waitUntil(lambda: row.pin.solved.text().startswith("Solved"), timeout=WAIT_MS)
    assert row.stack.currentWidget() is row.pin
    window.show_step("medium")
    shot(window, tmp_path, "pinned")
    row.pin.find_button.click()
    qtbot.waitUntil(lambda: "root" in row.pin.solved.text(), timeout=WAIT_MS)
    assert window.model is not None
    window.edit("pressure", "build", "1 bar")
    assert window.model.get("pressure", "pins") is MISSING
    window.undo.undo()
    assert window.model.get("pressure", "pins") is not MISSING
    assert window.model.get("pressure") is MISSING
    window.edit("pressure", "pins", MISSING)
    window._show_roots("pressure", _failed())
    assert not row.issue.isHidden()
    window._show_modes(_failed())
    assert "No modes" in window.explorer.status.text()

    # A route without modes, and a setup solprop refuses.
    window.new_from_card("capillary-modal")
    window.show_view("dispersion")
    qtbot.waitUntil(
        lambda: "check finds an error" in window.explorer.status.text(),
        timeout=WAIT_MS,
    )
    window.set_model(
        RecipeModel(MULTIMODE.replace('route = "capillary"', 'route = "gnlse"')),
        base=None,
    )
    window.request_modes(force=True)
    assert "no modes to explore" in window.explorer.status.text()
    window.close()
    assert window.bridge.request("examples") == -1  # closed: no kernel is started


def test_the_window_designs_charts_and_moves_along(qtbot: QtBot, tmp_path: Path):
    window = MainWindow(workspace=tmp_path / "workspace")
    qtbot.addWidget(window)
    window.show()
    window.confirm = lambda _title, _text: "discard"
    window.open_design()  # nothing open yet
    assert "Open a capillary" in window.statusBar().currentMessage()
    window.open_explorer()
    qtbot.waitUntil(lambda: _ready(window), timeout=WAIT_MS)

    # The modes halfway along.
    leave(window.explorer.position.number, "50")
    qtbot.waitUntil(
        lambda: "z = 50 cm" in window.explorer.status.text(), timeout=WAIT_MS
    )

    # The design tool writes the pressure and the energy, in one undo step.
    qtbot.waitUntil(
        lambda: window.descriptions.get("capillary") is not None, timeout=WAIT_MS
    )
    window.explorer.design_button.click()
    dialog = window.design_dialog
    assert dialog is not None
    leave(dialog.fields["pressure"].fields["zdw", "wavelength"].number, "500")
    leave(dialog.fields["energy"].fields["soliton_order", "order"].number, "3")
    dialog.solve()
    qtbot.waitUntil(dialog.apply_button.isEnabled, timeout=WAIT_MS)
    dialog.apply()
    assert window.model is not None
    assert window.model.get("pressure").startswith("1.27")  # pyright: ignore[reportAttributeAccessIssue]
    window.undo.undo()
    assert window.model.get("pressure") == "1 bar"
    assert window.model.get("energy") == "100 uJ"
    leave(dialog.fields["pressure"].fields["zdw", "wavelength"].number, "50")
    dialog.solve()
    qtbot.waitUntil(lambda: "No design" in dialog.outcome.text(), timeout=WAIT_MS)
    window.open_design()  # again: a fresh dialog
    assert window.design_dialog is not dialog
    window.design_dialog.close()  # pyright: ignore[reportOptionalMemberAccess]

    # The ZDW chart of the setup's fill.
    window.show_view("chart")
    window.chart.count.setValue(3)
    window.chart.wavelengths.setText("500 nm")
    window.chart.draw_button.click()
    qtbot.waitUntil(lambda: len(window.chart.curves) == 1, timeout=WAIT_MS)
    window._show_tool("chart", _failed())
    assert "No chart" in window.chart.status.text()

    # Routes without either.
    window.set_model(
        RecipeModel(MULTIMODE.replace('route = "capillary"', 'route = "gnlse"')),
        base=None,
    )
    window.request_chart({"radii": [1e-4], "wavelengths": ["500 nm"]})
    assert "capillary setup" in window.chart.status.text()
    qtbot.waitUntil(
        lambda: window.descriptions.get("gnlse") is not None, timeout=WAIT_MS
    )
    window.open_design()
    assert "no design targets" in window.statusBar().currentMessage()
    window.close()


def _failed() -> Reply:
    return Reply("modes", 0, 0, error="it failed")
