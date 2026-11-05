# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Post-process step's compressor: its stages, and the kernel's chain on a run."""

import math
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit, QSpinBox
from pytestqt.qtbot import QtBot
from solprop.exceptions import SolpropError

from solprop_gui.model.compressor import (
    FIELDS,
    KINDS,
    Field,
    found_values,
    new_stage,
    payload,
)
from solprop_gui.service import handlers
from solprop_gui.ui.main_window import MainWindow
from solprop_gui.ui.postprocess import StageEditor
from test_results import beam, run  # noqa: F401 - fixtures
from test_ui import WAIT_MS, private_settings, shot  # noqa: F401 - a fixture


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    """Ignore the material-range warnings of the wide bands."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def ask(kind: str, body: Mapping[str, object]) -> Any:
    return handlers.HANDLERS[kind](body)


def chain(path: str, *stages: Any, **extra: object) -> Any:
    body = {"path": path, "index": -1, "stages": payload(stages), **extra}
    return ask("compress_chain", body)


def test_a_stage_is_described_edited_and_fixed_at_what_was_found():
    for kind in KINDS:
        for auto in (False, True):
            stage = new_stage(kind, auto=auto)
            assert stage.describe().startswith(KINDS[kind])
            shown = [spec.name for spec in FIELDS[kind] if spec.shown(auto)]
            assert shown and all(name in stage.values for name in shown)
    assert Field("x", "x").shown(True) and Field("x", "x").shown(False)
    with pytest.raises(KeyError):
        new_stage("lens")
    fixed = new_stage("taylor")
    assert found_values(fixed, {"values": {"gdd": 1.0}}) == [fixed]
    # Mirrors with wedges: the wedge's thickness found, or the guide's start.
    mirrors = new_stage("mirrors", auto=True).with_value("wedge", "caf2")
    _, wedge = found_values(mirrors, {})
    assert wedge.values["medium"] == "caf2" and wedge.values["thickness"] == 1e-3
    blank = new_stage("prisms").with_value("separation", None)
    assert "R - cm" in blank.describe()


def test_the_catalogue_names_glasses_gases_and_mirrors():
    found = ask("optics_catalogue", {})
    assert "fused silica" in found["glasses"] and "Air" in found["gases"]
    assert found["media"] == found["glasses"] + found["gases"]
    assert "PC70" in found["mirrors"]


def test_fixed_elements_are_applied_in_order(run: str):  # noqa: F811 - a fixture
    air = new_stage("material").with_value("medium", "Air")
    air = air.with_value("thickness", 0.5)  # no pressure: one bar
    stages = [
        new_stage("taylor").with_value("gdd", -10e-30),
        air,
        new_stage("material").with_value("pressure", 2.0),  # a glass: ignored
        new_stage("mirrors"),
        new_stage("prisms"),
        new_stage("gratings").with_value("incidence", 0.4),
        new_stage("gratings").with_value("separation", 0.0),  # Littrow, D = 0
    ]
    found = chain(run, *stages, modes=0, bandpass=[6e-7, 1e-6])
    assert found["after"].shape == found["t"].shape == found["limit"].shape
    assert [stage["auto"] for stage in found["stages"]] == [False] * len(stages)
    pulses = found["pulses"]
    assert pulses["limit"]["peak"] >= pulses["before"]["peak"] * 0.999
    assert math.isclose(found["wavelength"], 8e-7, rel_tol=1e-3)
    empty = chain(run)  # no elements: after is before
    assert np.allclose(empty["after"], empty["before"])


def test_auto_elements_are_searched_one_at_a_time(run: str):  # noqa: F811 - a fixture
    taylor = new_stage("taylor", auto=True).with_value("order", 2)
    found = chain(run, new_stage("material").with_value("thickness", 2e-3), taylor)
    (fixed, searched) = found["stages"]
    assert not fixed["auto"] and searched["auto"] and searched["values"]["fod"] == 0.0
    assert "Compression(taylor" in searched["text"] and searched["quality"] <= 1.0
    pulses = found["pulses"]
    assert pulses["after"]["peak"] >= 0.9 * pulses["before"]["peak"]
    material = new_stage("material", auto=True)
    wedged = new_stage("mirrors", auto=True).with_value("max_reflections", 2)
    wedged = wedged.with_value("wedge", "fused silica").with_value("wedge_max", 2e-3)
    plain = new_stage("mirrors", auto=True).with_value("max_reflections", 1)
    found = chain(run, material, wedged, plain)
    reports = found["stages"]
    assert 0.0 <= reports[0]["values"]["thickness"] <= 5e-3
    assert set(reports[1]["values"]) == {"reflections", "thickness"}
    assert set(reports[2]["values"]) == {"reflections"}  # no wedge: no thickness


def test_pairs_are_searched_within_their_ranges(run: str):  # noqa: F811 - a fixture
    prisms = new_stage("prisms", auto=True)
    gratings = new_stage("gratings", auto=True).with_value("separation_max", 0.02)
    found = chain(run, prisms, gratings)
    prism, grating = found["stages"]
    assert 0.01 <= prism["values"]["insertion"] <= 0.02
    assert prism["values"]["separation"] == pytest.approx(0.6)  # held
    assert 0.01 <= grating["values"]["separation"] <= 0.02


def test_a_compressor_says_what_it_cannot_do(run: str, beam: str):  # noqa: F811
    with pytest.raises(SolpropError, match="field samples"):
        chain(beam, new_stage("taylor"))
    with pytest.raises(SolpropError, match="its thickness"):
        chain(run, new_stage("material").with_value("thickness", None))
    with pytest.raises(SolpropError, match="needs both its ends"):
        chain(run, new_stage("prisms", auto=True).with_value("insertion_max", None))
    no_range = new_stage("material", auto=True)
    no_range = no_range.with_value("thickness_min", None)
    with pytest.raises(SolpropError, match="thickness range"):
        chain(run, no_range.with_value("thickness_max", None))
    no_range = new_stage("gratings", auto=True).with_value("separation_min", None)
    with pytest.raises(SolpropError, match="separation range"):
        chain(run, no_range.with_value("separation_max", None))
    lens = {"kind": "lens", "auto": False, "values": {}}
    body = {"path": run, "index": -1, "stages": [lens]}
    with pytest.raises(SolpropError, match="no compressor stage"):
        ask("compress_chain", body)
    with pytest.raises(SolpropError, match="no compressor stage"):
        ask("compress_chain", {**body, "stages": [{**lens, "auto": True}]})


# ------------------------------------------------------------------ the page


def test_the_page_builds_a_compressor_and_takes_what_was_found(  # noqa: PLR0915
    qtbot: QtBot,
    tmp_path: Path,
    run: str,  # noqa: F811 - a fixture
):
    window = MainWindow(workspace=tmp_path / "runs")
    qtbot.addWidget(window)
    window.show()
    page = window.postprocess
    window.show_step("postprocess")
    assert "results file first" in page.target.text()
    assert not page.compress_button.isEnabled()
    page.compress()  # nothing open: nothing asked
    window.results.open(run)
    qtbot.waitUntil(lambda: bool(window.results.plane.planes), timeout=WAIT_MS)
    qtbot.waitUntil(lambda: bool(page.editor.catalogue), timeout=WAIT_MS)
    window.show_step("postprocess")
    assert "z = " in page.target.text()
    # A material, edited by hand: a thickness, a bad entry, a blank one.
    page.add("material")
    thickness = page.editor.inputs["thickness"]
    assert isinstance(thickness, QLineEdit)
    type_into(thickness, "2")
    assert page.stages[0].values["thickness"] == pytest.approx(2e-3)
    type_into(thickness, "thick")
    type_into(thickness, "")
    assert thickness.styleSheet() and page.stages[0].values["thickness"] == 2e-3
    pressure = page.editor.inputs["pressure"]
    assert isinstance(pressure, QLineEdit)
    type_into(pressure, "")  # optional: blank is fine
    # Chirped mirrors with wedges, searched.
    page.add("mirrors")
    page.editor.auto.setChecked(True)
    wedge = page.editor.inputs["wedge"]
    assert isinstance(wedge, QComboBox) and wedge.itemData(0) is None
    wedge.setCurrentIndex(wedge.findData("fused silica"))
    most = page.editor.inputs["max_reflections"]
    assert isinstance(most, QSpinBox)
    most.setValue(2)
    assert page.stages[1].auto and page.stages[1].values["max_reflections"] == 2
    # A prism pair, its pass switched; moved up, down, and removed.
    page.add("prisms")
    double = page.editor.inputs["double_pass"]
    assert isinstance(double, QCheckBox)
    double.setChecked(False)
    assert page.stages[2].values["double_pass"] is False
    page.shift(-1)
    assert page.stages[1].kind == "prisms"
    page.shift(1)
    page.shift(1)  # past the end: stays
    assert page.stages[2].kind == "prisms"
    page.remove()
    assert [stage.kind for stage in page.stages] == ["material", "mirrors"]
    # Compress, and take the values found into the chain.
    assert not page.take_button.isEnabled()
    page.compress_button.click()
    qtbot.waitUntil(lambda: "Done" in page.status.text(), timeout=WAIT_MS)
    assert window.views.currentWidget() is page.view
    assert page.view.table.rowCount() == 2 and "transform limit" in (
        page.view.summary.text()
    )
    shot(window, tmp_path, "postprocess")
    assert page.take_button.isEnabled()
    page.take_found()
    assert [stage.kind for stage in page.stages] == ["material", "mirrors", "material"]
    assert not any(stage.auto for stage in page.stages)
    page.take_found()  # nothing found for this chain yet: nothing changes
    # An element solprop cannot build: its reason is said.
    page.stages[0] = page.stages[0].with_value("thickness", None)
    page.compress()
    qtbot.waitUntil(lambda: "its thickness" in page.status.text(), timeout=WAIT_MS)
    page.view.clear()
    assert page.view.table.rowCount() == 0
    # A file whose compression solprop refuses says so.
    window.results.capabilities = {"refused": {"compression": "a beam"}}
    page.refresh()
    assert "a beam" in page.target.text() and not page.compress_button.isEnabled()
    window.close()


def test_an_empty_editor_and_a_stage_s_auto_switch(qtbot: QtBot):
    editor = StageEditor()
    qtbot.addWidget(editor)
    seen: list[Any] = []
    editor.changed.connect(seen.append)
    assert not editor.auto.isVisibleTo(editor)
    editor.auto.setChecked(True)  # no stage: nothing
    editor._set("x", 1)
    assert seen == []
    editor.set_catalogue({"glasses": ["bk7"], "mirrors": []})
    editor.show_stage(new_stage("mirrors", auto=True).with_value("wedge", "odd"))
    wedge = editor.inputs["wedge"]
    assert isinstance(wedge, QComboBox) and wedge.currentText() == "odd"  # kept
    editor.auto.setChecked(False)
    assert seen[-1].auto is False and "reflections" in editor.inputs


def type_into(field: QLineEdit, text: str) -> None:
    field.setText(text)
    field.editingFinished.emit()
