# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The registered types' editors and the guided routes' cards, offscreen."""

from collections.abc import Mapping
from pathlib import Path

from pytestqt.qtbot import QtBot

from solprop_gui.model.recipe import MISSING, RecipeModel
from solprop_gui.ui.fields import ParameterRow
from solprop_gui.ui.main_window import MainWindow
from solprop_gui.ui.objects import ObjectField, ObjectListField
from test_design import SLAB
from test_ui import WAIT_MS, leave, private_settings, shot  # noqa: F401 - a fixture


def keyword(
    name: str, kind: str = "number", unit: str | None = "m", **extra: object
) -> dict[str, object]:
    meta: dict[str, object] = {
        "name": name,
        "forms": [{"kind": kind, "unit": unit, **extra}],
        "summary": f"The {name}.",
        "level": "basic",
        "display": "um" if unit == "m" else None,
    }
    return meta


CATALOG = {
    "Box": [
        {**keyword("width"), "required": True},
        {**keyword("depth"), "level": "advanced"},
        keyword("shape", "choice", None, choices=["flat", "round"]),
    ],
    "Ring": [keyword("radius")],
}


def row(meta: Mapping[str, object]) -> ParameterRow:
    return ParameterRow(meta, "object", catalog=CATALOG)


def test_an_object_field_edits_one_type_s_table(qtbot: QtBot, tmp_path: Path):
    field = ObjectField(["Box", "Ring", "Unknown"], CATALOG, row)
    qtbot.addWidget(field)
    assert field.types == ["Box", "Ring"]  # a type the kernel did not describe is out
    seen = []
    field.changed.connect(seen.append)
    box = field.by_name["Box"]
    assert not box.rows["depth"].isVisibleTo(field)  # advanced, unset: behind "more"
    assert box.more.text() == "more (1)"
    box.more.setChecked(True)
    assert box.rows["depth"].isVisibleTo(field) and box.more.text() == "fewer"
    leave(box.rows["width"].number.number, "10")  # pyright: ignore[reportAttributeAccessIssue]
    assert seen[-1] == {"type": "Box", "width": "10 um"}
    box.rows["width"].reset.click()  # pyright: ignore[reportAttributeAccessIssue]
    assert seen[-1] == {"type": "Box"}
    field.picker.setCurrentIndex(field.picker.findData("Ring"))
    assert seen[-1] == {"type": "Ring"}
    field.set_value({"type": "Box", "width": "5 um", "depth": "1 um", "x": 1})
    assert field.table() == {"type": "Box", "width": "5 um", "depth": "1 um"}
    assert field.fits({"type": "Ring"}) and not field.fits({"type": "Unknown"})
    assert not field.fits([{"type": "Box"}])
    field.set_value("not a table")  # the type picked, empty
    assert field.table() == {"type": "Box"}
    shot(field, tmp_path, "object")


def test_an_object_list_adds_removes_and_keeps_one_table(qtbot: QtBot):
    field = ObjectListField(["Box", "Ring"], CATALOG, row)
    qtbot.addWidget(field)
    seen = []
    field.changed.connect(seen.append)
    assert field.value() is None
    field.add_button.click()
    assert seen[-1] == [{"type": "Box"}]
    field.items[0].picker.setCurrentIndex(1)
    assert seen[-1] == [{"type": "Ring"}]
    field.add_item()
    assert len(seen[-1]) == 2
    field.remove_item(field.items[0])
    assert seen[-1] == [{"type": "Box"}]
    field.remove_item(field.items[0])
    assert seen[-1] is None
    # One table loaded is written back as one table while one is left.
    field.set_value({"type": "Ring", "radius": "3 um"})
    assert field.single and field.value() == {"type": "Ring", "radius": "3 um"}
    field.add_item()
    assert isinstance(seen[-1], list) and len(seen[-1]) == 2
    assert field.fits([{"type": "Box"}]) and not field.fits([])
    assert not field.fits([{"type": "Nope"}]) and not field.fits("x")
    field.set_value("x")
    assert field.items == []


def test_a_row_offers_its_types_and_falls_back_to_text(qtbot: QtBot):
    meta = keyword("lattice", "object", None, types=["Box"], recipe=True)
    single = ParameterRow(meta, catalog=CATALOG)
    qtbot.addWidget(single)
    assert single.modes == ["object"]
    seen = []
    single.edited.connect(lambda *args: seen.append(args))
    single.set_value({"type": "Box", "width": "2 um"})
    assert single.stack.currentWidget() is single.object
    single.object.picker.setCurrentIndex(0)
    leave(single.object.current().rows["width"].number.number, "3")  # pyright: ignore[reportAttributeAccessIssue]
    assert seen[-1] == ("lattice", "build", {"type": "Box", "width": "3 um"})
    single.set_value(MISSING)
    listed = ParameterRow({**meta, "name": "pulses"}, catalog=CATALOG)
    qtbot.addWidget(listed)
    assert listed.modes == ["objects"]
    listed.set_value([{"type": "Box"}])
    assert listed.stack.currentWidget() is listed.objects
    listed.set_value(MISSING)
    bare = ParameterRow(meta)  # no catalog: as text
    qtbot.addWidget(bare)
    assert bare.modes == ["text"]
    single.set_per_width(True)
    assert "per metre of y" in single.label.text()
    single.set_per_width(False)
    assert single.label.text() == "Lattice"


def _route(window: MainWindow, route: str) -> bool:
    model = window.model
    return model is not None and model.route == route and window._built_route == route


def test_the_guided_cards_start_from_solprop_s_examples(qtbot: QtBot, tmp_path: Path):
    window = MainWindow(workspace=tmp_path / "workspace")
    qtbot.addWidget(window)
    window.show()
    window.confirm = lambda _title, _text: "discard"

    # The waveguide: its mode edited in its own rows.
    window.new_from_card("waveguide")
    qtbot.waitUntil(lambda: _route(window, "waveguide"), timeout=WAIT_MS)
    modes = window.pages["geometry"].rows["modes", "build"]
    assert modes.stack.currentWidget() is modes.objects
    rect = modes.objects.items[0].current()
    assert rect.name == "RectMode"
    leave(rect.rows["half_width_y"].number.number, "60")  # pyright: ignore[reportAttributeAccessIssue]
    assert window.model is not None
    assert window.model.get("modes")[0]["half_width_y"] == "60 um"  # pyright: ignore[reportIndexIssue]
    window.undo.undo()
    assert window.model.get("modes")[0]["half_width_y"] == "50 um"  # pyright: ignore[reportIndexIssue]
    window.show_step("geometry")
    shot(window, tmp_path, "waveguide")

    # A slab set: the energy is per metre of y.
    window.set_model(RecipeModel(SLAB), base=None)
    energy = window.pages["input"].rows["energy", "build"]
    qtbot.waitUntil(lambda: "per metre" in energy.label.text(), timeout=WAIT_MS)
    qtbot.waitUntil(lambda: "µJ/m" in window.summary.text(), timeout=WAIT_MS)

    # The antiresonant fibre: its lattice is a table of its own.
    window.new_from_card("antiresonant")
    qtbot.waitUntil(
        lambda: window.model is not None and window.model.get("lattice") is not MISSING,
        timeout=WAIT_MS,
    )
    lattice = window.pages["geometry"].rows["lattice", "build"]
    assert lattice.stack.currentWidget() is lattice.object
    assert lattice.object.current().name == "TubeLattice"

    # The GNLSE fibre.
    window.new_from_card("gnlse")
    qtbot.waitUntil(lambda: _route(window, "gnlse"), timeout=WAIT_MS)
    window.new_from_card("capillary-averaged")  # a blank card, as before
    qtbot.waitUntil(lambda: _route(window, "capillary"), timeout=WAIT_MS)
    window.close()
