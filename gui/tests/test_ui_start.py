# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The start page: the geometry classes' gallery, the examples by class, offscreen."""

from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QTreeWidgetItem
from pytestqt.qtbot import QtBot

from solprop_gui.help.figures import FIGURES, figure_text
from solprop_gui.model.recipe import MISSING
from solprop_gui.registry.routes import CARDS, card, card_of, type_of
from solprop_gui.ui.main_window import MainWindow
from solprop_gui.ui.start import NewSetupDialog, StartPage, figure_pixmap
from test_ui import WAIT_MS, private_settings, shot  # noqa: F401 - a fixture

EXAMPLES = [
    {
        "name": "spm",
        "title": "SPM",
        "route": "capillary",
        "build": {"mode_average": True},
    },
    {
        "name": "beam",
        "title": "A beam",
        "route": "free",
        "build": {"space": {"type": "FreeGrid"}},
    },
    {"name": "slab", "title": "A slab", "route": "planar", "build": {}},
    {
        "name": "stripe",
        "title": "A stripe",
        "route": "free",
        "build": {"space": {"type": "LineGrid"}},
    },
    {"name": "odd", "title": "Odd", "route": "nope", "build": {}},
]


def test_every_shown_class_has_a_picture_that_resolves_for_qt():
    for key in FIGURES:
        for dark in (False, True):
            text = figure_text(key, dark=dark)
            assert text is not None and "var(" not in text
    assert {entry.key for entry in CARDS if not entry.hidden} <= set(FIGURES)
    # Drawn at the screen's pixel ratio: sharp on a high-density screen.
    pixmap = figure_pixmap("slab", QSize(200, 150), ratio=2.0)
    assert pixmap is not None and pixmap.width() == 400
    assert pixmap.devicePixelRatio() == 2.0


def test_the_gallery_and_the_examples_by_class(qtbot: QtBot, tmp_path: Path):
    page = StartPage()
    qtbot.addWidget(page)
    page.resize(1100, 900)
    assert "planar" not in page.tiles  # not offered for now: not shown
    assert set(page.tiles) == {entry.key for entry in CARDS if not entry.hidden}
    assert not page.tiles["step-index"].isEnabled()  # until solprop can say it
    assert not page.tiles["capillary-averaged"].icon().isNull()
    chosen = []
    page.card_chosen.connect(chosen.append)
    page.tiles["antiresonant"].click()
    assert chosen == ["antiresonant"]
    page.show_examples(EXAMPLES, ["capillary", "free"])
    items = [group(page, i) for i in range(page.examples.topLevelItemCount())]
    groups = [item.text(0) for item in items]
    assert (
        groups[0].startswith("Capillary, mode-averaged") and groups[-1] == "Other (1)"
    )
    assert [e["name"] for e in page.examples_of("cartesian")] == ["beam"]
    assert [e["name"] for e in page.examples_of("line")] == ["stripe"]
    assert not any(g.startswith("Planar") for g in groups)  # hidden with its card
    picked = []
    page.example_chosen.connect(picked.append)
    first = items[0]
    page._example_activated(child(first), 0)
    page._example_activated(first, 0)  # a group: nothing
    assert picked == ["spm"]
    assert figure_pixmap("nope", page.size()) is None
    shot(page, tmp_path, "start")


def test_a_new_setup_dialog_offers_the_template_and_the_examples(qtbot: QtBot):
    examples = [
        {"name": "cartesian_beam", "title": "T"},
        {"name": "other", "title": "O"},
    ]
    dialog = NewSetupDialog(card("cartesian"), examples)
    qtbot.addWidget(dialog)
    assert dialog.options.count() == 2  # the template is the class's example
    dialog.options.setCurrentRow(1)
    dialog.accept()
    assert dialog.choice == "other"
    blank = NewSetupDialog(card("capillary-modal"), [])
    qtbot.addWidget(blank)
    blank.accept()
    assert blank.choice == ""
    line = NewSetupDialog(card("line"), [])
    qtbot.addWidget(line)
    item = line.options.item(0)
    assert item is not None and "made this class" in item.text()


def test_a_class_begins_from_its_template_or_an_example(qtbot: QtBot, tmp_path: Path):
    window = MainWindow(workspace=tmp_path / "workspace")
    qtbot.addWidget(window)
    window.show()
    window.confirm = lambda _title, _text: "discard"
    page = window.pages["geometry"]
    window.show_step("geometry")
    assert page.empty.isVisible() and not page.fields_area.isVisible()
    window.choose_setup = lambda _entry, _examples: None  # cancelled
    window.begin_class("capillary-modal")
    assert window.model is None
    window.choose_setup = lambda _entry, _examples: ""  # the template
    window.begin_class("capillary-modal")
    qtbot.waitUntil(lambda: window._built_route == "capillary", timeout=WAIT_MS)
    assert not page.empty.isVisible()
    qtbot.waitUntil(lambda: "Geometry class" in window.summary.text(), timeout=WAIT_MS)
    window.choose_setup = lambda _entry, _examples: "soliton"  # an example
    window.begin_class("gnlse")
    qtbot.waitUntil(
        lambda: window.model is not None and window.model.route == "gnlse",
        timeout=WAIT_MS,
    )
    # A class without an example of its own: another's, made this class.
    window.choose_setup = lambda _entry, _examples: ""
    window.begin_class("line")
    qtbot.waitUntil(lambda: _titled(window, card("line").title), timeout=WAIT_MS)
    assert window.model is not None
    assert type_of(window.model.get("space")) == "LineGrid"
    assert card_of(window.model.route, window.model.document["build"]) is card("line")
    # A hidden class, still made (its code stays): the slab's rewrite drops a seed.
    window.new_from_card("slab")
    qtbot.waitUntil(lambda: _titled(window, card("slab").title), timeout=WAIT_MS)
    assert window.model.get("seed") is MISSING
    window.close()


def _titled(window: MainWindow, title: str) -> bool:
    return window.model is not None and window.model.title == title


def group(page: StartPage, index: int) -> QTreeWidgetItem:
    item = page.examples.topLevelItem(index)
    assert item is not None
    return item


def child(item: QTreeWidgetItem) -> QTreeWidgetItem:
    found = item.child(0)
    assert found is not None
    return found


def test_the_pictures_are_the_generator_s():
    import importlib.util  # noqa: PLC0415 - only here

    script = Path(__file__).resolve().parents[1] / "tools" / "geometry_icons.py"
    spec = importlib.util.spec_from_file_location("geometry_icons", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pictures = module.pictures()
    assert set(pictures) == set(FIGURES)
    for key, text in pictures.items():
        assert (module.FIGURES / f"{key}.svg").read_text(encoding="utf-8") == text, key
