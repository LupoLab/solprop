# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The start page: the geometry classes' gallery, the examples by class, offscreen."""

from pathlib import Path

from PySide6.QtWidgets import QTreeWidgetItem
from pytestqt.qtbot import QtBot

from solprop_gui.help.figures import FIGURES, figure_text
from solprop_gui.registry.routes import CARDS, card
from solprop_gui.ui.main_window import MainWindow
from solprop_gui.ui.start import NewSetupDialog, StartPage, figure_pixmap
from test_ui import WAIT_MS, private_settings, shot  # noqa: F401 - a fixture

ROOT = Path(__file__).resolve().parents[2]

EXAMPLES = [
    {
        "name": "spm",
        "title": "SPM",
        "route": "capillary",
        "build": {"mode_average": True},
    },
    {"name": "beam", "title": "A beam", "route": "free", "build": {}},
    {"name": "slab", "title": "A slab", "route": "planar", "build": {}},
    {"name": "odd", "title": "Odd", "route": "nope", "build": {}},
]


def test_the_figures_are_the_manual_s_and_resolve_for_qt():
    for key, name in FIGURES.items():
        packaged = ROOT / "gui" / "src" / "solprop_gui" / "help" / "figures"
        guide = ROOT / "docs" / "guide" / "figures" / f"{name}.svg"
        assert (packaged / f"{name}.svg").read_bytes() == guide.read_bytes(), name
        for dark in (False, True):
            text = figure_text(key, dark=dark)
            assert text is not None and "var(" not in text
    assert {entry.key for entry in CARDS} <= set(FIGURES)


def test_the_gallery_and_the_examples_by_class(qtbot: QtBot, tmp_path: Path):
    page = StartPage()
    qtbot.addWidget(page)
    page.resize(1100, 900)
    assert not page.tiles["planar"].isEnabled()  # disabled for now
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
    assert [e["name"] for e in page.examples_of("free")] == ["beam"]
    planar = items[groups.index("Planar waveguide (1)")]
    assert child(planar).isDisabled() and not planar.isExpanded()
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
    dialog = NewSetupDialog(card("free"), examples)
    qtbot.addWidget(dialog)
    assert dialog.options.count() == 2  # the template is the class's example
    dialog.options.setCurrentRow(1)
    dialog.accept()
    assert dialog.choice == "other"
    blank = NewSetupDialog(card("capillary-modal"), [])
    qtbot.addWidget(blank)
    blank.accept()
    assert blank.choice == ""


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
    window.close()


def group(page: StartPage, index: int) -> QTreeWidgetItem:
    item = page.examples.topLevelItem(index)
    assert item is not None
    return item


def child(item: QTreeWidgetItem) -> QTreeWidgetItem:
    found = item.child(0)
    assert found is not None
    return found
