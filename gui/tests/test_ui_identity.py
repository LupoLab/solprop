# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The application's icon, its About dialog and its help, offscreen."""

import importlib.util
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from solprop_gui.ui import help_window as help_module
from solprop_gui.ui import identity
from solprop_gui.ui.help_window import HelpWindow
from solprop_gui.ui.identity import AboutDialog, app_icon, apply_identity, authors

TOOL = Path(__file__).resolve().parents[1] / "tools" / "app_icon.py"
ICON = Path(identity.__file__).resolve().parent / "assets" / "icon"


def test_the_committed_icon_is_the_tool_s() -> None:
    spec = importlib.util.spec_from_file_location("app_icon", TOOL)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert (ICON / "solprop.svg").read_text() == tool.svg()
    assert (ICON / "solprop-compact.svg").read_text() == tool.svg(compact=True)
    for size in tool.SIZES:
        assert (ICON / f"solprop-{size}.png").is_file()


def test_the_application_carries_its_icon_and_name(qtbot: QtBot) -> None:
    del qtbot  # for its QApplication
    icon = app_icon()
    assert not icon.isNull()
    assert {size.width() for size in icon.availableSizes()} >= {16, 32, 256, 1024}
    application = QApplication.instance()
    assert isinstance(application, QApplication)
    apply_identity(application)
    assert application.applicationName() == "solprop"
    assert application.desktopFileName() == "solprop-gui"
    assert not application.windowIcon().isNull()
    assert identity.set_windows_app_id() is (sys.platform == "win32")


def test_about_names_the_versions_the_links_and_the_authors(qtbot: QtBot) -> None:
    dialog = AboutDialog()
    qtbot.addWidget(dialog)
    assert "solprop-gui" in dialog.versions.text()
    assert "Qt" in dialog.versions.text() and "Python" in dialog.versions.text()
    for address in (
        "github.com/LupoLab/solprop",
        "pypi.org/project/solprop",
        "readthedocs",
    ):
        assert address in dialog.links.text()
    assert "John C. Travers" in authors()
    dialog.grab()


def test_authors_without_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str) -> object:
        raise identity.PackageNotFoundError

    monkeypatch.setattr(identity, "metadata", missing)
    assert authors() == ["John C. Travers", "Christian Brahms"]
    assert identity._solprop_version() == "unknown"


def test_the_guide_reads_offline_and_follows_its_links(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = HelpWindow()
    qtbot.addWidget(window)
    window.show()
    assert window.names[0] == "index" and "first-run" in window.names
    assert window.contents.count() == len(window.names)
    assert window.current == "index"
    assert "desktop front end" in window.page.toPlainText().lower()
    # A link within the chapter turns the page, at its section.
    window.follow(QUrl("first-run.md#run-it"))
    assert window.current == "first-run"
    assert window.contents.currentRow() == window.names.index("first-run")
    window.follow(QUrl("#read-the-results"))
    assert window.current == "first-run"
    window.contents.setCurrentRow(window.names.index("scans"))
    assert window.current == "scans"
    window.show_page("no-such-page")
    assert window.current == "scans"
    # The pictures fit the page.
    window.show_page("index")
    window.page.fit_pictures()
    assert "images/home.png" in window.page.toMarkdown()
    # Anything else opens in the browser.
    opened: list[str] = []
    monkeypatch.setattr(
        help_module.QDesktopServices,
        "openUrl",
        lambda url: opened.append(url.toString()),
    )
    window.follow(QUrl("https://solprop.readthedocs.io/en/latest/"))
    assert opened == ["https://solprop.readthedocs.io/en/latest/"]


def test_a_window_with_no_guide_points_online(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(help_module, "chapter_folder", lambda: None)
    window = HelpWindow()
    qtbot.addWidget(window)
    assert window.names == []
    assert "not installed" in window.page.toPlainText()
    window.page.fit_pictures()  # no pictures, no folder: nothing to do
