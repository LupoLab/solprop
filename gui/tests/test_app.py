# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The command and the main window."""

import pytest
from PySide6.QtWidgets import QLabel
from pytestqt.qtbot import QtBot

from solprop_gui import __version__
from solprop_gui.app import main
from solprop_gui.ui.main_window import MainWindow


def test_version_answers_and_exits(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == __version__


def test_the_main_window_opens(qtbot: QtBot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    assert window.windowTitle() == "solprop"
    placeholder = window.centralWidget()
    assert isinstance(placeholder, QLabel)
    assert __version__ in placeholder.text()
