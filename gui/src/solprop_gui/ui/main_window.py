# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The main window."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow, QWidget

from solprop_gui import __version__

__all__ = ["MainWindow"]


class MainWindow(QMainWindow):
    """The application's one main window.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owning widget; None for a top-level window.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("solprop")
        placeholder = QLabel(
            f"solprop GUI {__version__}\n\nIn development: nothing can be set up yet."
        )
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(placeholder)
        self.resize(1280, 800)
