# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What the application is: its icon, its name to the system, and its About dialog.

The icon is ``assets/icon/``, drawn by ``gui/tools/app_icon.py``: PNGs from 16 to 1024
pixels, so each platform takes the size it draws. macOS shows the window icon in the
dock; Windows groups the taskbar button under the process's application id, which is
set before the first window so that the button shows this icon rather than Python's;
Linux desktops match the window to a ``solprop-gui.desktop`` file by its name.
"""

import ctypes
import platform
import sys
from importlib.metadata import PackageNotFoundError, metadata
from pathlib import Path
from typing import Final

from PySide6 import __version__ as pyside_version
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from solprop_gui import __version__
from solprop_gui.help.guide import MANUAL
from solprop_gui.ui.theme import named_label

__all__ = [
    "APP_ID",
    "LINKS",
    "AboutDialog",
    "app_icon",
    "apply_identity",
    "authors",
    "set_windows_app_id",
]

_ICON: Final = Path(__file__).resolve().parent / "assets" / "icon"
"""The icon's renderings."""

APP_ID: Final = "LupoLab.solprop.gui"
"""The application id Windows groups the taskbar button by."""

LINKS: Final = {
    "manual": MANUAL,
    "guide": MANUAL + "guide/front-end/index.html",
    "source": "https://github.com/LupoLab/solprop",
    "issues": "https://github.com/LupoLab/solprop/issues",
    "pypi": "https://pypi.org/project/solprop/",
}
"""Where the project lives."""

COPYRIGHT: Final = "© 2026 John C. Travers and contributors"
"""The copyright line, as the source files' SPDX headers give it."""


def app_icon() -> QIcon:
    """Return the application's icon, with every size drawn.

    Returns
    -------
    PySide6.QtGui.QIcon
        The icon; empty if its files are missing.
    """
    icon = QIcon()
    for path in sorted(_ICON.glob("solprop-*.png")):
        icon.addFile(str(path))
    return icon


def set_windows_app_id() -> bool:
    """Give the process its own application id on Windows, before any window.

    Without it Windows groups the window under ``python.exe`` and shows Python's icon on
    the taskbar.

    Returns
    -------
    bool
        Whether it was set: only on Windows.
    """
    if sys.platform != "win32":
        return False
    shell = ctypes.windll.shell32  # pyright: ignore[reportAttributeAccessIssue]
    shell.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    return True


def apply_identity(application: QApplication) -> None:
    """Name the application to the system and give it its icon.

    Parameters
    ----------
    application : PySide6.QtWidgets.QApplication
        The application.
    """
    application.setApplicationName("solprop")
    application.setApplicationVersion(__version__)
    application.setDesktopFileName("solprop-gui")
    application.setWindowIcon(app_icon())


def authors() -> list[str]:
    """Return the package's authors, as its metadata names them.

    Returns
    -------
    list of str
        Each author's name, without an address; solprop's own when the metadata is
        missing.
    """
    try:
        found = metadata("solprop-gui")
    except PackageNotFoundError:
        return ["John C. Travers", "Christian Brahms"]
    names: list[str] = []
    # Those with an address come first, as solprop's metadata lists them.
    for field in ("Author-email", "Author"):
        for entry in found.get_all(field) or []:
            for person in str(entry).split(","):
                name = person.split("<", 1)[0].strip().strip('"')
                if name and name not in names:
                    names.append(name)
    return names


def _solprop_version() -> str:
    """Return the installed solprop's version.

    Returns
    -------
    str
        The version, or ``unknown``.
    """
    try:
        return str(metadata("solprop")["Version"])
    except PackageNotFoundError:
        return "unknown"


class AboutDialog(QDialog):
    """Who made solprop, which versions are running, and where the project lives.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    versions : PySide6.QtWidgets.QLabel
        The versions of the front end, solprop, Qt and Python.
    links : PySide6.QtWidgets.QLabel
        The manual, the source, the issues and the package index.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Lay the dialog out."""
        super().__init__(parent)
        self.setWindowTitle("About solprop")
        self.setObjectName("page")
        picture = QLabel()
        picture.setPixmap(app_icon().pixmap(96, 96))
        picture.setAlignment(Qt.AlignmentFlag.AlignTop)
        title = named_label("solprop", "h2")
        summary = named_label(
            "Ultrafast nonlinear pulse propagation in waveguides and free space, "
            "set up, run and read in one window.",
            "note",
        )
        summary.setWordWrap(True)
        self.versions = named_label(
            f"solprop-gui {__version__} · solprop {_solprop_version()} · "
            f"Qt {pyside_version} · Python {platform.python_version()}",
            "note",
        )
        self.versions.setWordWrap(True)
        self.versions.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.links = QLabel(
            " · ".join(
                f'<a href="{LINKS[key]}">{text}</a>'
                for key, text in (
                    ("manual", "The manual"),
                    ("guide", "This window's guide"),
                    ("source", "Source on GitHub"),
                    ("pypi", "solprop on PyPI"),
                    ("issues", "Report an issue"),
                )
            )
        )
        self.links.setOpenExternalLinks(True)
        self.links.setWordWrap(True)
        credits_ = named_label(
            f"{COPYRIGHT}.<br>By {', '.join(authors())}.<br><br>"
            "solprop is free software: you can redistribute it and modify it under "
            "the terms of the GNU General Public License, version 3 or (at your "
            "option) any later version. It comes with no warranty. The window's "
            "typeface, IBM Plex Sans, is under the SIL Open Font License 1.1.",
            "note",
        )
        credits_.setWordWrap(True)
        text = QVBoxLayout()
        text.setSpacing(8)
        for widget in (title, summary, self.versions, self.links, credits_):
            text.addWidget(widget)
        row = QHBoxLayout()
        row.setSpacing(20)
        row.addWidget(picture)
        row.addLayout(text, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        column = QVBoxLayout(self)
        column.setContentsMargins(24, 24, 24, 16)
        column.addLayout(row)
        column.addWidget(buttons)
        self.setMinimumWidth(600)
        self.adjustSize()
