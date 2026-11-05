# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The workflow window's look: calm and light, with white plots (design D18)."""

from pathlib import Path
from typing import Final

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QLabel, QLayout, QWidget

__all__ = [
    "ACCENT",
    "AXIS",
    "FONT",
    "GREY",
    "INK",
    "STYLE",
    "clear_layout",
    "load_font",
    "named_label",
    "repolish",
]

INK: Final = "#1b2430"
"""The colour of text."""

GREY: Final = "#5a6675"
"""The colour of secondary text."""

AXIS: Final = "#b4bcc6"
"""The colour of a plot's axes."""

ACCENT: Final = "#0b6e99"
"""The colour of the primary action and of what is selected."""

FONT: Final = "IBM Plex Sans"
"""The window's font, carried in the package under the Open Font License."""

_ASSETS = Path(__file__).parent / "assets"

_CHEVRON = (_ASSETS / "chevron.svg").as_posix()

STYLE: Final = f"""
* {{ font-size: 13px; }}
QMainWindow, QWidget#page, QScrollArea, QStackedWidget {{ background: #fff; }}
QWidget#page QLabel {{ color: {INK}; }}
QFrame#top {{ border-bottom: 1px solid #e3e6ea; background: #fff; }}
QPushButton#brand {{ border: 0; font-weight: 600; font-size: 15px; background: none;
    padding: 4px; color: {INK}; }}
QLabel#context {{ color: {GREY}; }}
QPushButton#stage {{ border: 0; border-radius: 6px; padding: 6px 14px; color: {GREY};
    font-weight: 500; background: none; }}
QPushButton#stage:hover {{ background: #f1f3f5; }}
QPushButton#stage[current="true"] {{ background: #e8f1f5; color: #0a5a7d; }}
QPushButton#stage:disabled {{ color: #a4adb8; background: none; }}
QPushButton#primary {{ background: {ACCENT}; color: #fff; border: 0; border-radius: 6px;
    font-weight: 500; min-width: 96px; padding: 7px 14px; }}
QPushButton#primary:hover {{ background: #095d82; }}
QPushButton#primary:disabled {{ background: #b9c4cc; }}
QToolButton#overflow {{ border: 0; border-radius: 6px; padding: 5px 8px;
    font-size: 16px; color: {GREY}; background: none; }}
QToolButton#overflow:hover {{ background: #f1f3f5; }}
QToolButton#overflow::menu-indicator {{ image: none; }}
QLabel#h1 {{ font-size: 28px; font-weight: 600; }}
QLabel#h2 {{ font-size: 15px; font-weight: 600; }}
QLabel#h3 {{ font-size: 13px; font-weight: 600; color: #3d4957; }}
QFrame#tile {{ border: 1px solid #dfe3e8; border-radius: 10px; background: #fff;
    min-height: 78px; }}
QFrame#tile:hover {{ border-color: {ACCENT}; background: #fafcfd; }}
QFrame#tile[enabled="false"] {{ background: #f7f8f9; }}
QLabel#tileTitle {{ font-weight: 600; background: none; }}
QLabel#tileText {{ color: {GREY}; font-size: 13px; background: none; }}
QFrame#item {{ border-radius: 6px; background: none; }}
QFrame#item:hover {{ background: #f1f3f5; }}
QLabel#itemTitle {{ font-weight: 500; background: none; }}
QLabel#itemText {{ color: {GREY}; font-size: 12px; background: none; }}
QPushButton#plain {{ background: #fff; border: 1px solid #cdd3da; border-radius: 6px;
    padding: 5px 12px; color: {INK}; }}
QPushButton#plain:hover {{ background: #f6f7f8; }}
QToolButton#plain {{ background: #fff; border: 1px solid #cdd3da; border-radius: 6px;
    padding: 4px 10px; color: {INK}; }}
QToolButton#plain:hover {{ background: #f6f7f8; }}
QToolButton#plain::menu-indicator {{ image: none; }}
QFrame#form {{ border-right: 1px solid #e3e6ea; background: #fff; }}
QLineEdit {{ border: 1px solid #cdd3da; border-radius: 5px; padding: 4px 7px;
    background: #fff; color: {INK}; selection-background-color: #bcdbe8;
    selection-color: {INK}; }}
QLineEdit:focus {{ border: 2px solid {ACCENT}; padding: 3px 6px; }}
QComboBox {{ border: 1px solid #cdd3da; border-radius: 5px; padding: 4px 7px;
    background: #fff; color: {INK}; }}
QComboBox:focus {{ border: 2px solid {ACCENT}; padding: 3px 6px; }}
QComboBox::drop-down {{ border: 0; width: 20px; }}
QComboBox::down-arrow {{ image: url("{_CHEVRON}"); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ border: 1px solid #d7dce2; background: #fff;
    color: {INK}; selection-background-color: #e8f1f5;
    selection-color: #0a5a7d; outline: 0; }}
QLabel#help {{ color: {GREY}; font-size: 12px; }}
QLabel#derivedLabel {{ color: {GREY}; font-size: 12px; }}
QLabel#derivedValue {{ font-size: 19px; font-weight: 500; }}
QLabel#ready {{ color: #1c7a4a; font-weight: 500; }}
QLabel#ready[warn="true"] {{ color: #8a5200; }}
QLabel#ready[error="true"] {{ color: #b3261e; }}
QFrame#preview {{ border: 1px dashed #d7dce2; border-radius: 8px; background: #fff; }}
QLabel#note {{ color: {GREY}; font-size: 13px; }}
QFrame#rail {{ border-right: 1px solid #e3e6ea; background: #fff; }}
QPushButton#railItem {{ border: 0; border-radius: 6px; text-align: left;
    padding: 7px 10px; color: #3d4957; background: none; }}
QPushButton#railItem:hover {{ background: #f1f3f5; }}
QPushButton#railItem[current="true"] {{ background: #e8f1f5; color: #0a5a7d;
    font-weight: 500; }}
QPushButton#railItem:disabled {{ color: #a4adb8; background: none; }}
QFrame#scrubber {{ border-top: 1px solid #e3e6ea; background: #fff; }}
QLabel#scrubberText {{ font-weight: 500; color: {INK}; }}
QSlider::groove:horizontal {{ height: 6px; background: #e8ebee; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 16px; height: 16px;
    margin: -5px 0; border-radius: 8px; }}
QPushButton#derivedTarget {{ font-size: 19px; font-weight: 500; color: #0a5a7d;
    border: 0; border-bottom: 1px dashed #7fb0c5; background: none; padding: 0;
    text-align: left; }}
QPushButton#derivedTarget[pinned="true"] {{ background: #e8f1f5; border: 0;
    border-radius: 4px; padding: 0 5px; }}
QFrame#targetEditor {{ background: #f4f8fa; border: 1px solid #d5e4ec;
    border-radius: 8px; }}
QLabel#solved {{ color: #0a5a7d; font-size: 12px; }}
QPushButton#link {{ border: 0; color: #0a5a7d; text-decoration: underline;
    font-size: 12px; background: none; padding: 0; }}
QFrame#panel {{ border-left: 1px solid #e3e6ea; background: #fff; }}
QLabel#control {{ font-size: 12px; color: #3d4957; font-weight: 500; }}
QLabel#plotTitle {{ font-size: 12px; font-weight: 500; color: #3d4957; }}
QLabel#pageNote {{ color: {GREY}; font-size: 12px; }}
QFrame#segments {{ border: 1px solid #cdd3da; border-radius: 6px; background: #fff; }}
QPushButton#segment {{ border: 0; background: none; padding: 4px 8px; font-size: 12px;
    border-radius: 5px; color: {INK}; }}
QPushButton#segment[current="true"] {{ background: #e8f1f5; color: #0a5a7d;
    font-weight: 500; }}
QFrame#strip {{ border-bottom: 1px solid #e3e6ea; background: #fff; }}
QProgressBar#run-progress {{ border: 0; background: #e8ebee; border-radius: 3px;
    max-height: 6px; min-height: 6px; color: transparent; }}
QProgressBar#run-progress::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QFrame#advice {{ border: 1px solid #e6cfa8; background: #fdf8ee; border-radius: 8px; }}
QFrame#drawer {{ border-left: 1px solid #e3e6ea; background: #fff; }}
QPlainTextEdit#drawerText {{ border: 0; background: #fff; color: {INK};
    font-family: "Menlo", "Consolas", monospace; font-size: 12px; }}
"""
"""The window's style sheet."""


def load_font() -> bool:
    """Make the carried font the application's, plots' axes included.

    Without it, the window keeps the system's font and works as well.

    Returns
    -------
    bool
        Whether every weight loaded.
    """
    results = [
        QFontDatabase.addApplicationFont(str(path))
        for path in sorted((_ASSETS / "fonts").glob("*.ttf"))
    ]
    loaded = bool(results) and -1 not in results
    if loaded:
        font = QFont(FONT)
        # What a plot's axes take; the style sheet sets the widgets' own size.
        font.setPixelSize(12)
        QApplication.setFont(font)
    return loaded


def repolish(widget: QWidget) -> None:
    """Apply a changed style property.

    Parameters
    ----------
    widget : PySide6.QtWidgets.QWidget
        The widget.
    """
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def clear_layout(layout: QLayout) -> None:
    """Remove and delete everything a layout holds, nested layouts included.

    Parameters
    ----------
    layout : PySide6.QtWidgets.QLayout
        The layout.
    """
    while layout.count():
        item = layout.takeAt(0)
        if item is None:
            continue
        widget = item.widget()
        inner = item.layout()
        if widget is not None:
            # At once: deleteLater alone leaves it drawn until the next event loop turn.
            widget.hide()
            widget.setParent(None)  # pyright: ignore[reportArgumentType, reportCallIssue]
            widget.deleteLater()
        elif inner is not None:
            clear_layout(inner)


def named_label(text: str, name: str) -> QLabel:
    """Return a label with a style name.

    Parameters
    ----------
    text : str
        Its text.
    name : str
        Its object name, which the style sheet addresses.

    Returns
    -------
    PySide6.QtWidgets.QLabel
        The label.
    """
    label = QLabel(text)
    label.setObjectName(name)
    return label
