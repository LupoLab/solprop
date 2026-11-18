# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Parts of a settings panel: a collapsible section, and the keyword an issue names."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolButton, QVBoxLayout, QWidget

__all__ = ["Section", "issue_keyword"]


def issue_keyword(parameter: object) -> list[str]:
    """Return the keywords an issue's parameter paths name.

    Parameters
    ----------
    parameter : object
        The issue's parameter: a path, a list of paths, or ``None``.

    Returns
    -------
    list of str
        The first step of each path: the keyword whose row shows the issue.
    """
    if parameter is None:
        return []
    paths = parameter if isinstance(parameter, list | tuple) else [parameter]
    names = []
    for path in paths:
        text = str(path)
        for separator in (".", "["):
            text = text.split(separator, 1)[0]
        names.append(text)
    return names


class Section(QWidget):
    """A titled, collapsible part of a settings panel, as COMSOL's settings have.

    Parameters
    ----------
    title : str
        Its title.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """Make the section, open and empty."""
        super().__init__(parent)
        self.header = QToolButton()
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(True)
        self.header.setAutoRaise(True)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setArrowType(Qt.ArrowType.DownArrow)
        self.header.setStyleSheet("QToolButton { font-weight: bold; border: none; }")
        self.header.toggled.connect(self.set_open)
        self.body = QWidget()
        self.rows = QVBoxLayout(self.body)
        self.rows.setContentsMargins(12, 0, 0, 4)
        self.rows.setSpacing(0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.body)

    def set_open(self, open_: bool) -> None:
        """Show or fold the section's body.

        Parameters
        ----------
        open_ : bool
            Whether to show it.
        """
        self.header.setArrowType(
            Qt.ArrowType.DownArrow if open_ else Qt.ArrowType.RightArrow
        )
        self.body.setVisible(open_)

    def add(self, widget: QWidget) -> None:
        """Add a widget to the section's body.

        Parameters
        ----------
        widget : PySide6.QtWidgets.QWidget
            The widget.
        """
        self.rows.addWidget(widget)
