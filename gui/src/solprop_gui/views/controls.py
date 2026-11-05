# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The small controls beside a page's plots."""

from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.ui.theme import named_label, repolish

__all__ = ["LabelledSlider", "Segments", "control_panel"]


class Segments(QWidget):
    """A labelled choice of a few things, as joined buttons.

    Parameters
    ----------
    title : str
        What it chooses.
    options : Sequence of tuple
        Each value and what its button says.
    current : object
        The value chosen.

    Attributes
    ----------
    chosen : Signal
        Emitted with the value the user picks.
    """

    chosen = Signal(object)

    def __init__(
        self, title: str, options: Sequence[tuple[object, str]], current: object
    ) -> None:
        """Make the buttons."""
        super().__init__()
        self.setObjectName("page")
        self.options = list(options)
        self.buttons: list[QPushButton] = []
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(5)
        column.addWidget(named_label(title, "control"))
        frame = QFrame()
        frame.setObjectName("segments")
        row = QHBoxLayout(frame)
        row.setContentsMargins(1, 1, 1, 1)
        row.setSpacing(0)
        for value, text in self.options:
            button = QPushButton(text)
            button.setObjectName("segment")
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _on=False, v=value: self._pick(v))
            self.buttons.append(button)
            row.addWidget(button)
        column.addWidget(frame)
        self.set_current(current)

    def set_current(self, value: object) -> None:
        """Show a value as chosen, without announcing it.

        Parameters
        ----------
        value : object
            The value.
        """
        for button, (option, _text) in zip(self.buttons, self.options, strict=True):
            button.setProperty("current", option == value)
            repolish(button)

    def _pick(self, value: object) -> None:
        """Take a click.

        Parameters
        ----------
        value : object
            The button's value.
        """
        self.set_current(value)
        self.chosen.emit(value)


class LabelledSlider(QWidget):
    """A slider with what it sets and its value above it.

    Parameters
    ----------
    title : str
        What it sets.
    low : int
        Its least value.
    high : int
        Its greatest value.
    value : int
        Its value.
    suffix : str, optional
        What follows the value.

    Attributes
    ----------
    moved : Signal
        Emitted with the value as the user moves it.
    slider : PySide6.QtWidgets.QSlider
        The slider.
    """

    moved = Signal(int)

    def __init__(
        self, title: str, low: int, high: int, value: int, suffix: str = ""
    ) -> None:
        """Make the slider."""
        super().__init__()
        self.setObjectName("page")
        self.suffix = suffix
        self.shown = named_label(f"{value}{suffix}", "control")
        head = QHBoxLayout()
        head.addWidget(named_label(title, "control"))
        head.addStretch(1)
        head.addWidget(self.shown)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.setRange(low, high)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(self._changed)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(5)
        column.addLayout(head)
        column.addWidget(self.slider)

    def set_range(self, low: int, high: int) -> None:
        """Change the range, without announcing it.

        Parameters
        ----------
        low : int
            The least value.
        high : int
            The greatest value.
        """
        self.slider.blockSignals(True)
        self.slider.setRange(low, high)
        self.slider.blockSignals(False)

    def set_value(self, value: int) -> None:
        """Show a value, without announcing it.

        Parameters
        ----------
        value : int
            The value.
        """
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self.shown.setText(f"{self.slider.value()}{self.suffix}")

    def _changed(self, value: int) -> None:
        """Take a move.

        Parameters
        ----------
        value : int
            The value.
        """
        self.shown.setText(f"{value}{self.suffix}")
        self.moved.emit(value)


def control_panel(*widgets: QWidget, note: QLabel | None = None) -> QWidget:
    """Stack a page's controls.

    Parameters
    ----------
    *widgets : PySide6.QtWidgets.QWidget
        The controls, in order.
    note : PySide6.QtWidgets.QLabel or None, optional
        A note below them.

    Returns
    -------
    PySide6.QtWidgets.QWidget
        The panel.
    """
    panel = QWidget()
    panel.setObjectName("page")
    column = QVBoxLayout(panel)
    column.setContentsMargins(0, 0, 0, 0)
    column.setSpacing(14)
    for widget in widgets:
        column.addWidget(widget)
    if note is not None:
        note.setWordWrap(True)
        column.addWidget(note)
    column.addStretch(1)
    return panel
