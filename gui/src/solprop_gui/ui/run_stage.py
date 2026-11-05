# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Run stage: the run's view under a progress strip (design 10.2)."""

from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.ui.run_panel import RunPanel
from solprop_gui.ui.theme import named_label

__all__ = ["RunStage"]


class RunStage(QWidget):
    """What a run shows while it goes.

    The stage lays out the widgets a :class:`~solprop_gui.ui.run_panel.RunPanel` holds;
    the panel itself is never shown, and keeps the run's state.

    Parameters
    ----------
    panel : solprop_gui.ui.run_panel.RunPanel
        The run's panel.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.
    """

    def __init__(self, panel: RunPanel, parent: QWidget | None = None) -> None:
        """Lay out the strip, the advice and the view."""
        super().__init__(parent)
        self.setObjectName("page")
        strip = QFrame()
        strip.setObjectName("strip")
        row = QHBoxLayout(strip)
        row.setContentsMargins(16, 8, 16, 8)
        row.setSpacing(14)
        panel.state_label.setWordWrap(False)
        row.addWidget(panel.state_label)
        row.addWidget(panel.bar, 1)
        row.addWidget(panel.lamp)
        self.advice = QFrame()
        self.advice.setObjectName("advice")
        advice = QVBoxLayout(self.advice)
        self.advice_title = named_label("", "h3")
        self.advice_text = named_label("", "note")
        self.advice_text.setWordWrap(True)
        advice.addWidget(self.advice_title)
        advice.addWidget(self.advice_text)
        self.advice.hide()
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(strip)
        column.addWidget(self.advice)
        column.addWidget(panel.view, 1)

    def show_advice(self, title: str, message: str) -> None:
        """Say why a run failed; the planes it reached stay open.

        Parameters
        ----------
        title : str
            What happened.
        message : str
            The message solprop gave.
        """
        self.advice_title.setText(title)
        self.advice_text.setText(message)
        self.advice.show()

    def clear_advice(self) -> None:
        """Take the advice away, for a new run."""
        self.advice.hide()
