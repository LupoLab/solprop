# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Analyse stage: a rail of pages, one large, and one z scrubber (design 11.2)."""

from collections.abc import Mapping, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.classes.problem import AnalysisPage, page_refusal
from solprop_gui.registry.units import si
from solprop_gui.ui.theme import clear_layout, named_label, repolish

__all__ = ["AnalyseStage", "ZScrubber"]


class ZScrubber(QFrame):
    """The position along the propagation, shared by every page.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    moved : Signal
        Emitted with the plane's place among the positions.
    """

    moved = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the scrubber, with no planes."""
        super().__init__(parent)
        self.setObjectName("scrubber")
        self.positions: list[float] = []
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setObjectName("z-scrubber")
        self.slider.valueChanged.connect(self._changed)
        self.text = named_label("", "scrubberText")
        self.text.setMinimumWidth(190)
        self.text.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 8, 16, 8)
        row.setSpacing(14)
        row.addWidget(named_label("z", "h3"))
        row.addWidget(self.slider, 1)
        row.addWidget(self.text)

    def set_positions(self, positions: Sequence[float]) -> None:
        """Take the planes' positions, and go to the last.

        Parameters
        ----------
        positions : Sequence of float
            Where each plane is, m.
        """
        self.positions = [float(z) for z in positions]
        last = max(len(self.positions) - 1, 0)
        self.slider.blockSignals(True)
        self.slider.setRange(0, last)
        self.slider.setValue(last)
        self.slider.blockSignals(False)
        self.setEnabled(bool(self.positions))
        self._show()

    def go_to(self, place: int) -> None:
        """Move to a plane, without announcing it.

        Parameters
        ----------
        place : int
            Its place among the positions.
        """
        self.slider.blockSignals(True)
        self.slider.setValue(place)
        self.slider.blockSignals(False)
        self._show()

    def step(self, steps: int) -> None:
        """Move by a number of planes.

        Parameters
        ----------
        steps : int
            How many, signed.
        """
        self.slider.setValue(self.slider.value() + steps)

    def _changed(self, place: int) -> None:
        """Announce a move.

        Parameters
        ----------
        place : int
            The new place.
        """
        self._show()
        self.moved.emit(place)

    def _show(self) -> None:
        """Say where the scrubber is."""
        if not self.positions:
            self.text.setText("No planes")
            return
        place = self.slider.value()
        count = len(self.positions)
        self.text.setText(
            f"{si(self.positions[place], 'm')} (plane {place + 1} of {count})"
        )


class AnalyseStage(QWidget):
    """A rail of a class's pages, one shown large.

    Parameters
    ----------
    views : Mapping
        The widget of each page key that has one; any other page says it is not built.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    buttons : dict
        The rail's buttons, by page key.
    current : str or None
        The page shown.
    scrubber : ZScrubber
        The position shared by every page.
    """

    def __init__(
        self, views: Mapping[str, QWidget], parent: QWidget | None = None
    ) -> None:
        """Make the frame, with no pages."""
        super().__init__(parent)
        self.setObjectName("page")
        self.views = dict(views)
        self.buttons: dict[str, QPushButton] = {}
        self.pages: list[AnalysisPage] = []
        self.current: str | None = None
        self.rail = QFrame()
        self.rail.setObjectName("rail")
        self.rail.setFixedWidth(170)
        self.items = QVBoxLayout(self.rail)
        self.items.setContentsMargins(8, 10, 8, 10)
        self.items.setSpacing(2)
        self.title = named_label("", "h2")
        self.stack = QStackedWidget()
        for view in self.views.values():
            self.stack.addWidget(view)
        self.unbuilt = named_label("", "note")
        self.unbuilt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stack.addWidget(self.unbuilt)
        page = QVBoxLayout()
        page.setContentsMargins(16, 10, 12, 8)
        page.setSpacing(8)
        page.addWidget(self.title)
        page.addWidget(self.stack, 1)
        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)
        middle.addWidget(self.rail)
        middle.addLayout(page, 1)
        self.scrubber = ZScrubber()
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addLayout(middle, 1)
        column.addWidget(self.scrubber)
        for key, action in (
            (Qt.Key.Key_Left, lambda: self.scrubber.step(-1)),
            (Qt.Key.Key_Right, lambda: self.scrubber.step(1)),
            (Qt.Key.Key_Up, lambda: self.step_page(-1)),
            (Qt.Key.Key_Down, lambda: self.step_page(1)),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(action)

    def set_pages(
        self, pages: Sequence[AnalysisPage], capabilities: Mapping[str, object]
    ) -> None:
        """Offer a class's pages on a file, each disabled with its reason.

        Parameters
        ----------
        pages : Sequence of AnalysisPage
            The class's pages.
        capabilities : Mapping
            What the file can give, from the kernel's ``describe_output``.
        """
        self.pages = list(pages)
        clear_layout(self.items)
        self.buttons = {}
        for page in self.pages:
            button = QPushButton(page.title)
            button.setObjectName("railItem")
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _on=False, key=page.key: self.show_page(key))
            refusal = page_refusal(page, capabilities)
            button.setEnabled(refusal is None)
            button.setToolTip(refusal or "")
            self.buttons[page.key] = button
            self.items.addWidget(button)
        self.items.addStretch(1)
        kept = self.current in self.buttons and self.buttons[self.current].isEnabled()
        first = next(
            (p.key for p in self.pages if self.buttons[p.key].isEnabled()), None
        )
        chosen = self.current if kept else first
        if chosen is not None:
            self.show_page(chosen)

    def show_page(self, key: str) -> None:
        """Show one page large.

        Parameters
        ----------
        key : str
            The page.
        """
        self.current = key
        title = next(page.title for page in self.pages if page.key == key)
        self.title.setText(title)
        if key in self.views:
            self.stack.setCurrentWidget(self.views[key])
        else:
            self.unbuilt.setText(f"{title}: not built yet.")
            self.stack.setCurrentWidget(self.unbuilt)
        for name, button in self.buttons.items():
            button.setProperty("current", name == key)
            repolish(button)

    def step_page(self, steps: int) -> None:
        """Move through the pages that can be shown.

        Parameters
        ----------
        steps : int
            How many, signed; the move stops at either end.
        """
        offered = [key for key, button in self.buttons.items() if button.isEnabled()]
        if self.current in offered:
            place = offered.index(self.current) + steps
            self.show_page(offered[max(min(place, len(offered) - 1), 0)])
