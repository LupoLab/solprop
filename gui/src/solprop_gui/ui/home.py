# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Home: shown when nothing is open, it asks what you want to do (design 5.2)."""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.classes.catalog import CLASSES
from solprop_gui.model.workspace import RunEntry
from solprop_gui.ui.theme import clear_layout, named_label

__all__ = ["HomePage"]

_RECENT = 6
"""How many recent runs or setups Home lists."""


class _Clickable(QFrame):
    """A frame that acts as a button.

    Parameters
    ----------
    name : str
        Its object name, which the style sheet addresses.
    title : str
        Its first line.
    text : str
        Its second line.
    tip : str, optional
        What its tooltip says, as a reason when it is disabled.
    """

    clicked = Signal()

    def __init__(self, name: str, title: str, text: str, tip: str = "") -> None:
        """Make the frame with its two lines."""
        super().__init__()
        self.setObjectName(name)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tip)
        kind = "tile" if name == "tile" else "item"
        box = QVBoxLayout(self)
        box.setContentsMargins(
            *((16, 14, 16, 14) if kind == "tile" else (10, 7, 10, 7))
        )
        box.setSpacing(2)
        box.addWidget(named_label(title, f"{kind}Title"))
        detail = named_label(text, f"{kind}Text")
        detail.setWordWrap(True)
        box.addWidget(detail)
        box.addStretch(1)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt's name
        """Announce a click.

        Parameters
        ----------
        event : PySide6.QtGui.QMouseEvent
            The release.
        """
        if self.isEnabled():
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class HomePage(QWidget):
    """What you can begin.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    class_chosen : Signal
        Emitted with a problem class's key.
    other_chosen : Signal
        Emitted when Other geometry is chosen.
    design_chosen : Signal
        Emitted when the person asks to design a waveguide.
    open_results_wanted : Signal
        Emitted when a results file is to be picked.
    open_setup_wanted : Signal
        Emitted when a setup file is to be picked.
    results_chosen : Signal
        Emitted with a recent run's results file.
    setup_chosen : Signal
        Emitted with a recent setup's file.
    """

    class_chosen = Signal(str)
    other_chosen = Signal()
    design_chosen = Signal()
    open_results_wanted = Signal()
    open_setup_wanted = Signal()
    results_chosen = Signal(str)
    setup_chosen = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:  # noqa: PLR0915 - a layout
        """Make the page, listing nothing recent yet."""
        super().__init__(parent)
        self.setObjectName("page")
        self.tiles: dict[str, _Clickable] = {}
        main = QVBoxLayout()
        main.setSpacing(14)
        main.addWidget(named_label("What are you simulating?", "h1"))
        tiles = QGridLayout()
        tiles.setSpacing(12)
        for index, problem in enumerate(CLASSES):
            tile = _Clickable("tile", problem.title, problem.summary)
            tile.clicked.connect(lambda key=problem.key: self.class_chosen.emit(key))
            self.tiles[problem.key] = tile
            tiles.addWidget(tile, index // 2, index % 2)
        other = _Clickable(
            "tile",
            "Other geometry",
            "Every other route, in the step window for now: free space, rectangular "
            "and slab guides, general modes.",
        )
        other.clicked.connect(self.other_chosen)
        self.tiles["other"] = other
        tiles.addWidget(other, 2, 0)
        design = _Clickable(
            "tile",
            "Design a waveguide",
            "Dispersion, loss and zero-dispersion maps, with no pulse and no run.",
        )
        design.clicked.connect(self.design_chosen)
        self.tiles["design"] = design
        tiles.addWidget(design, 2, 1)
        main.addLayout(tiles)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        for text, signal in (
            ("Open results…", self.open_results_wanted),
            ("Open a setup…", self.open_setup_wanted),
        ):
            button = QPushButton(text)
            button.setObjectName("plain")
            button.clicked.connect(signal)
            buttons.addWidget(button)
        buttons.addStretch(1)
        main.addLayout(buttons)
        main.addStretch(1)

        self.runs = QVBoxLayout()
        self.runs.setSpacing(2)
        self.setups = QVBoxLayout()
        self.setups.setSpacing(2)
        side = QVBoxLayout()
        side.setSpacing(6)
        side.addWidget(named_label("Recent runs", "h3"))
        side.addLayout(self.runs)
        side.addSpacing(12)
        side.addWidget(named_label("Recent setups", "h3"))
        side.addLayout(self.setups)
        side.addStretch(1)

        content = QWidget()
        content.setObjectName("page")
        columns = QHBoxLayout(content)
        columns.setContentsMargins(40, 36, 40, 24)
        columns.setSpacing(40)
        columns.addLayout(main, 3)
        columns.addLayout(side, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    @staticmethod
    def _fill(
        box: QVBoxLayout,
        entries: Sequence[tuple[str, str, Callable[[], None]]],
        empty: str,
    ) -> None:
        """Replace a list's items.

        Parameters
        ----------
        box : PySide6.QtWidgets.QVBoxLayout
            The list.
        entries : Sequence of tuple
            Each item's title, detail and action.
        empty : str
            What an empty list says.
        """
        clear_layout(box)
        for title, detail, action in entries:
            item = _Clickable("item", title, detail)
            item.clicked.connect(action)
            box.addWidget(item)
        if not entries:
            box.addWidget(named_label(empty, "itemText"))

    def show_runs(self, runs: Sequence[RunEntry]) -> None:
        """List the workspace's latest runs.

        Parameters
        ----------
        runs : Sequence of RunEntry
            The runs, newest first.
        """
        entries = []
        for run in [run for run in runs if run.output is not None][:_RECENT]:
            when = datetime.fromtimestamp(run.modified, tz=UTC).astimezone()
            output = str(run.output)
            entries.append(
                (
                    run.name,
                    when.strftime("%d %b %Y, %H:%M"),
                    lambda path=output: self.results_chosen.emit(path),
                )
            )
        self._fill(self.runs, entries, "No runs yet.")

    def show_setups(self, setups: Sequence[str]) -> None:
        """List the recent setups.

        Parameters
        ----------
        setups : Sequence of str
            Their files, newest first.
        """
        entries = [
            (
                Path(path).name,
                str(Path(path).parent),
                lambda path=path: self.setup_chosen.emit(path),
            )
            for path in setups[:_RECENT]
        ]
        self._fill(self.setups, entries, "No setups yet.")
