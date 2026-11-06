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
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.classes.catalog import CLASSES
from solprop_gui.model.workspace import RunEntry
from solprop_gui.ui.theme import clear_layout, named_label
from solprop_gui.ui.tile_art import tile_pixmap

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
    art : str or None, optional
        Its drawing's key in :data:`~solprop_gui.ui.tile_art.ART`, or ``None``.
    """

    clicked = Signal()

    def __init__(
        self, name: str, title: str, text: str, tip: str = "", art: str | None = None
    ) -> None:
        """Make the frame with its two lines, and its drawing beside them."""
        super().__init__()
        self.setObjectName(name)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tip)
        kind = "tile" if name == "tile" else "item"
        row = QHBoxLayout(self)
        row.setContentsMargins(
            *((16, 14, 16, 14) if kind == "tile" else (10, 7, 10, 7))
        )
        row.setSpacing(14)
        if art is not None:
            drawing = QLabel()
            drawing.setObjectName("tileArt")
            drawing.setPixmap(tile_pixmap(art))
            drawing.setAlignment(Qt.AlignmentFlag.AlignTop)
            row.addWidget(drawing)
        box = QVBoxLayout()
        box.setSpacing(2)
        box.addWidget(named_label(title, f"{kind}Title"))
        detail = named_label(text, f"{kind}Text")
        detail.setWordWrap(True)
        box.addWidget(detail)
        box.addStretch(1)
        row.addLayout(box, 1)

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
    workspace_wanted : Signal
        Emitted when another folder for the runs is to be picked.
    workspace : PySide6.QtWidgets.QLabel
        Which folder the runs go in.
    """

    class_chosen = Signal(str)
    design_chosen = Signal()
    open_results_wanted = Signal()
    open_setup_wanted = Signal()
    results_chosen = Signal(str)
    setup_chosen = Signal(str)
    workspace_wanted = Signal()

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
        shown = [problem for problem in CLASSES if problem.home]
        for index, problem in enumerate(shown):
            tile = _Clickable("tile", problem.title, problem.summary, art=problem.key)
            tile.clicked.connect(lambda key=problem.key: self.class_chosen.emit(key))
            self.tiles[problem.key] = tile
            tiles.addWidget(tile, index // 2, index % 2)
        after = len(shown)
        design = _Clickable(
            "tile",
            "Design a waveguide",
            "Dispersion, loss and zero-dispersion maps, with no pulse and no run.",
            art="design",
        )
        design.clicked.connect(self.design_chosen)
        self.tiles["design"] = design
        tiles.addWidget(design, after // 2, after % 2)
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
        self.workspace = named_label("", "note")
        self.workspace.setWordWrap(True)
        change = QPushButton("Change…")
        change.setObjectName("plain")
        change.setToolTip("Choose the folder new runs are made in, and listed from.")
        change.clicked.connect(self.workspace_wanted)
        where = QHBoxLayout()
        where.addWidget(self.workspace, 1)
        where.addWidget(change)
        side.addLayout(where)
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

    def show_workspace(self, folder: Path) -> None:
        """Say which folder the runs go in.

        Parameters
        ----------
        folder : pathlib.Path
            The folder.
        """
        home = Path.home()
        shown = (
            f"~/{folder.relative_to(home)}"
            if folder.is_relative_to(home)
            else str(folder)
        )
        self.workspace.setText(f"In {shown}")
        self.workspace.setToolTip(str(folder))

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
