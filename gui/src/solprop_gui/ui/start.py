# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The start page: an introduction, the geometry classes, and solprop's examples.

The welcome screen. It offers the geometry classes as a gallery of the manual's
figures (:mod:`solprop_gui.help.figures`); choosing one asks whether to begin from its
template or from one of solprop's examples of that class. The examples are listed
again below, grouped by geometry class, and files are opened from the top.
"""

from collections.abc import Mapping, Sequence

from PySide6.QtCore import QByteArray, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.help.figures import figure_text
from solprop_gui.registry.routes import CARDS, RouteCard, card, group_examples

__all__ = ["NewSetupDialog", "StartPage", "figure_pixmap"]

TILE = QSize(240, 214)
"""The size of a geometry class's picture in the gallery."""

_INTRO = (
    "<h2>solprop</h2>"
    "<p>Set up, run and read an ultrafast nonlinear propagation simulation: "
    "gas-filled capillaries and hollow fibres, solid fibres, waveguides and beams in "
    "free space. Choose a geometry class to begin, start from one of solprop's "
    "examples, or open a setup or a results file.</p>"
    "<p>Every setup is a solprop recipe: it saves as a recipe file, runs with "
    "<code>solprop run</code>, and exports as a Python script.</p>"
)


def _dark(widget: QWidget) -> bool:
    """Return whether a widget's palette is a dark one.

    Parameters
    ----------
    widget : PySide6.QtWidgets.QWidget
        The widget.

    Returns
    -------
    bool
        Whether its window colour is darker than mid-grey.
    """
    return widget.palette().color(QPalette.ColorRole.Window).lightness() < 128


def figure_pixmap(key: str, size: QSize, *, dark: bool = False) -> QPixmap | None:
    """Draw a geometry class's figure at a size.

    Parameters
    ----------
    key : str
        The card's key.
    size : PySide6.QtCore.QSize
        The picture's size; the figure keeps its aspect within it.
    dark : bool, optional
        Whether to draw it in the dark palette.

    Returns
    -------
    PySide6.QtGui.QPixmap or None
        The picture; ``None`` for a class without a figure.
    """
    text = figure_text(key, dark=dark)
    if text is None:
        return None
    renderer = QSvgRenderer(QByteArray(text.encode("utf-8")))
    image = QImage(size, QImage.Format.Format_ARGB32)
    image.fill(QColor(0, 0, 0, 0))
    painter = QPainter(image)
    try:
        renderer.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        renderer.render(painter)
    finally:
        painter.end()
    return QPixmap.fromImage(image)


class NewSetupDialog(QDialog):
    """Begin a geometry class from its template, or from one of its examples.

    Parameters
    ----------
    entry : RouteCard
        The geometry class.
    examples : Sequence of Mapping
        The examples of this class, solprop's: ``name`` and ``title``.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    choice : str or None
        After it is accepted: ``""`` for the template, or the example's name.
    """

    def __init__(
        self,
        entry: RouteCard,
        examples: Sequence[Mapping[str, object]],
        parent: QWidget | None = None,
    ) -> None:
        """List the template and the class's examples."""
        super().__init__(parent)
        self.setWindowTitle(f"New setup: {entry.title}")
        self.setObjectName("new-setup")
        self.choice: str | None = None
        heading = QLabel(f"<b>{entry.title}</b><br>{entry.summary}")
        heading.setWordWrap(True)
        self.options = QListWidget()
        self.options.setObjectName("new-setup-options")
        template = QListWidgetItem(
            "Template: the class's starting setup, to fill in"
            if entry.example is None
            else f"Template: from solprop's example '{entry.example}'"
        )
        template.setData(Qt.ItemDataRole.UserRole, "")
        self.options.addItem(template)
        for example in examples:
            if example["name"] == entry.example:
                continue  # it is the template
            item = QListWidgetItem(
                f"Example: {example.get('title') or example['name']}"
            )
            item.setData(Qt.ItemDataRole.UserRole, str(example["name"]))
            item.setToolTip(str(example["name"]))
            self.options.addItem(item)
        self.options.setCurrentRow(0)
        self.options.itemActivated.connect(lambda _item: self.accept())
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(self.options, 1)
        layout.addWidget(buttons)
        self.resize(520, 360)

    def accept(self) -> None:
        """Take the option picked."""
        item = self.options.currentItem()
        self.choice = "" if item is None else str(item.data(Qt.ItemDataRole.UserRole))
        super().accept()


class StartPage(QWidget):
    """The welcome screen: an introduction, the geometry classes and the examples.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    card_chosen : Signal
        Emitted with a geometry class's card key when its tile is clicked.
    example_chosen : Signal
        Emitted with an example's name.
    """

    card_chosen = Signal(str)
    example_chosen = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the page; the examples are filled in when the kernel lists them."""
        super().__init__(parent)
        self.grouped: dict[str, list[Mapping[str, object]]] = {}
        self.tiles: dict[str, QToolButton] = {}
        intro = QLabel(_INTRO)
        intro.setObjectName("start-intro")
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        self.open_button = QPushButton("Open setup…")
        self.open_button.setObjectName("open-setup-file")
        self.results_button = QPushButton("Open results…")
        self.results_button.setObjectName("open-results-file")
        self.explorer_button = QPushButton("Waveguide explorer")
        self.explorer_button.setObjectName("explorer")
        self.explorer_button.setToolTip(
            "A capillary to explore: its modes' dispersion, loss and nonlinearity, "
            "swept over a keyword, before any run."
        )
        files = QHBoxLayout()
        for button in (self.open_button, self.results_button, self.explorer_button):
            files.addWidget(button)
        files.addStretch(1)

        gallery = self._gallery()

        self.examples = QTreeWidget()
        self.examples.setObjectName("examples")
        self.examples.setHeaderHidden(True)
        self.examples.itemActivated.connect(self._example_activated)

        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(12, 12, 12, 12)
        column.setSpacing(10)
        column.addWidget(intro)
        column.addLayout(files)
        column.addWidget(QLabel("<h3>Choose a geometry class</h3>"))
        column.addWidget(gallery)
        column.addWidget(QLabel("<h3>Or start from one of solprop's examples</h3>"))
        column.addWidget(self.examples)
        column.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    def _gallery(self) -> QWidget:
        """Make the geometry classes' tiles, each the manual's figure of it.

        Returns
        -------
        PySide6.QtWidgets.QWidget
            The gallery.
        """
        gallery = QWidget()
        grid = QGridLayout(gallery)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)
        dark = _dark(self)
        for index, entry in enumerate(CARDS):
            tile = QToolButton()
            tile.setObjectName(f"card-{entry.key}")
            tile.setText(entry.title)
            tile.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            tile.setIconSize(TILE)
            tile.setAutoRaise(True)
            tile.setToolTip(entry.reason or entry.summary)
            tile.setEnabled(entry.enabled)
            pixmap = figure_pixmap(entry.key, TILE, dark=dark)
            if pixmap is not None:
                tile.setIcon(QIcon(pixmap))
            tile.clicked.connect(
                lambda _checked=False, key=entry.key: self.card_chosen.emit(key)
            )
            self.tiles[entry.key] = tile
            grid.addWidget(tile, index // 4, index % 4)
        return gallery

    def show_examples(
        self, examples: Sequence[Mapping[str, object]], routes: Sequence[str]
    ) -> None:
        """List solprop's examples by geometry class, those the window offers enabled.

        Parameters
        ----------
        examples : Sequence of Mapping
            ``name``, ``title``, ``route`` and ``build`` of each.
        routes : Sequence of str
            The routes the window can show.
        """
        self.grouped = group_examples(examples)
        self.examples.clear()
        for key, found in self.grouped.items():
            entry = None if key == "other" else card(key)
            title = "Other" if entry is None else entry.title
            group = QTreeWidgetItem([f"{title} ({len(found)})"])
            group.setFlags(group.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            usable = entry is not None and entry.enabled
            for example in found:
                item = QTreeWidgetItem([str(example.get("title") or example["name"])])
                item.setData(0, Qt.ItemDataRole.UserRole, example["name"])
                item.setToolTip(0, f"{example['name']} ({example['route']})")
                if not usable or example["route"] not in routes:
                    item.setDisabled(True)
                    item.setToolTip(0, entry.reason if entry and entry.reason else "")
                group.addChild(item)
            self.examples.addTopLevelItem(group)
            group.setExpanded(usable)
        self.examples.setMinimumHeight(min(420, 26 * (len(examples) + 4)))

    def examples_of(self, key: str) -> list[Mapping[str, object]]:
        """Return the examples of a geometry class.

        Parameters
        ----------
        key : str
            The card's key.

        Returns
        -------
        list of Mapping
            Its examples, as listed.
        """
        return list(self.grouped.get(key, []))

    def _example_activated(self, item: QTreeWidgetItem, _column: int) -> None:
        """Begin from an example picked in the list.

        Parameters
        ----------
        item : PySide6.QtWidgets.QTreeWidgetItem
            The item.
        _column : int
            Its column, unused.
        """
        name = item.data(0, Qt.ItemDataRole.UserRole)
        if name:
            self.example_chosen.emit(str(name))
