# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run and Analyse: one page under a progress strip, or a rail of pages (design 10, 11).

The Run stage is the class's Overview page with a progress strip, so what you watch grow
is what you then analyse: both stages are this one widget, and finishing a run takes
the strip away and brings the rail in.
"""

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.classes.problem import AnalysisPage, page_refusal
from solprop_gui.model.display import duration_text
from solprop_gui.model.snippets import snippet
from solprop_gui.registry.units import si
from solprop_gui.ui.analysis import AnalysisSession
from solprop_gui.ui.analysis_pages import AlongPage, OverviewPage, PageView
from solprop_gui.ui.run_panel import RunPanel
from solprop_gui.ui.theme import clear_layout, named_label, repolish

__all__ = ["AnalyseStage", "ZScrubber"]

_PANEL_WIDTH = 232
"""The width of the controls beside a page's plots, in pixels."""

_WIDE_PANEL = 360
"""The width of controls that hold a form, as a compressor chain's do."""


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
        self.follow = QCheckBox("Follow the newest plane")
        self.follow.setChecked(True)
        self.follow.hide()
        row.addWidget(self.follow)

    def set_positions(self, positions: Sequence[float], *, keep: bool = False) -> None:
        """Take the planes' positions, and go to the last.

        Parameters
        ----------
        positions : Sequence of float
            Where each plane is, m.
        keep : bool, optional
            Stay on the plane selected, as a run's newer planes arrive.
        """
        place = self.slider.value()
        self.positions = [float(z) for z in positions]
        last = max(len(self.positions) - 1, 0)
        self.slider.blockSignals(True)
        self.slider.setRange(0, last)
        self.slider.setValue(min(place, last) if keep else last)
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
    """A progress strip or a rail, one page large, its controls, and one z scrubber.

    Parameters
    ----------
    session : solprop_gui.ui.analysis.AnalysisSession
        The open file.
    panel : solprop_gui.ui.run_panel.RunPanel
        The run's panel, which is never shown itself.
    views : Mapping
        The page of each key that has one; any other page says it is not built.
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
    running : bool
        Whether the stage shows a run going, rather than the rail.
    overview : OverviewPage or None
        The page a run draws on.
    """

    def __init__(  # noqa: PLR0915 - the stage's parts
        self,
        session: AnalysisSession,
        panel: RunPanel | None,
        views: Mapping[str, QWidget],
        parent: QWidget | None = None,
    ) -> None:
        """Make the frame, with no pages."""
        super().__init__(parent)
        self.setObjectName("page")
        self.session = session
        self.views = dict(views)
        self.buttons: dict[str, QPushButton] = {}
        self.pages: list[AnalysisPage] = []
        self.current: str | None = None
        self.running = False
        self.overview = cast(
            "OverviewPage | None",
            next((v for v in self.views.values() if isinstance(v, OverviewPage)), None),
        )
        self._started = 0.0

        self.strip = QFrame()
        self.strip.setObjectName("strip")
        strip = QHBoxLayout(self.strip)
        strip.setContentsMargins(16, 8, 16, 8)
        strip.setSpacing(14)
        self.eta = named_label("", "note")
        if panel is not None:
            panel.state_label.setWordWrap(False)
            strip.addWidget(panel.state_label)
            strip.addWidget(panel.bar, 1)
            strip.addWidget(self.eta)
            strip.addWidget(panel.steps)
            strip.addWidget(panel.sparkline)
            strip.addWidget(panel.lamp)
            panel.bar.valueChanged.connect(self._progressed)
        self.strip.hide()
        self.advice = QFrame()
        self.advice.setObjectName("advice")
        advice = QVBoxLayout(self.advice)
        self.advice_title = named_label("", "h3")
        self.advice_text = named_label("", "note")
        self.advice_text.setWordWrap(True)
        advice.addWidget(self.advice_title)
        advice.addWidget(self.advice_text)
        self.advice.hide()

        self.rail = QFrame()
        self.rail.setObjectName("rail")
        self.rail.setFixedWidth(170)
        self.items = QVBoxLayout(self.rail)
        self.items.setContentsMargins(8, 10, 8, 10)
        self.items.setSpacing(2)
        self.title = named_label("", "h2")
        self.exports = QToolButton()
        self.exports.setObjectName("plain")
        self.exports.setText("Export ▾")
        self.exports.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.export_menu = QMenu(self.exports)
        self.exports.setMenu(self.export_menu)
        self.export_menu.aboutToShow.connect(self._fill_exports)
        self.said = named_label("", "note")
        self.stack = QStackedWidget()
        for view in self.views.values():
            self.stack.addWidget(view)
        self.unbuilt = named_label("", "note")
        self.unbuilt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stack.addWidget(self.unbuilt)
        head = QHBoxLayout()
        head.addWidget(self.title)
        head.addWidget(self.said, 1)
        head.addWidget(self.exports)
        self.head = QWidget()
        self.head.setObjectName("page")
        self.head.setLayout(head)
        head.setContentsMargins(0, 0, 0, 0)
        page = QVBoxLayout()
        page.setContentsMargins(16, 10, 12, 8)
        page.setSpacing(8)
        page.addWidget(self.head)
        page.addWidget(self.stack, 1)
        self.side = QFrame()
        self.side.setObjectName("panel")
        self.side.setFixedWidth(_PANEL_WIDTH)
        self.side_box = QVBoxLayout()
        self.side_box.setContentsMargins(14, 14, 14, 14)
        holder = QWidget()
        holder.setObjectName("page")
        holder.setLayout(self.side_box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(holder)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side = QVBoxLayout(self.side)
        side.setContentsMargins(0, 0, 0, 0)
        side.addWidget(scroll)
        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)
        middle.addWidget(self.rail)
        middle.addLayout(page, 1)
        middle.addWidget(self.side)
        self.scrubber = ZScrubber()
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self.strip)
        column.addWidget(self.advice)
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
        self.scrubber.moved.connect(self._scrubbed)
        session.plane_moved.connect(lambda: self.scrubber.go_to(session.place))
        session.described.connect(lambda _value: self._take_planes())
        for view in self.views.values():
            if isinstance(view, OverviewPage):
                view.rows_grew.connect(self._rows_grew)
        self.scrubber.follow.toggled.connect(self._follow_toggled)

    # ------------------------------------------------------------------ the rail

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
        """Show one page large, with its controls beside it.

        Parameters
        ----------
        key : str
            The page.
        """
        self.current = key
        title = next((page.title for page in self.pages if page.key == key), "")
        self.title.setText(title)
        view = self._view_of(key)
        if view is not None:
            self.stack.setCurrentWidget(view)
        else:
            self.unbuilt.setText(f"{title}: not built yet.")
            self.stack.setCurrentWidget(self.unbuilt)
        self._show_controls(view)
        self.exports.setVisible(isinstance(view, PageView))
        for name, button in self.buttons.items():
            button.setProperty("current", name == key)
            repolish(button)

    def _view_of(self, key: str | None) -> QWidget | None:
        """Return what draws a page.

        Parameters
        ----------
        key : str or None
            The page's key.

        Returns
        -------
        PySide6.QtWidgets.QWidget or None
            Its view, if it has one.
        """
        page = next((page for page in self.pages if page.key == key), None)
        name = key if page is None or page.view is None else page.view
        return self.views.get(name or "")

    def _show_controls(self, view: QWidget | None) -> None:
        """Put a page's controls beside it.

        Parameters
        ----------
        view : PySide6.QtWidgets.QWidget or None
            The page.
        """
        while self.side_box.count():
            item = self.side_box.takeAt(0)
            shown = None if item is None else item.widget()
            if shown is not None:
                shown.hide()
                shown.setParent(None)  # pyright: ignore[reportArgumentType, reportCallIssue]
        controls = view.controls if isinstance(view, PageView) else None  # pyright: ignore[reportOptionalMemberAccess]
        self.side.setVisible(controls is not None)
        if controls is not None:
            wide = controls.minimumSizeHint().width() > _PANEL_WIDTH - 28
            self.side.setFixedWidth(_WIDE_PANEL if wide else _PANEL_WIDTH)
            self.side_box.addWidget(controls)
            controls.show()

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

    def set_reference(self, z: float | None) -> None:
        """Say where a reference distance lies, for the pages along z to mark.

        Parameters
        ----------
        z : float or None
            The distance, m; ``None`` for none.
        """
        for view in self.views.values():
            if isinstance(view, AlongPage):
                view.set_reference(z)

    def set_wanted(self, statistics: Sequence[str]) -> None:
        """Say which statistics a class shows first.

        Parameters
        ----------
        statistics : Sequence of str
            Their names.
        """
        for view in self.views.values():
            if isinstance(view, AlongPage):
                cast("AlongPage", view).wanted = tuple(statistics)

    # ------------------------------------------------------------------ running

    def show_running(self, running: bool) -> None:
        """Show a run going, on the Overview under its strip, or the rail.

        Parameters
        ----------
        running : bool
            Whether it is the Run stage.
        """
        self.running = running
        self.strip.setVisible(running)
        self.rail.setVisible(not running)
        self.head.setVisible(not running)
        self.scrubber.follow.setVisible(running)
        if running and self.overview is not None:
            self.stack.setCurrentWidget(self.overview)
            self._show_controls(self.overview)
        elif not running and self.current is not None:
            self.show_page(self.current)

    def run_started(self) -> None:
        """Start the clock of a new run, and take any advice away."""
        self._started = time.monotonic()
        self.eta.clear()
        self.advice.hide()
        self.scrubber.set_positions([])
        self.scrubber.follow.setChecked(True)

    def _progressed(self, value: int) -> None:
        """Say how long the run has left, from how long it has taken.

        Parameters
        ----------
        value : int
            How far it is, in thousandths.
        """
        if self._started and 0 < value < 1000:
            taken = time.monotonic() - self._started
            left = taken * (1000 - value) / value
            self.eta.setText(f"about {duration_text(left)} left")
        else:
            self.eta.clear()

    def _rows_grew(self, positions: Sequence[float]) -> None:
        """Take a run's newest plane.

        Parameters
        ----------
        positions : Sequence of float
            Where its planes are, m.
        """
        following = self.scrubber.follow.isChecked()
        self.scrubber.set_positions(positions, keep=not following)

    def _follow_toggled(self, following: bool) -> None:
        """Go to the newest plane when asked to follow it.

        Parameters
        ----------
        following : bool
            Whether to follow.
        """
        if self.overview is not None:
            self.overview.following = following
            if following and self.overview.rows:
                self.scrubber.go_to(len(self.overview.rows) - 1)
                self.overview.show_place(len(self.overview.rows) - 1)

    def show_advice(self, title: str, message: str) -> None:
        """Say why a run stopped short; the planes it reached stay open.

        Parameters
        ----------
        title : str
            What happened.
        message : str
            What to try.
        """
        self.advice_title.setText(title)
        self.advice_text.setText(message)
        self.advice.show()

    # ------------------------------------------------------------------ the scrubber

    def _take_planes(self) -> None:
        """Offer the open file's planes."""
        self.scrubber.set_positions(self.session.positions)

    def _scrubbed(self, place: int) -> None:
        """Take a move of the scrubber.

        Parameters
        ----------
        place : int
            The plane's place.
        """
        if self.running and self.overview is not None:
            self.scrubber.follow.setChecked(False)
            self.overview.show_place(place)
        else:
            self.session.move_to(place)

    # ------------------------------------------------------------------ exporting

    def _page(self) -> PageView | None:
        """Return the page on screen.

        Returns
        -------
        PageView or None
            The page, if a built one is.
        """
        view = self._view_of(self.current)
        return view if isinstance(view, PageView) else None

    def _fill_exports(self) -> None:
        """Offer what the page on screen can export."""
        self.export_menu.clear()
        page = self._page()
        if page is None:
            return
        # solprop's own figures read a run's modes; a beam's or a planar run's have
        # none, so its page is saved as drawn.
        spatial = self.session.capabilities.get("basis") == "spatial"
        if page.figure() is not None and not spatial:
            self.export_menu.addAction("Publication figure…", self.export_figure)
        self.export_menu.addAction("This page as an image…", self.choose_page_image)
        if page.data() is not None:
            self.export_menu.addAction("Data as CSV…", self.export_data)
        if page.python() is not None:
            self.export_menu.addAction("Copy as Python", self.copy_python)

    def export_figure(self, path: str | None = None) -> None:
        """Have solprop draw the page as a publication figure.

        Parameters
        ----------
        path : str or None, optional
            The file; ``None`` asks.
        """
        page = self._page()
        figure = None if page is None else page.figure()
        if figure is None or self.session.path is None:
            return
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export figure", "figure.pdf", "Figures (*.pdf *.svg *.png)"
            )
        if path:
            self.session.ask(
                "result_figure",
                {"path": self.session.path, "file": path, **figure},
                lambda written: self.said.setText(
                    "Wrote " + ", ".join(Path(str(w)).name for w in written)
                ),
                key="page-figure",
                failed=lambda reason: self.said.setText(
                    f"The figure was not written: {reason}"
                ),
            )

    def choose_page_image(self) -> None:  # pragma: no cover - a modal dialog
        """Ask where to save the page as drawn, and save it."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save this page", "page.png", "PNG (*.png)"
        )
        if path:
            self.save_page_image(path)

    def save_page_image(self, path: str) -> None:
        """Save the page on screen as it is drawn.

        Parameters
        ----------
        path : str
            The file, a PNG.
        """
        page = self._page()
        if page is not None and page.grab().save(path):
            self.said.setText(f"Wrote {Path(path).name}")

    def export_data(self, path: str | None = None) -> None:
        """Write what the page shows as a CSV file.

        Parameters
        ----------
        path : str or None, optional
            The file; ``None`` asks.
        """
        page = self._page()
        found = None if page is None else page.data()
        if found is None:
            return
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export data", "data.csv", "CSV (*.csv)"
            )
        if path:
            headers, values = found
            np.savetxt(path, values, delimiter=",", header=",".join(headers))
            self.said.setText(f"Wrote {Path(path).name}")

    def python_text(self) -> str | None:
        """Return the script that reproduces the page on screen.

        Returns
        -------
        str or None
            The script; ``None`` for a page with none.
        """
        page = self._page()
        key = None if page is None else page.python()
        path = self.session.path
        if key is None or path is None:
            return None
        return snippet(
            key,
            path,
            z=self.session.z,
            plane=self.session.index,
            modes=self.session.mode,
        )

    def copy_python(self) -> None:
        """Copy the script that reproduces the page on screen."""
        text = self.python_text()
        if text is not None:
            QGuiApplication.clipboard().setText(text)
            self.said.setText("Copied the page as Python.")
