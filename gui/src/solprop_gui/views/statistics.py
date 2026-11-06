# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Statistics view: the recorded statistics against z, four panels of two.

The statistics a run recorded (``stats/*``) come with solprop's own summary and unit
(``solprop.stats.CATALOGUE``). The view is a 2 by 2 grid; each panel draws one
quantity on its left axis and optionally a second on its right, dashed, which is how
two statistics are read against each other. A quantity is a statistic, a column per
mode and component where it has several, or the run's step size with its rejected
steps marked (:data:`~solprop_gui.model.stats_layout.STEP`), which is where a run's
cost and its difficult stretches show. What each panel shows is chosen on the Results
step (:class:`StatsSelector`).
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.model.stats_layout import (
    PANELS,
    STEP,
    Layout,
    default_layout,
    kept_layout,
)
from solprop_gui.views.plotting import PALETTE, HoverLayout, choose_figure, curve_lookup

__all__ = ["StatisticsView", "StatsSelector"]

_PLAIN = ("J", "W", "s", "m", "rad")
"""Units pyqtgraph may prefix (GW, fs); any other goes in the label as it is."""


class _Panel:
    """One panel: a plot with a left axis, and a second view box on its right.

    Parameters
    ----------
    layout : solprop_gui.views.plotting.HoverLayout
        The grid it is drawn in.
    row, column : int
        Its place.
    """

    def __init__(self, layout: HoverLayout, row: int, column: int) -> None:
        """Make the plot and its right-hand view box, linked in z."""
        self.plot = layout.add_plot(row, column, "")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel("bottom", "z (cm)")
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self.right = pg.ViewBox()
        self.plot.showAxis("right")
        scene = self.plot.scene()
        if scene is not None:
            scene.addItem(self.right)
        self.plot.getAxis("right").linkToView(self.right)
        self.right.setXLink(self.plot)
        self.plot.getViewBox().sigResized.connect(self._fit)
        self.items: list[pg.GraphicsObject] = []
        self.curves: list[pg.PlotDataItem] = []
        layout.add_readout(self.plot, self.lookup, dot=True)

    def _fit(self) -> None:
        """Lay the right view box over the plot's own."""
        self.right.setGeometry(self.plot.getViewBox().sceneBoundingRect())
        self.right.linkedViewChanged(self.plot.getViewBox(), self.right.XAxis)

    def clear(self) -> None:
        """Remove what the panel draws."""
        for item in self.items:
            box = item.getViewBox()
            if box is not None:
                box.removeItem(item)
        self.items = []
        self.curves = []
        self.legend.clear()
        self.plot.setTitle("")
        for side in ("left", "right"):
            axis = self.plot.getAxis(side)
            axis.setLabel("")
            axis.setLogMode(False, False)
        self.plot.getAxis("right").setStyle(showValues=False)

    def lookup(self, x: float, y: float) -> tuple[float, float, str] | None:
        """Read the left quantity's curve nearest the mouse.

        Parameters
        ----------
        x, y : float
            The mouse, in the plot's coordinates.

        Returns
        -------
        tuple or None
            The point and its label; ``None`` with nothing drawn.
        """
        best: tuple[float, float, str] | None = None
        distance = np.inf
        for curve in self.curves:
            hit = curve_lookup(curve, "z = {0:.4g} cm: {1:.4g}")(x, y)
            if hit is not None and abs(hit[1] - y) < distance:
                distance = abs(hit[1] - y)
                best = hit
        return best


class StatisticsView(QWidget):
    """A results file's statistics against z, in four panels of up to two each.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    figure_wanted : Signal
        Emitted with a figure's ``file`` and ``plot`` (solprop's ``plot_stats``).
    layout_changed : Signal
        Emitted with the :data:`~solprop_gui.model.stats_layout.Layout` shown, the
        quantities to choose from and their descriptions, when a file's statistics
        arrive.
    """

    figure_wanted = Signal(object)
    layout_changed = Signal(object, object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the empty panels."""
        super().__init__(parent)
        self.stats: dict[str, Mapping[str, object]] = {}
        self.z = np.zeros(0)
        self.steps: Mapping[str, object] = {}
        self.chosen: Layout | None = None
        self.summary = QLabel("No results file open.")
        self.summary.setObjectName("stats-summary")
        self.summary.setWordWrap(True)
        self.figure_button = QPushButton("Export figures…")
        self.figure_button.setToolTip("Every statistic, drawn by solprop (plot_stats).")
        self.figure_button.clicked.connect(self._choose_figure)
        self.plots = HoverLayout()
        self.panels = [_Panel(self.plots, index // 2, index % 2) for index in range(4)]
        line = QHBoxLayout()
        line.addWidget(self.summary, 1)
        line.addWidget(self.figure_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.addLayout(line)
        layout.addWidget(self.plots, 1)

    @property
    def names(self) -> list[str]:
        """Return the quantities a panel can show.

        Returns
        -------
        list of str
            The recorded statistics, then the step size.
        """
        return [*self.stats, STEP]

    def panel_choice(self, index: int) -> tuple[str | None, str | None]:
        """Return what a panel shows.

        Parameters
        ----------
        index : int
            The panel, row by row.

        Returns
        -------
        tuple
            Its left and right quantities; ``None`` for none.
        """
        return (self.chosen or default_layout([]))[index]

    def set_statistics(self, reply: Mapping[str, object]) -> None:
        """Show a file's statistics and step history.

        Parameters
        ----------
        reply : Mapping
            The kernel's ``statistics`` reply.
        """
        self.z = np.asarray(reply["z"], dtype=float)
        entries: Sequence[Mapping[str, object]] = reply["stats"]  # pyright: ignore[reportAssignmentType]
        self.stats = {str(entry["name"]): entry for entry in entries}
        self.steps = reply["steps"]  # pyright: ignore[reportAttributeAccessIssue]
        names = list(self.stats)
        self.chosen = (
            default_layout(names)
            if self.chosen is None
            else kept_layout(self.chosen, names)
        )
        self.figure_button.setEnabled(bool(self.stats))
        accepted = len(self.steps.get("z", []))  # pyright: ignore[reportArgumentType]
        rejected = len(self.steps.get("rejected_z", []))  # pyright: ignore[reportArgumentType]
        self.summary.setText(
            f"{len(self.stats)} statistics at {self.z.size} planes; "
            f"{accepted} accepted steps, {rejected} rejected."
        )
        self.redraw()
        self.layout_changed.emit(self.chosen, self.names, self.summaries())

    def summaries(self) -> dict[str, str]:
        """Return each quantity's description, as solprop gives it.

        Returns
        -------
        dict
            By name: the summary and unit.
        """
        found = {
            name: f"{entry['summary']} ({entry['units'] or 'no unit'})"
            for name, entry in self.stats.items()
        }
        found[STEP] = "The run's step size against z, the rejected steps marked."
        return found

    def set_layout(self, layout: Layout) -> None:
        """Show a layout chosen.

        Parameters
        ----------
        layout : Layout
            The four panels' left and right quantities.
        """
        self.chosen = tuple(layout)
        self.redraw()

    def redraw(self) -> None:
        """Draw every panel; a panel with nothing chosen gives its space to the rest."""
        layout = self.chosen or default_layout([])
        self._arrange([left is not None or right is not None for left, right in layout])
        for panel, (left, right) in zip(self.panels, layout, strict=True):
            panel.clear()
            titles = []
            for side, name in (("left", left), ("right", right)):
                if name is not None and (name == STEP or name in self.stats):
                    titles.append(self._draw(panel, side, name))
            panel.plot.setTitle(" · ".join(titles))
            panel.plot.getViewBox().autoRange()
            panel.right.autoRange()

    def _arrange(self, shown: Sequence[bool]) -> None:
        """Lay out the panels shown: one or two in a column, more in the grid.

        Parameters
        ----------
        shown : Sequence of bool
            Whether each panel is shown.
        """
        grid = self.plots.ci
        for panel in self.panels:
            if panel.plot in grid.items:
                grid.removeItem(panel.plot)
        visible = [panel for panel, on in zip(self.panels, shown, strict=True) if on]
        for place, panel in enumerate(visible):
            row, column = (place, 0) if len(visible) <= 2 else divmod(place, 2)
            grid.addItem(panel.plot, row=row, col=column)
        for panel, on in zip(self.panels, shown, strict=True):
            panel.plot.setVisible(on)
            panel.right.setVisible(on)

    def _draw(self, panel: _Panel, side: str, name: str) -> str:
        """Draw one quantity on one side of a panel.

        Parameters
        ----------
        panel : _Panel
            The panel.
        side : {'left', 'right'}
            Its axis.
        name : str
            The statistic, or :data:`STEP`.

        Returns
        -------
        str
            Its name, for the panel's title (solprop's summary is too long for
            one; the selector shows it).
        """
        box = panel.plot.getViewBox() if side == "left" else panel.right
        style = Qt.PenStyle.SolidLine if side == "left" else Qt.PenStyle.DashLine
        offset = 0 if side == "left" else 3
        axis = panel.plot.getAxis(side)
        axis.setStyle(showValues=True)
        if name == STEP:
            return self._draw_steps(panel, box, axis, style)
        entry = self.stats[name]
        units = str(entry["units"])
        plain = units in _PLAIN
        axis.setLabel(
            name if plain or not units else f"{name} ({units})",
            units=units if plain else None,
        )
        columns: Sequence[Mapping[str, object]] = entry["columns"]  # pyright: ignore[reportAssignmentType]
        for number, column in enumerate(columns):
            colour = PALETTE[(number + offset) % len(PALETTE)]
            curve = pg.PlotDataItem(
                self.z * 100,
                np.asarray(column["y"], dtype=float),
                pen=pg.mkPen(colour, width=1.4, style=style),
                symbol="o",
                symbolSize=4,
                symbolBrush=colour,
            )
            box.addItem(curve)
            panel.items.append(curve)
            label = str(column["label"]) if len(columns) > 1 else name
            panel.legend.addItem(curve, label)
            if side == "left":
                panel.curves.append(curve)
        return name

    def _draw_steps(
        self,
        panel: _Panel,
        box: pg.ViewBox,
        axis: pg.AxisItem,
        style: Qt.PenStyle,
    ) -> str:
        """Draw the step size, on a log axis, with the rejected steps marked.

        Parameters
        ----------
        panel : _Panel
            The panel.
        box : pyqtgraph.ViewBox
            The view box of its side.
        axis : pyqtgraph.AxisItem
            The side's axis.
        style : PySide6.QtCore.Qt.PenStyle
            The line's style.

        Returns
        -------
        str
            The title.
        """
        steps = self.steps
        z = np.asarray(steps.get("z", []), dtype=float) * 100
        h = np.asarray(steps.get("h", []), dtype=float)
        rejected_z = np.asarray(steps.get("rejected_z", []), dtype=float) * 100
        rejected_h = np.asarray(steps.get("rejected_h", []), dtype=float)
        axis.setLogMode(False, True)  # pyright: ignore[reportArgumentType] - its stub
        axis.setLabel("step", units="m")
        curve = pg.PlotDataItem(
            z,
            np.log10(np.maximum(h, 1e-30)),
            pen=pg.mkPen(PALETTE[0], width=1.0, style=style),
            connect="finite",
        )
        marks = pg.ScatterPlotItem(
            rejected_z,
            np.log10(np.maximum(rejected_h, 1e-30)),
            symbol="x",
            size=7,
            pen=pg.mkPen("#c62828"),
        )
        for item in (curve, marks):
            box.addItem(item)
            panel.items.append(item)
        panel.legend.addItem(curve, "step, accepted")
        panel.legend.addItem(marks, "rejected")
        return STEP

    def _choose_figure(self) -> None:  # pragma: no cover - a modal dialog
        """Ask where to write the figures, and ask for them."""
        path = choose_figure(self, "statistics.pdf")
        if path:
            self.export_figure(path)

    def export_figure(self, path: str) -> None:
        """Ask for solprop's figures of every statistic.

        Parameters
        ----------
        path : str
            The file (numbered for several).
        """
        self.figure_wanted.emit({"file": path, "plot": "plot_stats"})

    def clear(self) -> None:
        """Forget the file shown."""
        self.stats = {}
        self.z = np.zeros(0)
        self.steps = {}
        for panel in self.panels:
            panel.clear()
        self.figure_button.setEnabled(False)
        self.summary.setText("No results file open.")


class StatsSelector(QWidget):
    """Choose what each of the Statistics view's panels shows.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    changed : Signal
        Emitted with the new :data:`~solprop_gui.model.stats_layout.Layout`.
    """

    changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make a left and a right choice for each panel."""
        super().__init__(parent)
        self.choices: list[tuple[QComboBox, QComboBox]] = []
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(QLabel("left axis"), 0, 1)
        grid.addWidget(QLabel("right axis"), 0, 2)
        places = ("Top left", "Top right", "Bottom left", "Bottom right")
        for index, place in enumerate(places[:PANELS]):
            pair = (QComboBox(), QComboBox())
            for side, box in zip(("left", "right"), pair, strict=True):
                box.setObjectName(f"stats-{index}-{side}")
                box.setSizeAdjustPolicy(
                    QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
                )
                box.setMinimumContentsLength(8)
                box.currentIndexChanged.connect(self._picked)
            grid.addWidget(QLabel(place), index + 1, 0)
            grid.addWidget(pair[0], index + 1, 1)
            grid.addWidget(pair[1], index + 1, 2)
            self.choices.append(pair)
        self.set_layout(default_layout([]), [STEP])

    def set_layout(
        self,
        layout: Layout,
        names: Sequence[str],
        summaries: Mapping[str, str] | None = None,
    ) -> None:
        """Show a layout and the quantities to choose from, without announcing it.

        Parameters
        ----------
        layout : Layout
            The four panels' choices.
        names : Sequence of str
            The quantities.
        summaries : Mapping or None, optional
            Each quantity's description, shown as its tooltip.
        """
        summaries = summaries or {}
        for pair, panel in zip(self.choices, layout, strict=True):
            for box, name in zip(pair, panel, strict=True):
                box.blockSignals(True)
                box.clear()
                box.addItem("(none)", None)
                for option in names:
                    box.addItem(option, option)
                    box.setItemData(
                        box.count() - 1,
                        summaries.get(option, ""),
                        Qt.ItemDataRole.ToolTipRole,
                    )
                box.setCurrentIndex(max(box.findData(name), 0))
                box.blockSignals(False)

    def chosen(self) -> Layout:
        """Return the layout chosen.

        Returns
        -------
        Layout
            The four panels' left and right quantities.
        """
        return tuple(
            (left.currentData(), right.currentData()) for left, right in self.choices
        )

    def _picked(self, _index: int) -> None:
        """Announce the layout after a choice.

        Parameters
        ----------
        _index : int
            The choice's place, unused.
        """
        self.changed.emit(self.chosen())
