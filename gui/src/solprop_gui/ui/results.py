# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Results step: one solprop results file, its views, and how they are drawn.

The panel holds the file's controls, the workspace's runs (any earlier run opens from
there), the plot options and what the Statistics view's panels show; the views
(Overview, Plane, Statistics, Scan) are shown by the window among its views. The
compare of several runs is kept (:meth:`ResultsPage.add_to_compare`) but not offered.

A file opens whether the window ran it or not: what it can show is asked of solprop
(``capabilities.output``), and its planes are reduced in the kernel. A file that
records its recipe can give its setup back ("Open its setup", the resolved
recipe, design 4).
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from solprop.service.messages import Reply

from solprop_gui.model.plot_options import PlotOptions
from solprop_gui.model.workspace import RunEntry
from solprop_gui.ui.bridge import ServiceBridge
from solprop_gui.ui.pages import Section
from solprop_gui.ui.plot_options import PlotOptionsPanel, SpectrogramOptions
from solprop_gui.views.compare import CompareView
from solprop_gui.views.live import OverviewView
from solprop_gui.views.plane import PlaneView
from solprop_gui.views.scan_browser import ScanView
from solprop_gui.views.statistics import StatisticsView, StatsSelector

__all__ = ["ResultsPage"]


class ResultsPage(QWidget):
    """Open a results file, list the workspace's runs, and choose how it is drawn.

    Parameters
    ----------
    bridge : solprop_gui.ui.bridge.ServiceBridge
        The service.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    setup_opened : Signal
        Emitted with (recipe text, results file) when the user opens a run's setup.
    view_wanted : Signal
        Emitted with the name of the view to bring forward.
    described : Signal
        Emitted with what the open file can give, the kernel's ``describe_output``.
    """

    setup_opened = Signal(str, str)
    view_wanted = Signal(str)
    described = Signal(object)

    def __init__(  # noqa: PLR0915 - the page's layout
        self, bridge: ServiceBridge, parent: QWidget | None = None
    ) -> None:
        """Make the page, with no file."""
        super().__init__(parent)
        self.bridge = bridge
        self.path: str | None = None
        self.capabilities: dict[str, object] | None = None
        self._pending: dict[int, tuple[str, str]] = {}
        self.options = PlotOptions()
        self.plot_options = PlotOptionsPanel("results")
        self.spectrogram_options = SpectrogramOptions()
        self.spectrogram_options.changed.connect(
            lambda gate, positions: self.plot_options.changed.emit(
                replace(self.options, gate_fwhm=gate, positions=positions)
            )
        )

        self.open_button = QPushButton("Open results…")
        self.open_button.setObjectName("open-results")
        self.open_button.clicked.connect(self.choose)
        self.setup_button = QPushButton("Open its setup")
        self.setup_button.setObjectName("open-setup")
        self.setup_button.setToolTip(
            "Loads the recipe the file records, as resolved: the values that rebuild "
            "this run."
        )
        self.setup_button.clicked.connect(self.open_setup)
        self.info = QLabel("No results file open.")
        self.info.setObjectName("results-info")
        self.info.setWordWrap(True)
        self.runs = QListWidget()
        self.runs.setObjectName("runs")
        self.runs.itemActivated.connect(
            lambda item: self.open(item.data(Qt.ItemDataRole.UserRole))
        )

        self.overview = OverviewView()
        self.compare = CompareView()
        self.plane = PlaneView()
        self.statistics = StatisticsView()
        self.scan = ScanView()
        self.plane.plane_wanted.connect(self._ask_plane)
        self.plane.spectrogram_wanted.connect(self._ask_spectrogram)
        self.plane.figure_wanted.connect(self._ask_figure)
        self.statistics.figure_wanted.connect(self._ask_figure)
        self.stats_selector = StatsSelector()
        self.stats_selector.changed.connect(self.statistics.set_layout)
        self.statistics.layout_changed.connect(self.stats_selector.set_layout)

        file = QWidget()
        buttons = QGridLayout(file)
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addWidget(self.open_button, 0, 0)
        buttons.addWidget(self.setup_button, 0, 1)
        buttons.addWidget(self.info, 1, 0, 1, 2)
        self.runs.setMinimumHeight(80)
        sections = QWidget()
        column = QVBoxLayout(sections)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        for title, widget, stretch in (
            ("Results file", file, 0),
            ("Runs in the workspace", self.runs, 1),
            ("Plot options", self.plot_options, 0),
            ("Spectrogram", self.spectrogram_options, 0),
            ("Statistics panels", self.stats_selector, 0),
        ):
            section = Section(title)
            section.add(widget)
            column.addWidget(section, stretch)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(sections)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(QLabel("<b>Results</b>"))
        layout.addWidget(scroll, 1)
        bridge.replied.connect(self.on_reply)
        self._update_buttons()

    def show_runs(self, runs: Sequence[RunEntry]) -> None:
        """List the workspace's runs.

        Parameters
        ----------
        runs : Sequence of RunEntry
            The runs, newest first.
        """
        self.runs.clear()
        for run in runs:
            if run.output is None:
                continue
            item = QListWidgetItem(run.name)
            item.setData(Qt.ItemDataRole.UserRole, str(run.output))
            item.setToolTip(str(run.output))
            self.runs.addItem(item)

    def _update_buttons(self) -> None:
        """Enable what the open file allows."""
        has_file = self.path is not None
        recipe = bool(self.capabilities and self.capabilities.get("recipe"))
        self.setup_button.setEnabled(has_file and recipe)

    def choose(self) -> None:
        """Pick a results file and open it."""
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Open results", "", "solprop results (*.h5 *.hdf5);;All files (*)"
        )
        if chosen:
            self.open(chosen)

    def _ask(self, kind: str, payload: dict[str, object], purpose: str) -> None:
        """Send a request about a file, remembering what it is for.

        Parameters
        ----------
        kind : str
            The handler.
        payload : dict
            Its arguments.
        purpose : str
            What the reply is for.
        """
        request = self.bridge.request(kind, payload, key=f"results-{purpose}")
        self._pending[request] = (purpose, str(payload.get("path")))

    def open(self, path: str | Path, *, show: bool = True) -> None:
        """Open a results file.

        Parameters
        ----------
        path : str or pathlib.Path
            The file.
        show : bool, optional
            Whether to bring its Overview forward.
        """
        self.path = str(path)
        self.capabilities = None
        self.info.setText(f"{_short(self.path)}: reading…")
        self.info.setToolTip(self.path)
        self.overview.clear()
        self.plane.clear()
        self.statistics.clear()
        self._update_buttons()
        if show:
            self.view_wanted.emit("results")
        self._ask("describe_output", {"path": self.path}, "describe")
        self._ask_overview()
        self._ask("statistics", {"path": self.path}, "statistics")

    def _ask_overview(self) -> None:
        """Ask for the open file's Overview, its power within the options' band."""
        if self.path is not None:
            payload = {"path": self.path, "bandpass": self.options.bandpass}
            self._ask("overview", payload, "overview")

    def apply_options(self, options: PlotOptions) -> None:
        """Take new plot options, asking solprop again where they change its data.

        The bandpass changes what solprop computes for the Overview, and the axis
        too for a plane; the rest only how the views draw.

        Parameters
        ----------
        options : PlotOptions
            The options.
        """
        band = self.options.bandpass != options.bandpass
        again = not self.options.same_request(options)
        spectrogram = self.options.spectrogram() != options.spectrogram()
        self.options = options
        self.plot_options.set_options(options)
        self.spectrogram_options.set_options(options)
        self.overview.apply_options(options)
        self.plane.apply_options(options)
        if self.path is None:
            return
        if band:
            self._ask_overview()
        if again and self.plane.planes:
            self._ask_plane(self.plane.index(), self.plane.selection())
        if spectrogram:
            self.plane.ask_spectrogram()

    def open_scan(self, path: str | Path, *, show: bool = True) -> None:
        """Open a scan's results file in the scan browser.

        Parameters
        ----------
        path : str or pathlib.Path
            The scan's file.
        show : bool, optional
            Whether to bring the browser forward.
        """
        self.info.setText(f"{_short(str(path))}: reading the scan…")
        self.scan.status.setText("Reading the scan…")
        if show:
            self.view_wanted.emit("scan")
        self._ask("scan_overview", {"path": str(path)}, "scan")

    def _ask_plane(self, index: int, modes: object) -> None:
        """Ask for one plane of the open file.

        Parameters
        ----------
        index : int
            The plane.
        modes : object
            The mode selection.
        """
        if self.path is not None:
            payload = {
                "path": self.path,
                "index": index,
                "modes": modes,
                **self.options.request(),
            }
            self._ask("plane", payload, "plane")

    def _ask_spectrogram(self, index: int, modes: object) -> None:
        """Ask for a plane's spectrogram, as the plot options set it.

        Parameters
        ----------
        index : int
            The plane.
        modes : object
            The mode selection; a sum takes the first mode.
        """
        if self.path is None or self.capabilities is None:
            return
        options = self.options
        if self.capabilities.get("basis") == "spatial":
            # A beam has no one field to gate: its intensity across the centre.
            body = {"path": self.path, "index": index, "bandpass": options.bandpass}
            self._ask("beam", body, "spectrogram")
            return
        low, high = options.time_limits
        payload = {
            "path": self.path,
            "index": index,
            "modes": modes,
            "axis": options.axis,
            "bandpass": options.bandpass,
            "time_range": None if low is None or high is None else (low, high),
            "gate_fwhm": options.gate_fwhm,
            "positions": options.positions,
        }
        self._ask("spectrogram", payload, "spectrogram")

    def _ask_figure(self, figure: Mapping[str, object]) -> None:
        """Ask for one of solprop's plots of the open file as a figure.

        Parameters
        ----------
        figure : Mapping
            The figure's ``file``, ``plot`` and its keys.
        """
        if self.path is not None:
            self._ask("result_figure", {"path": self.path, **figure}, "figure")

    def open_setup(self) -> None:
        """Ask for the recipe the open file records, as resolved."""
        if self.path is not None:
            self._ask("recipe_of", {"path": self.path, "resolved": True}, "setup")

    def add_to_compare(self) -> None:
        """Add the open file's run to the compare view."""
        if self.path is not None:
            self._ask("overview", {"path": self.path}, "compare")

    def on_reply(self, reply: Reply) -> None:
        """Take a kernel reply to one of the page's requests.

        Parameters
        ----------
        reply : solprop.service.messages.Reply
            The reply; replies to other requests are ignored.
        """
        entry = self._pending.pop(reply.request_id, None)
        if entry is None:
            return
        purpose, path = entry
        if purpose == "scan":
            self._show_scan(path, reply)
            return
        if not reply.ok:
            if purpose in ("plane", "spectrogram"):
                self.plane.show_error(f"{purpose}: {reply.error}")
            else:
                self.info.setText(f"{_short(path)}: {reply.error}")
            return
        value = reply.value
        if path != self.path and purpose != "compare":
            return  # a reply about the file shown before
        if purpose in ("plane", "spectrogram", "statistics", "figure"):
            self._show_view_reply(purpose, value)
            return
        if purpose == "describe" and path == self.path and isinstance(value, dict):
            self._show_description(path, value)
        elif purpose == "overview" and path == self.path and isinstance(value, dict):
            self.overview.set_rows(value.get("axes", {}), value.get("rows", []))  # pyright: ignore[reportArgumentType]
        elif purpose == "compare" and isinstance(value, dict):
            name = Path(path).parent.name
            self.compare.add_run(name, value.get("axes", {}), value.get("rows", []))  # pyright: ignore[reportArgumentType]
            self.view_wanted.emit("compare")
        elif purpose == "setup" and isinstance(value, str):
            self.setup_opened.emit(value, path)

    def _show_scan(self, path: str, reply: Reply) -> None:
        """Show a scan's overview in the browser, or why there is none.

        Parameters
        ----------
        path : str
            The scan's file.
        reply : solprop.service.messages.Reply
            The ``scan_overview`` reply.
        """
        if reply.ok and isinstance(reply.value, Mapping):
            self.scan.set_scan(reply.value)
            self.info.setText(f"{_short(path)}: a scan")
        else:
            self.scan.show_error(f"No scan: {reply.error}")

    def _show_description(self, path: str, value: dict[str, object]) -> None:
        """Show what a results file holds, and offer its planes and views.

        Parameters
        ----------
        path : str
            The file.
        value : dict
            The kernel's ``describe_output`` reply.
        """
        self.capabilities = value
        outcome = value.get("outcome") or {}
        status = outcome.get("status") if isinstance(outcome, dict) else None
        text = (
            f"{_short(path)}: "
            f"{value.get('simulation_kind')} run, {status or 'no outcome'}"
        )
        message = outcome.get("message") if isinstance(outcome, dict) else None
        if message:
            text += f" ({message})"
        planes = value.get("field_planes") or []
        text += f"; {len(planes)} planes with fields"  # pyright: ignore[reportArgumentType]
        if not value.get("recipe"):
            text += "; it records no recipe (a file from a script)"
        self.info.setText(text)
        self._update_buttons()
        z: list[float] = list(value.get("z") or [])  # pyright: ignore[reportArgumentType]
        planes = [int(p) for p in planes]  # pyright: ignore[reportGeneralTypeIssues]
        self.plane.set_planes(
            planes,
            [z[p] for p in planes if p < len(z)],
            list(value.get("labels") or []),  # pyright: ignore[reportArgumentType]
        )
        self.overview.set_spatial(value.get("basis") == "spatial")
        self.described.emit(value)

    def _show_view_reply(self, purpose: str, value: object) -> None:
        """Hand a view its reply.

        Parameters
        ----------
        purpose : str
            ``"plane"``, ``"spectrogram"``, ``"statistics"`` or
            ``"figure"``.
        value : object
            The reply's value.
        """
        if purpose == "figure":
            written: Sequence[str] = value if isinstance(value, list) else []
            self.info.setText("Wrote " + ", ".join(_short(f) for f in written))
        elif not isinstance(value, Mapping):
            return
        elif purpose == "plane":
            self.plane.show_plane(value)
        elif purpose == "spectrogram":
            self.plane.show_spectrogram(value)
        else:
            self.statistics.set_statistics(value)


def _short(path: str) -> str:
    """Name a results file by its folder and its name.

    Parameters
    ----------
    path : str
        The file.

    Returns
    -------
    str
        ``folder/name``.

    Examples
    --------
    >>> _short("/a/b/2026-11-02-rdw/run.h5")
    '2026-11-02-rdw/run.h5'
    """
    file = Path(path)
    return f"{file.parent.name}/{file.name}"
