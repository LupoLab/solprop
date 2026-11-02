# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Results page: any solprop results file, its Overview, and a compare.

A file opens whether the window ran it or not: what it can show is asked of solprop
(``capabilities.output``), and its planes are reduced in the kernel. A file that
records its recipe can give its setup back ("Open setup from this run", the resolved
recipe, design 4).
"""

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.service.messages import Reply
from solprop_gui.ui.bridge import ServiceBridge
from solprop_gui.views.compare import CompareView
from solprop_gui.views.live import OverviewView

__all__ = ["ResultsPage"]


class ResultsPage(QWidget):
    """Open a results file, show its Overview, and compare runs.

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
    """

    setup_opened = Signal(str, str)

    def __init__(self, bridge: ServiceBridge, parent: QWidget | None = None) -> None:
        """Make the page, with no file."""
        super().__init__(parent)
        self.bridge = bridge
        self.path: str | None = None
        self.capabilities: dict[str, object] | None = None
        self._pending: dict[int, tuple[str, str]] = {}

        self.open_button = QPushButton("Open results…")
        self.open_button.setObjectName("open-results")
        self.open_button.clicked.connect(self.choose)
        self.setup_button = QPushButton("Open setup from this run")
        self.setup_button.setObjectName("open-setup")
        self.setup_button.setToolTip(
            "Loads the recipe the file records, as resolved: the values that rebuild "
            "this run."
        )
        self.setup_button.clicked.connect(self.open_setup)
        self.compare_button = QPushButton("Add to compare")
        self.compare_button.setObjectName("add-compare")
        self.compare_button.clicked.connect(self.add_to_compare)
        self.clear_button = QPushButton("Clear compare")
        self.info = QLabel("No results file open.")
        self.info.setObjectName("results-info")
        self.info.setWordWrap(True)

        self.overview = OverviewView()
        self.compare = CompareView()
        self.clear_button.clicked.connect(self.compare.clear)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.overview, "Overview")
        self.tabs.addTab(self.compare, "Compare")

        buttons = QHBoxLayout()
        for button in (
            self.open_button,
            self.setup_button,
            self.compare_button,
            self.clear_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addLayout(buttons)
        layout.addWidget(self.info)
        layout.addWidget(self.tabs, 1)
        bridge.replied.connect(self.on_reply)
        self._update_buttons()

    def _update_buttons(self) -> None:
        """Enable what the open file allows."""
        has_file = self.path is not None
        recipe = bool(self.capabilities and self.capabilities.get("recipe"))
        self.setup_button.setEnabled(has_file and recipe)
        self.compare_button.setEnabled(has_file)

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

    def open(self, path: str | Path) -> None:
        """Open a results file.

        Parameters
        ----------
        path : str or pathlib.Path
            The file.
        """
        self.path = str(path)
        self.capabilities = None
        self.info.setText(f"{self.path}: reading…")
        self.overview.clear()
        self._update_buttons()
        self._ask("describe_output", {"path": self.path}, "describe")
        self._ask("overview", {"path": self.path}, "overview")

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
        reply : solprop_gui.service.messages.Reply
            The reply; replies to other requests are ignored.
        """
        entry = self._pending.pop(reply.request_id, None)
        if entry is None:
            return
        purpose, path = entry
        if not reply.ok:
            self.info.setText(f"{path}: {reply.error}")
            return
        value = reply.value
        if purpose == "describe" and path == self.path and isinstance(value, dict):
            self.capabilities = value
            outcome = value.get("outcome") or {}
            status = outcome.get("status") if isinstance(outcome, dict) else None
            text = (
                f"{Path(path).parent.name}/{Path(path).name}: "
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
        elif purpose == "overview" and path == self.path and isinstance(value, dict):
            self.overview.set_rows(value.get("axes", {}), value.get("rows", []))  # pyright: ignore[reportArgumentType]
        elif purpose == "compare" and isinstance(value, dict):
            name = Path(path).parent.name
            self.compare.add_run(name, value.get("axes", {}), value.get("rows", []))  # pyright: ignore[reportArgumentType]
            self.tabs.setCurrentWidget(self.compare)
        elif purpose == "setup" and isinstance(value, str):
            self.setup_opened.emit(value, path)
