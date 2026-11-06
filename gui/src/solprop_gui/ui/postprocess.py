# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Post-process step: a compressor after the run, built and searched.

The page builds a chain of :class:`~solprop_gui.model.compressor.Stage` elements --
a Taylor phase, a material (a glass, or a gas such as an air path), chirped mirrors
with optional wedges, a prism pair, a grating pair -- each fixed or searched ("auto")
by solprop, and asks the kernel to apply it to the plane the Results step shows
(:mod:`solprop_gui.service.postprocess`). The pulse before, after and at the transform
limit is drawn by :class:`~solprop_gui.views.compression.CompressionView`; the values
an auto stage found can be taken into the chain, to change by hand.
"""

from collections.abc import Mapping, Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.model.compressor import (
    FIELDS,
    KINDS,
    Field,
    Stage,
    found_values,
    new_stage,
    payload,
)
from solprop_gui.model.plot_options import parse_limit
from solprop_gui.service.messages import Reply
from solprop_gui.ui.analysis import AnalysisSession
from solprop_gui.ui.bridge import ServiceBridge
from solprop_gui.ui.pages import Section
from solprop_gui.ui.results import ResultsPage
from solprop_gui.views.compression import CompressionView

__all__ = ["PostprocessPage", "StageEditor"]

_BAD = "QLineEdit { border: 1px solid #c62828; }"
_NONE = "(none)"


class StageEditor(QWidget):
    """Edit one stage: whether solprop searches it, and its values.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    changed : Signal
        Emitted with the edited :class:`Stage`.
    """

    changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the editor, empty."""
        super().__init__(parent)
        self.stage: Stage | None = None
        self.catalogue: Mapping[str, Sequence[str]] = {}
        self.inputs: dict[str, QWidget] = {}
        self.auto = QCheckBox("Auto: solprop searches it")
        self.auto.setObjectName("stage-auto")
        self.auto.setToolTip(
            "Search this element's free values for the highest peak power, on the "
            "pulse the elements above it leave."
        )
        self.auto.toggled.connect(self._auto_toggled)
        self.form = QFormLayout()
        self.form.setContentsMargins(0, 0, 0, 0)
        self.empty = QLabel("Add an element, or pick one in the list.")
        self.empty.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.empty)
        layout.addWidget(self.auto)
        layout.addLayout(self.form)
        self.show_stage(None)

    def set_catalogue(self, catalogue: Mapping[str, Sequence[str]]) -> None:
        """Take the names the choices offer.

        Parameters
        ----------
        catalogue : Mapping
            ``glasses``, ``media`` and ``mirrors``, from the kernel.
        """
        self.catalogue = catalogue
        self.show_stage(self.stage)

    def show_stage(self, stage: Stage | None) -> None:
        """Show a stage's values, without announcing them.

        Parameters
        ----------
        stage : Stage or None
            The stage; ``None`` for none.
        """
        self.stage = stage
        while self.form.rowCount():
            self.form.removeRow(0)
        self.inputs = {}
        self.empty.setVisible(stage is None)
        self.auto.setVisible(stage is not None)
        if stage is None:
            return
        self.auto.blockSignals(True)
        self.auto.setChecked(stage.auto)
        self.auto.blockSignals(False)
        for spec in FIELDS[stage.kind]:
            if spec.shown(stage.auto):
                widget = self._input(spec, stage.values.get(spec.name))
                self.inputs[spec.name] = widget
                label = f"{spec.label} ({spec.unit})" if spec.unit else spec.label
                self.form.addRow(label, widget)

    def _input(self, spec: Field, value: object) -> QWidget:
        """Make the input for one value, showing it.

        Parameters
        ----------
        spec : Field
            The value's description.
        value : object
            Its value, in SI.

        Returns
        -------
        PySide6.QtWidgets.QWidget
            The input.
        """
        name = f"stage-{spec.name}"
        if spec.form == "switch":
            box = QCheckBox()
            box.setObjectName(name)
            box.setChecked(bool(value))
            box.toggled.connect(lambda on, n=spec.name: self._set(n, on))
            return box
        if spec.form == "count":
            spin = QSpinBox()
            spin.setObjectName(name)
            spin.setRange(1, 200)
            spin.setValue(int(value) if isinstance(value, int) else 1)
            spin.valueChanged.connect(lambda number, n=spec.name: self._set(n, number))
            return spin
        if spec.form == "choice":
            combo = QComboBox()
            combo.setObjectName(name)
            if spec.optional:
                combo.addItem(_NONE, None)
            for option in self.catalogue.get(spec.choices, []):
                combo.addItem(option, option)
            if value is not None and combo.findData(value) < 0:
                combo.addItem(str(value), value)
            combo.setCurrentIndex(max(combo.findData(value), 0))
            combo.currentIndexChanged.connect(
                lambda _i, n=spec.name, c=combo: self._set(n, c.currentData())
            )
            return combo
        line = QLineEdit()
        line.setObjectName(name)
        line.setPlaceholderText("blank" if spec.optional else "")
        if isinstance(value, int | float):
            line.setText(f"{value / spec.scale:.6g}")
        line.editingFinished.connect(lambda s=spec, edit=line: self._number(s, edit))
        return line

    def _number(self, spec: Field, line: QLineEdit) -> None:
        """Take a typed number, in the field's unit.

        Parameters
        ----------
        spec : Field
            The value's description.
        line : PySide6.QtWidgets.QLineEdit
            Its input.
        """
        try:
            typed = parse_limit(line.text())
        except ValueError:
            line.setStyleSheet(_BAD)
            return
        if typed is None and not spec.optional:
            line.setStyleSheet(_BAD)
            return
        line.setStyleSheet("")
        self._set(spec.name, None if typed is None else typed * spec.scale)

    def _set(self, name: str, value: object) -> None:
        """Change one value of the stage, and announce it.

        Parameters
        ----------
        name : str
            The value.
        value : object
            Its new value, in SI.
        """
        if self.stage is None or self.stage.values.get(name) == value:
            return
        self.stage = self.stage.with_value(name, value)
        self.changed.emit(self.stage)

    def _auto_toggled(self, on: bool) -> None:
        """Make the stage searched or fixed, and show its fields for that.

        Parameters
        ----------
        on : bool
            Whether it is searched.
        """
        if self.stage is None:
            return
        self.stage = Stage(self.stage.kind, on, dict(self.stage.values))
        self.show_stage(self.stage)
        self.changed.emit(self.stage)


class PostprocessPage(QWidget):
    """The Post-process step's panel: the compressor's elements, and its search.

    Parameters
    ----------
    bridge : solprop_gui.ui.bridge.ServiceBridge
        The service.
    results : solprop_gui.ui.results.ResultsPage
        The Results step, whose file, plane and mode selection are compressed.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    view_wanted : Signal
        Emitted when the compression is to be brought forward.
    """

    view_wanted = Signal()

    def __init__(  # noqa: PLR0915 - the page's layout
        self,
        bridge: ServiceBridge,
        results: ResultsPage | AnalysisSession,
        parent: QWidget | None = None,
    ) -> None:
        """Make the page, with an empty compressor."""
        super().__init__(parent)
        self.bridge = bridge
        self.results = results
        self.stages: list[Stage] = []
        self.found: list[Mapping[str, object]] = []
        self._pending: dict[int, str] = {}
        self.view = CompressionView()

        self.target = QLabel("")
        self.target.setObjectName("postprocess-target")
        self.target.setWordWrap(True)
        self.list = QListWidget()
        self.list.setObjectName("stages")
        self.list.currentRowChanged.connect(self._picked)
        self.add_button = QToolButton()
        self.add_button.setObjectName("add-stage")
        self.add_button.setText("Add")
        self.add_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.add_button)
        for kind, title in KINDS.items():
            menu.addAction(title, lambda k=kind: self.add(k))
        self.add_button.setMenu(menu)
        self.remove_button = QToolButton()
        self.remove_button.setText("Remove")
        self.remove_button.clicked.connect(self.remove)
        self.up_button = QToolButton()
        self.up_button.setText("Up")
        self.up_button.clicked.connect(lambda: self.shift(-1))
        self.down_button = QToolButton()
        self.down_button.setText("Down")
        self.down_button.clicked.connect(lambda: self.shift(1))
        buttons = QHBoxLayout()
        for button in (
            self.add_button,
            self.remove_button,
            self.up_button,
            self.down_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        self.editor = StageEditor()
        self.editor.changed.connect(self._edited)
        self.compress_button = QPushButton("Compress")
        self.compress_button.setObjectName("compress")
        self.compress_button.setToolTip(
            "Apply the elements in order; each auto element is searched by solprop "
            "on the pulse the elements above it leave."
        )
        self.compress_button.clicked.connect(self.compress)
        self.take_button = QPushButton("Take the values found")
        self.take_button.setObjectName("take-found")
        self.take_button.setToolTip(
            "Make each auto element fixed at what solprop found, to change by hand."
        )
        self.take_button.clicked.connect(self.take_found)
        self.status = QLabel("")
        self.status.setObjectName("postprocess-status")
        self.status.setWordWrap(True)

        chain = QWidget()
        chain_layout = QVBoxLayout(chain)
        chain_layout.setContentsMargins(0, 0, 0, 0)
        chain_layout.addWidget(self.list, 1)
        chain_layout.addLayout(buttons)
        actions = QWidget()
        actions_layout = QVBoxLayout(actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.addWidget(self.compress_button)
        actions_layout.addWidget(self.take_button)
        actions_layout.addWidget(self.status)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 0, 0)
        layout.setSpacing(2)
        self.heading = QLabel("<b>Post-process: compress the pulse</b>")
        layout.addWidget(self.heading)
        layout.addWidget(self.target)
        for title, widget, stretch in (
            ("Elements, in order", chain, 1),
            ("Element", self.editor, 0),
            ("Compress", actions, 0),
        ):
            section = Section(title)
            section.add(widget)
            layout.addWidget(section, stretch)
        bridge.replied.connect(self.on_reply)
        self._ask("optics_catalogue", {})
        self.refresh()

    # ------------------------------------------------------------------ the chain

    def add(self, kind: str) -> None:
        """Add an element of a kind at the end, and pick it.

        Parameters
        ----------
        kind : str
            One of :data:`~solprop_gui.model.compressor.KINDS`.
        """
        self.stages.append(new_stage(kind))
        self._show_list(len(self.stages) - 1)

    def remove(self) -> None:
        """Remove the element picked."""
        row = self.list.currentRow()
        if 0 <= row < len(self.stages):
            del self.stages[row]
            self._show_list(min(row, len(self.stages) - 1))

    def shift(self, step: int) -> None:
        """Move the element picked up or down the chain.

        Parameters
        ----------
        step : int
            -1 for up, 1 for down.
        """
        row = self.list.currentRow()
        target = row + step
        if 0 <= row < len(self.stages) and 0 <= target < len(self.stages):
            self.stages[row], self.stages[target] = (
                self.stages[target],
                self.stages[row],
            )
            self._show_list(target)

    def _show_list(self, row: int) -> None:
        """List the elements, and pick one.

        Parameters
        ----------
        row : int
            The element to pick; -1 for none.
        """
        self.found = []
        self.list.blockSignals(True)
        self.list.clear()
        for stage in self.stages:
            self.list.addItem(stage.describe())
        self.list.blockSignals(False)
        self.list.setCurrentRow(row)
        self._picked(row)
        self.refresh()

    def _picked(self, row: int) -> None:
        """Edit the element picked.

        Parameters
        ----------
        row : int
            Its place; -1 for none.
        """
        self.editor.show_stage(
            self.stages[row] if 0 <= row < len(self.stages) else None
        )

    def _edited(self, stage: Stage) -> None:
        """Take an edit of the element picked.

        Parameters
        ----------
        stage : Stage
            The element as edited.
        """
        row = self.list.currentRow()
        if 0 <= row < len(self.stages):
            self.stages[row] = stage
            item = self.list.item(row)
            if item is not None:
                item.setText(stage.describe())
            self.found = []
            self.refresh()

    def take_found(self) -> None:
        """Make the auto elements fixed at the values found."""
        if len(self.found) != len(self.stages):
            return
        fixed: list[Stage] = []
        for stage, report in zip(self.stages, self.found, strict=True):
            fixed.extend(found_values(stage, report))
        self.stages = fixed
        self._show_list(self.list.currentRow())

    # ------------------------------------------------------------------ the search

    def refresh(self) -> None:
        """Say what is compressed, and what can be done."""
        results = self.results
        path = results.path
        planes = results.plane.planes
        refused = (results.capabilities or {}).get("refused") or {}
        reason = refused.get("compression") if isinstance(refused, Mapping) else None
        if path is None or not planes:
            self.target.setText("Open a results file first.")
        elif reason:
            self.target.setText(f"No compression for this file: {reason}")
        else:
            mode = results.plane.modes.currentText() or "mode sum"
            z = results.plane.current_z
            where = "" if z is None else f" at z = {z * 100:.4g} cm"
            self.target.setText(
                f"The plane{where}, {mode}; the band chosen on the plane's page "
                "applies."
            )
        ready = path is not None and bool(planes) and not reason
        self.compress_button.setEnabled(ready and bool(self.stages))
        self.take_button.setEnabled(
            len(self.found) == len(self.stages)
            and any(stage.auto for stage in self.stages)
        )
        self.remove_button.setEnabled(bool(self.stages))
        self.up_button.setEnabled(len(self.stages) > 1)
        self.down_button.setEnabled(len(self.stages) > 1)

    def compress(self) -> None:
        """Ask the kernel to apply the chain to the plane, searching its auto stages."""
        results = self.results
        if results.path is None or not results.plane.planes or not self.stages:
            return
        self.status.setText("Compressing: solprop searches each auto element…")
        self.view_wanted.emit()
        self._ask(
            "compress_chain",
            {
                "path": results.path,
                "index": results.plane.index(),
                "modes": results.plane.selection(),
                "bandpass": results.options.bandpass,
                "stages": payload(self.stages),
            },
        )

    def _ask(self, kind: str, body: dict[str, object]) -> None:
        """Send a request, remembering what it is for.

        Parameters
        ----------
        kind : str
            The handler.
        body : dict
            Its arguments.
        """
        request = self.bridge.request(kind, body, key=f"postprocess-{kind}")
        self._pending[request] = kind

    def on_reply(self, reply: Reply) -> None:
        """Take a kernel reply to one of the page's requests.

        Parameters
        ----------
        reply : solprop_gui.service.messages.Reply
            The reply; replies to other requests are ignored.
        """
        kind = self._pending.pop(reply.request_id, None)
        if kind is None:
            return
        value = reply.value
        if not reply.ok or not isinstance(value, Mapping):
            self.status.setText(f"{kind}: {reply.error}")
            return
        if kind == "optics_catalogue":
            self.editor.set_catalogue(value)  # pyright: ignore[reportArgumentType]
            return
        self.found = list(value["stages"])  # pyright: ignore[reportArgumentType]
        self.view.show_compression(value, self.results.options.shown_limits("time"))
        searched = sum(1 for stage in self.stages if stage.auto)
        self.status.setText(
            f"Done: {len(self.stages)} elements, {searched} searched by solprop."
        )
        self.refresh()
