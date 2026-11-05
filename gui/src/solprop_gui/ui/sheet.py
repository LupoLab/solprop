# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Setup stage: one sheet, built from a problem class's definition (design 6.2).

The frame is the same for every class: sections of rows at the left, with one line of
help on the focused row and Advanced folded at the end; what the setup gives at the
right, with its plot and the Ready line.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from solprop.recipe.errors import RecipeError

from solprop_gui.classes.catalog import OTHER_KEY, class_of
from solprop_gui.classes.problem import (
    Derived,
    Field,
    ProblemClass,
    Target,
    derived_text,
    derived_value,
    field_changes,
    field_meta,
    field_value,
    negated_value,
    other_keywords,
    pin_entry,
    target_text,
)
from solprop_gui.model.display import size_text
from solprop_gui.model.quantities import number_in, write
from solprop_gui.model.recipe import MISSING, RecipeModel
from solprop_gui.registry.explore import PIN_SEARCH
from solprop_gui.registry.pages import GAS_TITLES, label
from solprop_gui.ui.fields import ParameterRow
from solprop_gui.ui.pages import Section, issue_keyword
from solprop_gui.ui.quantity import QuantityField
from solprop_gui.ui.run_options import RunOptions
from solprop_gui.ui.theme import clear_layout, named_label, repolish
from solprop_gui.views.cross_section import CrossSection
from solprop_gui.views.page_plots import ORANGE, LinePlot, PlotMarks

__all__ = ["LensEditor", "SheetStage", "TargetEditor"]

_NARROW = 1000
"""Below this width, in pixels, the plot moves below the form."""


class TargetEditor(QFrame):
    """The strip that asks what a derived number should be, and what moves to give it.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    accepted : Signal
        Emitted with the keyword that moves and the ``[pins]`` entry.
    released : Signal
        Emitted with the keyword whose target is to go.
    root_chosen : Signal
        Emitted with the keyword and which root it should take.
    """

    accepted = Signal(str, object)
    released = Signal(str)
    root_chosen = Signal(str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the strip, hidden."""
        super().__init__(parent)
        self.setObjectName("targetEditor")
        self.target: Target | None = None
        self.pinned: str | None = None
        self.title = QLabel("")
        self.value = QLineEdit()
        self.value.setObjectName("target-value")
        self.value.setFixedWidth(90)
        self.value.returnPressed.connect(self.accept)
        self.unit = named_label("", "note")
        self.moves_label = named_label("moves", "note")
        self.moves = QComboBox()
        self.moves.setObjectName("target-moves")
        self.problem = named_label("", "note")
        self.set_button = QPushButton("Set target")
        self.set_button.setObjectName("plain")
        self.set_button.clicked.connect(self.accept)
        self.release_button = QPushButton("Release")
        self.release_button.setObjectName("plain")
        self.release_button.clicked.connect(self._release)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("plain")
        cancel.clicked.connect(self.hide)
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 7, 12, 7)
        row.setSpacing(8)
        for widget in (
            self.title,
            self.value,
            self.unit,
            self.moves_label,
            self.moves,
            self.problem,
        ):
            row.addWidget(widget)
        row.addStretch(1)
        for widget in (self.set_button, self.release_button, cancel):
            row.addWidget(widget)
        self.hide()

    def ask(
        self,
        number: Derived,
        shown: str,
        moves: Sequence[tuple[str, str]],
        pinned: str | None,
    ) -> None:
        """Open on a number.

        Parameters
        ----------
        number : Derived
            The number, which has a target.
        shown : str
            What it reads now, without its unit.
        moves : Sequence of tuple
            What can move: each keyword and its label.
        pinned : str or None
            The keyword this number's target holds now.
        """
        assert number.target is not None  # only such a number opens the strip
        self.target = number.target
        self.pinned = pinned
        self.title.setText(f"Set {number.label.lower()} to")
        self.value.setText(shown)
        self.unit.setText(number.target.unit or "")
        self.problem.clear()
        self.moves.clear()
        for name, title in moves:
            self.moves.addItem(title, name)
        if pinned is not None:
            self.moves.setCurrentIndex(max(self.moves.findData(pinned), 0))
        self.moves.setVisible(self.moves.count() > 1)
        self.moves_label.setText(
            "moves" if self.moves.count() > 1 else f"moves {moves[0][1].lower()}"
        )
        self.release_button.setVisible(pinned is not None)
        self.show()
        self.value.setFocus()
        self.value.selectAll()

    def accept(self) -> None:
        """Announce the target typed, or say why it does not read."""
        if self.target is None:
            return
        name = str(self.moves.currentData())
        search = PIN_SEARCH.get(name)
        try:
            entry = pin_entry(self.target, self.value.text(), search)
        except ValueError:
            self.problem.setText("Type a number.")
            return
        self.hide()
        self.accepted.emit(name, entry)

    def _release(self) -> None:
        """Announce that the target goes."""
        if self.pinned is not None:
            self.released.emit(self.pinned)
        self.hide()


class LensEditor(QFrame):
    """The strip that gives a beam's waist and focus from a lens.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    accepted : Signal
        Emitted with the focal length and the radius on the lens, m.
    """

    accepted = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the strip, hidden."""
        super().__init__(parent)
        self.setObjectName("targetEditor")
        self.focal = QuantityField("m", "cm")
        self.focal.setObjectName("lens-focal")
        self.radius = QuantityField("m", "mm")
        self.radius.setObjectName("lens-radius")
        self.problem = named_label("", "note")
        use = QPushButton("Use this lens")
        use.setObjectName("plain")
        use.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("plain")
        cancel.clicked.connect(self.hide)
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 7, 12, 7)
        row.setSpacing(8)
        for widget in (
            QLabel("Focal length"),
            self.focal,
            QLabel("Beam radius on it"),
            self.radius,
            self.problem,
        ):
            row.addWidget(widget)
        row.addStretch(1)
        row.addWidget(use)
        row.addWidget(cancel)
        self.hide()

    def accept(self) -> None:
        """Announce the lens, or say why it does not read."""
        try:
            focal, radius = self.focal.read(), self.radius.read()
            values = [float(number_in(v, "m", "m")) for v in (focal, radius)]
        except RecipeError, TypeError, ValueError:
            self.problem.setText("Give both as lengths.")
            return
        if min(values) <= 0:
            self.problem.setText("Both must be positive.")
            return
        self.problem.clear()
        self.hide()
        self.accepted.emit(values[0], values[1])


class SheetStage(QWidget):
    """A sheet.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    edited : Signal
        Emitted with (path, table, value) when a row changes.
    example_chosen : Signal
        Emitted with the name of a solprop example to start from.
    pins_changed : Signal
        Emitted with the keywords freed, each with its plain value, and any keyword
        and ``[pins]`` entry that now sets it.
    values_edited : Signal
        Emitted with the paths a row moved, each with its value, where it moves several.
    lens_wanted : Signal
        Emitted with a lens's focal length and the radius on it, m.
    scan_wanted : Signal
        Emitted with a keyword and its value when a row's menu asks to scan it.
    rows : dict
        The rows, by path and table.
    derived : dict
        The labels of the derived numbers, by key.
    """

    edited = Signal(str, str, object)
    example_chosen = Signal(str)
    pins_changed = Signal(object, object)
    values_edited = Signal(object)
    lens_wanted = Signal(float, float)
    scan_wanted = Signal(str, object)

    def __init__(self, parent: QWidget | None = None) -> None:  # noqa: PLR0915 - a layout
        """Make the empty frame."""
        super().__init__(parent)
        self.setObjectName("page")
        self.problem: ProblemClass | None = None
        self.rows: dict[tuple[str, str], ParameterRow] = {}
        self.derived: dict[str, QLabel | QPushButton] = {}
        self._advanced_rows: list[ParameterRow] = []
        self._advanced_waiting: list[Mapping[str, object]] = []
        self._advanced_catalog: object = None
        self._issues: list[Mapping[str, object]] = []
        self._examples: list[Mapping[str, object]] = []
        self._title: str | None = None
        self._lists: dict[str, Sequence[str]] = {}
        self._model: RecipeModel | None = None
        self._solved: dict[str, float] = {}
        self._notes: dict[str, QWidget] = {}
        self._shown_numbers: dict[str, float | None] = {}
        self._size: str | None = None
        self._verdict = ""
        self._fields: dict[str, Field] = {}
        self._switches: list[tuple[Field, QCheckBox]] = []
        self._example: Mapping[str, object] = {}

        self.examples = QComboBox()
        self.examples.setObjectName("examples")
        self.examples.activated.connect(self._example_picked)
        self.sections = QVBoxLayout()
        self.sections.setSpacing(10)
        self.help = named_label("", "help")
        self.help.setWordWrap(True)
        self.help.setMinimumHeight(32)
        self.help.setAlignment(Qt.AlignmentFlag.AlignTop)

        body = QWidget()
        body.setObjectName("page")
        column = QVBoxLayout(body)
        column.setContentsMargins(20, 14, 16, 16)
        column.setSpacing(10)
        column.addWidget(named_label("Start from", "h3"))
        column.addWidget(self.examples)
        column.addLayout(self.sections)
        column.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(body)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.form = QFrame()
        self.form.setObjectName("form")
        self.form.setMinimumWidth(470)
        form = QVBoxLayout(self.form)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(0)
        form.addWidget(scroll, 1)
        foot = QVBoxLayout()
        foot.setContentsMargins(20, 6, 16, 8)
        foot.addWidget(self.help)
        form.addLayout(foot)

        self.numbers = QGridLayout()
        self.numbers.setHorizontalSpacing(24)
        self.numbers.setVerticalSpacing(8)
        self.target_editor = TargetEditor()
        self.target_editor.accepted.connect(self._target_set)
        self.target_editor.released.connect(self._release)
        self.lens_editor = LensEditor()
        self.lens_editor.accepted.connect(self.lens_wanted)
        self.preview_title = named_label("", "h3")
        self.curve = LinePlot(xlabel="", ylabel="")
        self._marks = PlotMarks(self.curve)
        self.marks = self._marks.lines
        self.curve.hide()
        self.zero = pg.InfiniteLine(angle=0, pos=0, pen=pg.mkPen("#d7dce2"))
        self.curve.addItem(self.zero, ignoreBounds=True)  # pyright: ignore[reportCallIssue]
        self.preview = QFrame()
        self.preview.setObjectName("preview")
        preview = QVBoxLayout(self.preview)
        self.preview_note = named_label("", "note")
        self.preview_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.addWidget(self.preview_note)
        self.ready = named_label("Checking…", "ready")
        self.ready.setWordWrap(True)
        self.gives = QWidget()
        self.gives.setObjectName("page")
        right = QVBoxLayout(self.gives)
        right.setContentsMargins(20, 14, 20, 14)
        right.setSpacing(12)
        right.addWidget(named_label("What this gives", "h3"))
        right.addLayout(self.numbers)
        right.addWidget(self.target_editor)
        right.addWidget(self.lens_editor)
        right.addWidget(self.preview_title)
        right.addWidget(self.curve, 2)
        right.addWidget(self.preview, 1)
        # Where a class has no preview, its numbers stay at the top.
        self.filler = QWidget()
        self.filler.setObjectName("page")
        right.addWidget(self.filler, 1)
        self.loss_plot = LinePlot(
            xlabel="Wavelength (nm)", ylabel="Loss (dB/m)", unit="dB/m"
        )
        self.loss_plot.set_log(True)
        self.loss_mark = pg.InfiniteLine(
            angle=90, pen=pg.mkPen(ORANGE, style=Qt.PenStyle.DashLine)
        )
        self.loss_plot.addItem(self.loss_mark, ignoreBounds=True)  # pyright: ignore[reportCallIssue]
        self.cross_section = CrossSection()
        self.below = QWidget()
        self.below.setObjectName("page")
        below = QHBoxLayout(self.below)
        below.setContentsMargins(0, 0, 0, 0)
        below.setSpacing(12)
        below.addWidget(self.loss_plot, 3)
        below.addWidget(self.cross_section, 1)
        self.below.hide()
        right.addWidget(self.below, 1)
        right.addWidget(self.ready)
        self.run_options = RunOptions()
        right.addWidget(self.run_options)

        self.split = QBoxLayout(QBoxLayout.Direction.LeftToRight, self)
        self.split.setContentsMargins(0, 0, 0, 0)
        self.split.setSpacing(0)
        self.split.addWidget(self.form, 2)
        self.split.addWidget(self.gives, 3)
        app = QApplication.instance()
        if isinstance(app, QApplication):
            app.focusChanged.connect(self._focus_moved)

    def event(self, event: QEvent) -> bool:
        """Lay the plot below the form on a narrow window.

        Parameters
        ----------
        event : PySide6.QtCore.QEvent
            The event.

        Returns
        -------
        bool
            Whether it was handled.
        """
        if event.type() == QEvent.Type.Resize:
            narrow = self.width() < _NARROW
            self.split.setDirection(
                QBoxLayout.Direction.TopToBottom
                if narrow
                else QBoxLayout.Direction.LeftToRight
            )
        return super().event(event)

    def _focus_moved(self, _old: QObject | None, new: QObject | None) -> None:
        """Show the help of the row that took the focus.

        Parameters
        ----------
        _old : PySide6.QtCore.QObject or None
            What had the focus.
        new : PySide6.QtCore.QObject or None
            What has it.
        """
        while new is not None and not isinstance(new, ParameterRow):
            new = new.parent()
        if new is not None and new.name in {name for name, _ in self.rows}:
            self.help.setText(str(new.meta.get("summary") or ""))

    # ------------------------------------------------------------------ building

    def build(
        self,
        problem: ProblemClass,
        description: Mapping[str, object],
        build: Mapping[str, object],
        title: str | None = None,
    ) -> None:
        """Make the sheet of a class.

        Parameters
        ----------
        problem : ProblemClass
            The class.
        description : Mapping
            The kernel's description of its route.
        build : Mapping
            The open recipe's ``[build]`` table, which says what its objects are.
        title : str or None, optional
            The open recipe's title, which says which example it is.
        """
        self.problem = problem
        self._title = title
        self.rows.clear()
        self._notes.clear()
        self._fields.clear()
        self._switches = []
        self.lens_editor.hide()
        self._solved = {}
        self._shown_numbers = {}
        self._size = None
        self.target_editor.hide()
        self._advanced_rows = []
        clear_layout(self.sections)
        catalog = description.get("types")
        for part in problem.sections:
            section = Section(part.title)
            section.header.setChecked(not part.folded)
            shown = False
            for row in part.fields:
                if row.member is not None:
                    section.add(self._switch(row))
                    shown = True
                    continue
                meta = field_meta(row.path, description, build)
                if meta is None:
                    continue
                self._fields[row.path] = row
                widget = self._row(self._entry(row, meta), catalog, on_sheet=True)
                if row.blank is not None:
                    for box in widget.findChildren(QLineEdit):
                        if not box.placeholderText():
                            box.setPlaceholderText(row.blank)
                section.add(widget)
                note = self._solved_note(row.path)
                self._notes[row.path] = note
                section.add(note)
                shown = True
            if problem.lens is not None and problem.lens[0] in self._section_paths(
                part
            ):
                section.add(self._lens_link())
            section.setVisible(shown)
            self.sections.addWidget(section)
        self._add_advanced(problem, description, catalog)
        self._show_numbers(problem)
        self._show_preview(problem)
        self.help.clear()
        self.ready.setText("Checking…")
        self.ready.setProperty("error", False)
        self.ready.setProperty("warn", False)
        repolish(self.ready)
        self._fill_examples()

    def _show_preview(self, problem: ProblemClass) -> None:
        """Show the class's preview: its curve, a note of one to come, or nothing.

        Parameters
        ----------
        problem : ProblemClass
            The class.
        """
        self.preview_title.setText(problem.preview)
        self.preview_note.setText("This plot comes with the class's live sheet.")
        self.below.hide()
        self.curve.setVisible(problem.curve is not None)
        self.preview.setVisible(problem.curve is None and bool(problem.preview))
        self.preview_title.setVisible(bool(problem.preview))
        self.filler.setVisible(problem.curve is None and not problem.preview)

    @staticmethod
    def _section_paths(part: object) -> set[str]:
        """Return the paths a section of a sheet shows.

        Parameters
        ----------
        part : object
            The section.

        Returns
        -------
        set of str
            Its rows' paths.
        """
        fields: Sequence[Field] = part.fields  # pyright: ignore[reportAttributeAccessIssue]
        return {row.path for row in fields}

    def _lens_link(self) -> QWidget:
        """Make the link that gives the beam from a lens.

        Returns
        -------
        PySide6.QtWidgets.QWidget
            The link.
        """
        holder = QWidget()
        holder.setObjectName("page")
        row = QHBoxLayout(holder)
        row.setContentsMargins(116, 0, 0, 2)
        link = QPushButton("From a lens…")
        link.setObjectName("link")
        link.setCursor(Qt.CursorShape.PointingHandCursor)
        link.clicked.connect(self.lens_editor.show)
        row.addWidget(link)
        row.addStretch(1)
        return holder

    def _switch(self, row: Field) -> QWidget:
        """Make a row that says whether a list holds an entry.

        Parameters
        ----------
        row : Field
            The sheet's row.

        Returns
        -------
        PySide6.QtWidgets.QWidget
            The row.
        """
        box = QCheckBox(row.label or label(row.path))
        box.setObjectName(f"switch-{row.member}")
        box.clicked.connect(lambda on, r=row: self._move(r, on))
        self._switches.append((row, box))
        return box

    def _move(self, row: Field, value: object) -> None:
        """Announce what a row of the sheet moves.

        Parameters
        ----------
        row : Field
            The row.
        value : object
            What it now shows.
        """
        if self._model is None or self.problem is None:
            return
        build = self._model.document.get("build", {})
        changes = field_changes(self.problem, row, value, build, self._example)
        self.values_edited.emit(changes)

    def set_example(self, build: Mapping[str, object]) -> None:
        """Take the ``[build]`` table of the class's example, where switches read from.

        Parameters
        ----------
        build : Mapping
            The table.
        """
        self._example = build

    def apply_lens(self, waist: float, distance: float) -> None:
        """Set the beam a lens gives.

        Parameters
        ----------
        waist : float
            The waist, m.
        distance : float
            Its distance from the launch, m.
        """
        if self.problem is None or self.problem.lens is None:
            return
        waist_path, launch_path, _wavelength = self.problem.lens
        launch = self._literal(launch_path, distance)
        self.values_edited.emit(
            [
                (waist_path, self._literal(waist_path, waist)),
                (launch_path, negated_value(launch)),
            ]
        )

    def _row_edited(self, path: str, table: str, value: object) -> None:
        """Take an edit of one of the sheet's own rows.

        Parameters
        ----------
        path : str
            Its path.
        table : str
            Its table.
        value : object
            The new value.
        """
        row = self._fields.get(path)
        if row is None or table != "build":
            self.edited.emit(path, table, value)
            return
        self._move(row, value)

    def _entry(self, row: Field, meta: Mapping[str, object]) -> dict[str, object]:
        """Return the metadata of a sheet's row: its path, its label, what it offers.

        Parameters
        ----------
        row : Field
            The sheet's row.
        meta : Mapping
            Its keyword's metadata.

        Returns
        -------
        dict
            What :class:`~solprop_gui.ui.fields.ParameterRow` reads.
        """
        name = row.path.rsplit(".", 1)[-1]
        entry: dict[str, object] = {
            **meta,
            "name": row.path,
            "label": row.label or label(name),
        }
        if row.planes_over is not None:
            entry["forms"] = [{"kind": "integer", "minimum": 1}]
            entry["default_value"] = None
            entry["default"] = None
        listed = self._lists.get(row.choices or "")
        if listed:
            forms: Sequence[Mapping[str, object]] = meta.get("forms", ())  # pyright: ignore[reportAssignmentType]
            entry["forms"] = [{"kind": "choice", "choices": list(listed)}, *forms]
            entry["choice_titles"] = GAS_TITLES
        return entry

    def _solved_note(self, path: str) -> QWidget:
        """Make the line under a row that says a target sets it.

        Parameters
        ----------
        path : str
            The row's path.

        Returns
        -------
        PySide6.QtWidgets.QWidget
            The line, hidden.
        """
        note = QWidget()
        note.setObjectName("page")
        row = QHBoxLayout(note)
        row.setContentsMargins(116, 0, 0, 2)
        row.setSpacing(6)
        row.addWidget(named_label("", "solved"))
        link = QPushButton("release")
        link.setObjectName("link")
        link.setCursor(Qt.CursorShape.PointingHandCursor)
        link.clicked.connect(lambda _on=False, p=path: self._release(p))
        row.addWidget(link)
        row.addStretch(1)
        note.hide()
        return note

    def _row(
        self, meta: Mapping[str, object], catalog: object, *, on_sheet: bool = False
    ) -> ParameterRow:
        """Make one row.

        Parameters
        ----------
        meta : Mapping
            Its keyword's metadata.
        catalog : object
            The registered types the route can name.
        on_sheet : bool, optional
            Whether it is one of the sheet's own, rather than one under Advanced.

        Returns
        -------
        solprop_gui.ui.fields.ParameterRow
            The row, connected and held.
        """
        row = ParameterRow(
            meta,
            "build",
            catalog=catalog,  # pyright: ignore[reportArgumentType]
            one_line=on_sheet,
            on_sheet=on_sheet,
        )
        row.edited.connect(self._row_edited if on_sheet else self.edited)
        row.scan_wanted.connect(self.scan_wanted)
        self.rows[row.name, "build"] = row
        return row

    def _add_advanced(
        self, problem: ProblemClass, description: Mapping[str, object], catalog: object
    ) -> None:
        """Add the folded section that holds every keyword the sheet does not.

        Parameters
        ----------
        problem : ProblemClass
            The class.
        description : Mapping
            The kernel's description of its route.
        catalog : object
            The registered types the route can name.
        """
        section = Section("Advanced")
        section.header.setChecked(False)
        find = QLineEdit()
        find.setObjectName("find-setting")
        find.setPlaceholderText("Find any setting")
        find.textChanged.connect(self._filter_advanced)
        section.add(find)
        # Made as the section is first opened or searched: dozens of rows, some with
        # a form per type, made the sheet slow to open.
        self._advanced_waiting = [
            meta
            for meta in other_keywords(problem, description)
            if any(form.get("recipe", True) for form in _forms(meta))
        ]
        self._advanced_catalog = catalog
        section.header.toggled.connect(lambda shown: shown and self.fill_advanced())
        self.advanced = section
        self.sections.addWidget(section)

    def fill_advanced(self) -> None:
        """Make the Advanced rows, once, and show the recipe's values in them."""
        if not self._advanced_waiting:
            return
        waiting, self._advanced_waiting = self._advanced_waiting, []
        for meta in waiting:
            row = self._row(meta, self._advanced_catalog)
            self._advanced_rows.append(row)
            self.advanced.add(row)
        if self._model is not None:
            self.show_values(self._model)
            self.show_issues(self._issues)

    def _filter_advanced(self, text: str) -> None:
        """Show the advanced rows that match what was typed.

        Parameters
        ----------
        text : str
            What to find, in a row's name, label or summary.
        """
        self.fill_advanced()
        wanted = text.strip().lower()
        for row in self._advanced_rows:
            found = f"{row.name} {row.title} {row.meta.get('summary', '')}".lower()
            row.setVisible(wanted in found)

    def _show_numbers(self, problem: ProblemClass) -> None:
        """Lay out the class's derived numbers, unknown until the kernel replies.

        Parameters
        ----------
        problem : ProblemClass
            The class.
        """
        clear_layout(self.numbers)
        self.derived = {}
        for index, number in enumerate(problem.derived):
            tile = QVBoxLayout()
            tile.setSpacing(0)
            tile.addWidget(named_label(number.label, "derivedLabel"))
            value = self._number_widget(number)
            tile.addWidget(value)
            self.derived[number.key] = value
            self.numbers.addLayout(tile, index // 4, index % 4)

    def _number_widget(self, number: Derived) -> QLabel | QPushButton:
        """Make what shows a derived number: a button where a target can be typed.

        Parameters
        ----------
        number : Derived
            The number.

        Returns
        -------
        PySide6.QtWidgets.QLabel or PySide6.QtWidgets.QPushButton
            The widget.
        """
        if number.target is None:
            return named_label("…", "derivedValue")
        button = QPushButton("…")
        button.setObjectName("derivedTarget")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        button.setToolTip(
            "Click to type the value you want; an input moves to give it."
        )
        button.clicked.connect(lambda _on=False, n=number: self.open_target(n))
        return button

    def _pin_of(self, target: Target) -> tuple[str, Mapping[str, object]] | None:
        """Return the keyword a target sets now, and its ``[pins]`` entry.

        Parameters
        ----------
        target : Target
            The target.

        Returns
        -------
        tuple or None
            The keyword and the entry; ``None`` when it sets nothing.
        """
        if self._model is None:
            return None
        for name in target.moves:
            entry = self._model.get(name, "pins")
            if isinstance(entry, Mapping) and entry.get("target") == target.kind:
                return name, entry
        return None

    def open_target(self, number: Derived) -> None:
        """Ask what a derived number should be.

        Parameters
        ----------
        number : Derived
            The number.
        """
        if self._model is None or number.target is None:
            return
        target = number.target
        held = self._pin_of(target)
        # An energy and a peak power cannot both be set: the one in use moves.
        moves = [
            (name, self._move_title(name))
            for name in target.moves
            if name in ("pressure", "radius")
            or self._model.get(name) is not MISSING
            or self._model.get(name, "pins") is not MISSING
        ]
        if held is not None:
            shown = target_text(target, held[1]).split(" ")[0]
        else:
            value = self._shown_numbers.get(number.key)
            shown = "" if value is None else _plain_number(number, value)
        self.target_editor.ask(number, shown, moves, None if held is None else held[0])

    def _move_title(self, name: str) -> str:
        """Return what a moving keyword is called.

        Parameters
        ----------
        name : str
            The keyword.

        Returns
        -------
        str
            Its row's label.
        """
        row = self.rows.get((name, "build"))
        return row.title if row is not None else label(name)

    def _literal(self, name: str, value: float) -> object:
        """Return a solved value as the plain value a row holds, in the row's unit.

        Parameters
        ----------
        name : str
            The keyword.
        value : float
            The value, in the keyword's unit.

        Returns
        -------
        object
            A quantity string, or the number where the keyword has no unit.
        """
        row = self.rows.get((name, "build"))
        field = None if row is None else getattr(row, "number", None)
        keyword_unit = None if field is None else field.keyword_unit
        if field is None or keyword_unit is None:
            return value
        unit = field.unit() or keyword_unit
        text = f"{float(number_in(float(f'{value:.6g}'), keyword_unit, unit)):.6g}"
        return write(text, keyword_unit, unit)

    def _guide_pins(self, moving: str) -> list[tuple[str, object]]:
        """Return the guide's other pins, which a new one replaces, with their values.

        solprop allows one pin on the guide or fill, so typing a second target frees
        the first at the value it solved.

        Parameters
        ----------
        moving : str
            The keyword that is about to be set by a target.

        Returns
        -------
        list of tuple
            Each keyword to free, and the plain value it keeps.
        """
        if self._model is None or moving not in ("pressure", "radius"):
            return []
        freed: list[tuple[str, object]] = []
        for name in ("pressure", "radius"):
            if name != moving and self._model.get(name, "pins") is not MISSING:
                freed.append((name, self._kept_value(name)))
        return freed

    def _kept_value(self, name: str) -> object:
        """Return the value a keyword keeps as its target goes.

        Parameters
        ----------
        name : str
            The keyword.

        Returns
        -------
        object
            The value last solved; with none yet, the default's unset value.
        """
        solved = self._solved.get(name)
        return MISSING if solved is None else self._literal(name, solved)

    def _target_set(self, name: str, entry: Mapping[str, object]) -> None:
        """Announce a typed target.

        Parameters
        ----------
        name : str
            The keyword that moves.
        entry : Mapping
            The ``[pins]`` entry.
        """
        target = self.target_editor.target
        freed = list(self._guide_pins(name))
        if target is not None:
            held = self._pin_of(target)
            if held is not None and held[0] != name:
                freed.append((held[0], self._kept_value(held[0])))
        self.pins_changed.emit(freed, (name, entry))

    def _release(self, name: str) -> None:
        """Turn a target back into the plain value it solved.

        Parameters
        ----------
        name : str
            The keyword.
        """
        self.pins_changed.emit([(name, self._kept_value(name))], None)

    def _fill_examples(self) -> None:
        """Offer the examples of the sheet's class."""
        self.examples.clear()
        if self.problem is None:
            return
        for entry in self._examples:
            build: Mapping[str, object] = entry["build"]  # pyright: ignore[reportAssignmentType]
            found = class_of(str(entry["route"]), build)
            other = found is None and self.problem.key == OTHER_KEY
            if found is self.problem or (
                other and entry["route"] == self.problem.route
            ):
                self.examples.addItem(str(entry["title"]), str(entry["name"]))
        # A recipe from elsewhere is no example, and the list then shows none.
        self.examples.setCurrentIndex(self.examples.findText(self._title or ""))

    def set_examples(self, examples: Sequence[Mapping[str, object]]) -> None:
        """Take solprop's examples, of which a sheet offers its class's.

        Parameters
        ----------
        examples : Sequence of Mapping
            The kernel's ``examples`` reply.
        """
        self._examples = list(examples)
        self._fill_examples()

    def _example_picked(self, index: int) -> None:
        """Start from the example picked.

        Parameters
        ----------
        index : int
            Its place in the list.
        """
        self.example_chosen.emit(str(self.examples.itemData(index)))

    # ------------------------------------------------------------------ showing

    def show_values(self, model: RecipeModel) -> None:
        """Show the recipe's values in the rows.

        Parameters
        ----------
        model : solprop_gui.model.recipe.RecipeModel
            The recipe.
        """
        self._model = model
        build = model.document.get("build", {})
        for (path, table), row in self.rows.items():
            pin = model.get(path, "pins")
            if isinstance(pin, Mapping) and row.on_sheet:
                self._show_solved(path, row, pin)
                continue
            if isinstance(pin, Mapping) and row.pin_targets:
                row.set_pin(pin)  # pyright: ignore[reportArgumentType]
                continue
            row.stack.setEnabled(True)
            if path in self._notes:
                self._notes[path].hide()
            field = self._fields.get(path)
            value = (
                model.get(path, table) if field is None else field_value(field, build)
            )
            # What was typed is already there, as it was typed.
            if row.value != value:
                row.set_value(value)
        for field, box in self._switches:
            box.setChecked(bool(field_value(field, build)))
        self._mark_targets()

    def _show_solved(
        self, path: str, row: ParameterRow, pin: Mapping[str, object]
    ) -> None:
        """Show a row a target sets: its solved value, and what it was solved for.

        Parameters
        ----------
        path : str
            The row's path.
        row : solprop_gui.ui.fields.ParameterRow
            The row.
        pin : Mapping
            Its ``[pins]`` entry.
        """
        solved = self._solved.get(path)
        row.set_value(MISSING if solved is None else self._literal(path, solved))
        row.stack.setEnabled(False)
        note = self._notes.get(path)
        if note is None or self.problem is None:
            return
        text = note.findChild(QLabel, "solved")
        for number in self.problem.derived:
            target = number.target
            if target is not None and target.kind == pin.get("target"):
                if isinstance(text, QLabel):
                    what = number.label.lower()
                    text.setText(f"solved for {what} {target_text(target, pin)}")
                note.show()

    def _mark_targets(self) -> None:
        """Mark the numbers a target holds."""
        if self.problem is None:
            return
        for number in self.problem.derived:
            widget = self.derived.get(number.key)
            if number.target is not None and widget is not None:
                widget.setProperty("pinned", self._pin_of(number.target) is not None)
                repolish(widget)

    def show_issues(self, issues: Sequence[Mapping[str, object]]) -> None:
        """Show solprop's issues beside their rows, and its verdict on the Ready line.

        Parameters
        ----------
        issues : Sequence of Mapping
            The issues: ``message``, ``parameter`` and ``severity``.
        """
        self._issues = list(issues)
        for issue in issues:
            where = issue.get("parameter")
            named = where if isinstance(where, list | tuple) else [where]
            # An issue on an Advanced keyword needs its row made to stand beside it.
            if (
                any(path and (path, "build") not in self.rows for path in named)
                and issue.get("severity") == "error"
            ):
                self.fill_advanced()
        for row in self.rows.values():
            row.set_issue(None)
        for issue in issues:
            where = issue.get("parameter")
            paths = where if isinstance(where, list | tuple) else [where]
            for path in paths:
                keyword = issue_keyword(path)[0] if path else None
                for (name, _), row in self.rows.items():
                    if name in (path, keyword):
                        row.set_issue(
                            str(issue.get("message")), str(issue.get("severity"))
                        )
        errors = [i for i in issues if i.get("severity") == "error"]
        warnings = [i for i in issues if i.get("severity") != "error"]
        if errors:
            text = f"Not ready: {errors[0].get('message')}"
        elif warnings:
            text = f"Ready, with a warning: {warnings[0].get('message')}"
        else:
            text = "Ready to run"
        self._verdict = text
        if errors:
            self._size = None
        self._show_ready()
        self.ready.setProperty("error", bool(errors))
        self.ready.setProperty("warn", bool(warnings) and not errors)
        repolish(self.ready)

    def _show_ready(self) -> None:
        """Write the Ready line: solprop's verdict, and the run's size where known."""
        parts = [self._verdict]
        if self._size is not None:
            parts.append(self._size)
        self.ready.setText("  ·  ".join(parts))

    def show_size(self, size: float | None) -> None:
        """Show the size of the output the run will write.

        Parameters
        ----------
        size : float or None
            Bytes; ``None`` when it is not known.
        """
        self._size = None if size is None else f"{size_text(size)} of output"
        self._show_ready()

    def set_list(self, name: str, values: Sequence[str]) -> None:
        """Take what a kernel request lists, for the rows that offer it.

        Parameters
        ----------
        name : str
            The request.
        values : Sequence of str
            What it lists.
        """
        self._lists[name] = list(values)

    def show_error(self, message: str) -> None:
        """Say that the numbers could not be worked out, as when a target has no root.

        Parameters
        ----------
        message : str
            The reason solprop gave, with the attainable range.
        """
        self.show_summary(None, message)
        self._size = None
        self._verdict = f"Not ready: {message}"
        self.ready.setProperty("error", True)
        repolish(self.ready)
        self._show_ready()

    def show_curve(self, curve: Mapping[str, object]) -> None:
        """Show the preview's curve and marks, and the numbers that come with them.

        Parameters
        ----------
        curve : Mapping
            The kernel's reply: its ``plot`` and ``numbers``.
        """
        extra = curve.get("numbers")
        if isinstance(extra, Mapping):
            self._show_extra(extra)
        plot = curve.get("plot")
        if not isinstance(plot, Mapping):
            self.curve.set_data(np.zeros(0), [np.zeros(0)])
            return
        x = np.asarray(plot["x"], dtype=float)
        self.curve.set_xlabel(str(plot["xlabel"]))
        self.curve.set_ylabel(str(plot["ylabel"]))
        self.curve.unit = str(plot["unit"])
        self.curve.set_data(x, [np.asarray(plot["y"], dtype=float)])
        low, high = plot["limit"]
        self.curve.set_view((float(x[0]), float(x[-1])), (float(low), float(high)))
        self.zero.setVisible(bool(plot.get("zero")))
        self._show_marks(plot.get("marks") or ())  # pyright: ignore[reportArgumentType]
        self._show_below(curve.get("loss"), curve.get("shapes"))
        self._mark_targets()

    def _show_below(self, loss: object, shapes: object) -> None:
        """Show a guide's loss and cross-section, where its class has them.

        Parameters
        ----------
        loss : object
            The reply's ``loss``, or ``None``.
        shapes : object
            The reply's ``shapes``, or ``None``.
        """
        drawn = isinstance(shapes, list) and bool(shapes)
        self.below.setVisible(isinstance(loss, Mapping) or drawn)
        self.loss_plot.setVisible(isinstance(loss, Mapping))
        self.cross_section.set_shapes(shapes if isinstance(shapes, list) else [])
        self.cross_section.setVisible(drawn)
        if not isinstance(loss, Mapping):
            return
        x = np.asarray(loss["x"], dtype=float)
        self.loss_plot.set_data(x, [np.asarray(loss["y"], dtype=float)])
        self.loss_plot.fit()
        self.loss_mark.setPos(float(loss["pump"]))

    def _show_marks(self, marks: Sequence[Sequence[object]]) -> None:
        """Mark positions on the preview.

        Parameters
        ----------
        marks : Sequence of Sequence
            Each mark's label and position, ``None`` where it has none.
        """
        self._marks.show(marks)

    def _show_extra(self, values: Mapping[str, object]) -> None:
        """Show the numbers that arrive with the curve.

        Parameters
        ----------
        values : Mapping
            By key, each value or ``None``.
        """
        if self.problem is None:
            return
        for number in self.problem.derived:
            if number.key in values:
                value = derived_value(values, number.key)
                self._shown_numbers[number.key] = value
                self.derived[number.key].setText(derived_text(number, value))

    def show_summary(
        self, summary: Mapping[str, object] | None, error: object = None
    ) -> None:
        """Show the numbers solprop worked out.

        Parameters
        ----------
        summary : Mapping or None
            The kernel's summary; ``None`` when it failed.
        error : object, optional
            Why it failed.
        """
        if self.problem is None:
            return
        known = () if self.problem.curve is None else ("rdw", "loss")
        for number in self.problem.derived:
            value = derived_value(summary, number.key)
            widget = self.derived[number.key]
            if summary is not None and number.key in known:
                continue  # these come with the curve
            self._shown_numbers[number.key] = value
            widget.setText(derived_text(number, value))
            if number.target is None:
                widget.setToolTip("" if summary is not None else str(error or ""))
        pinned = None if summary is None else summary.get("pinned")
        if isinstance(pinned, Mapping):
            self._solved = {str(k): float(v) for k, v in pinned.items()}
            if self._model is not None:
                self.show_values(self._model)


def _plain_number(number: Derived, value: float) -> str:
    """Return a derived number as a target's box first shows it.

    Parameters
    ----------
    number : Derived
        The number, which has a target.
    value : float
        Its SI value.

    Returns
    -------
    str
        The number, in the target's unit.

    Examples
    --------
    >>> order = Target("soliton_order", "order", None, ())
    >>> _plain_number(Derived("n", "N", target=order), 3.741)
    '3.74'
    >>> zdw = Target("zdw", "wavelength", "nm", ())
    >>> _plain_number(Derived("z", "ZDW", target=zdw), 4.923e-7)
    '492'
    """
    target = number.target
    scaled = value * 1e9 if target is not None and target.unit == "nm" else value
    return f"{scaled:.3g}"


def _forms(meta: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    """Return a keyword's forms, as its metadata lists them.

    Parameters
    ----------
    meta : Mapping
        The keyword's metadata.

    Returns
    -------
    Sequence of Mapping
        Its forms; none where it lists none.
    """
    forms = meta.get("forms", ())
    return forms if isinstance(forms, list | tuple) else ()
