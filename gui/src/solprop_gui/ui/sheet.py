# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Setup stage: one sheet, built from a problem class's definition (design 6.2).

The frame is the same for every class: sections of rows at the left, with one line of
help on the focused row and Advanced folded at the end; what the setup gives at the
right, with its plot and the Ready line.
"""

from collections.abc import Mapping, Sequence

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.classes.catalog import class_of
from solprop_gui.classes.problem import (
    ProblemClass,
    derived_text,
    derived_value,
    field_meta,
    other_keywords,
)
from solprop_gui.model.recipe import MISSING, RecipeModel
from solprop_gui.registry.pages import label
from solprop_gui.ui.fields import ParameterRow
from solprop_gui.ui.pages import Section, issue_keyword
from solprop_gui.ui.theme import clear_layout, named_label, repolish

__all__ = ["SheetStage"]

_NARROW = 1000
"""Below this width, in pixels, the plot moves below the form."""


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
    rows : dict
        The rows, by path and table.
    derived : dict
        The labels of the derived numbers, by key.
    """

    edited = Signal(str, str, object)
    example_chosen = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:  # noqa: PLR0915 - a layout
        """Make the empty frame."""
        super().__init__(parent)
        self.setObjectName("page")
        self.problem: ProblemClass | None = None
        self.rows: dict[tuple[str, str], ParameterRow] = {}
        self.derived: dict[str, QLabel] = {}
        self._advanced_rows: list[ParameterRow] = []
        self._examples: list[Mapping[str, object]] = []
        self._title: str | None = None

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
        self.form.setMinimumWidth(440)
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
        self.preview_title = named_label("", "h3")
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
        right.addWidget(self.preview_title)
        right.addWidget(self.preview, 1)
        right.addWidget(self.ready)

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
        self._advanced_rows = []
        clear_layout(self.sections)
        catalog = description.get("types")
        for part in problem.sections:
            section = Section(part.title)
            section.header.setChecked(not part.folded)
            shown = False
            for row in part.fields:
                meta = field_meta(row.path, description, build)
                if meta is None:
                    continue
                name = row.path.rsplit(".", 1)[-1]
                entry = {**meta, "name": row.path, "label": row.label or label(name)}
                section.add(self._row(entry, catalog, one_line=True))
                shown = True
            section.setVisible(shown)
            self.sections.addWidget(section)
        self._add_advanced(problem, description, catalog)
        self._show_numbers(problem)
        self.preview_title.setText(problem.preview)
        self.preview_note.setText("This plot comes with the class's live sheet.")
        self.help.clear()
        self.ready.setText("Checking…")
        self.ready.setProperty("error", False)
        self.ready.setProperty("warn", False)
        repolish(self.ready)
        self._fill_examples()

    def _row(
        self, meta: Mapping[str, object], catalog: object, *, one_line: bool = False
    ) -> ParameterRow:
        """Make one row.

        Parameters
        ----------
        meta : Mapping
            Its keyword's metadata.
        catalog : object
            The registered types the route can name.
        one_line : bool, optional
            Whether its label sits beside its editor.

        Returns
        -------
        solprop_gui.ui.fields.ParameterRow
            The row, connected and held.
        """
        row = ParameterRow(meta, "build", catalog=catalog, one_line=one_line)  # pyright: ignore[reportArgumentType]
        row.edited.connect(self.edited)
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
        for meta in other_keywords(problem, description):
            forms: Sequence[Mapping[str, object]] = meta.get("forms", ())  # pyright: ignore[reportAssignmentType]
            if any(form.get("recipe", True) for form in forms):
                row = self._row(meta, catalog)
                self._advanced_rows.append(row)
                section.add(row)
        self.advanced = section
        self.sections.addWidget(section)

    def _filter_advanced(self, text: str) -> None:
        """Show the advanced rows that match what was typed.

        Parameters
        ----------
        text : str
            What to find, in a row's name, label or summary.
        """
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
            value = named_label("…", "derivedValue")
            tile.addWidget(value)
            self.derived[number.key] = value
            self.numbers.addLayout(tile, index // 3, index % 3)

    def _fill_examples(self) -> None:
        """Offer the examples of the sheet's class."""
        self.examples.clear()
        if self.problem is None:
            return
        for entry in self._examples:
            build: Mapping[str, object] = entry["build"]  # pyright: ignore[reportAssignmentType]
            if class_of(str(entry["route"]), build) is self.problem:
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
        for (path, table), row in self.rows.items():
            pin = model.get(path, "pins") if row.pin_targets else MISSING
            if isinstance(pin, dict):
                row.set_pin(pin)
            else:
                row.set_value(model.get(path, table))

    def show_issues(self, issues: Sequence[Mapping[str, object]]) -> None:
        """Show solprop's issues beside their rows, and its verdict on the Ready line.

        Parameters
        ----------
        issues : Sequence of Mapping
            The issues: ``message``, ``parameter`` and ``severity``.
        """
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
        self.ready.setText(text)
        self.ready.setProperty("error", bool(errors))
        self.ready.setProperty("warn", bool(warnings) and not errors)
        repolish(self.ready)

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
        for number in self.problem.derived:
            value = derived_value(summary, number.key)
            widget = self.derived[number.key]
            widget.setText(derived_text(number, value))
            widget.setToolTip("" if summary is not None else str(error or ""))
