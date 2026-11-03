# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The settings panels: the start, and the setup steps built from solprop's metadata."""

from collections.abc import Iterable, Mapping, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.model.recipe import MISSING, RecipeModel
from solprop_gui.registry.pages import GROUP_TITLES, Page
from solprop_gui.ui.fields import ParameterRow
from solprop_gui.ui.objects import Catalog

__all__ = ["PER_WIDTH", "ParameterPage", "Section", "issue_keyword"]

PER_WIDTH = ("energy", "peak_power")
"""The keywords per metre of y on a slab set (solprop's ``PER_WIDTH_KEYWORDS``)."""


def issue_keyword(parameter: object) -> list[str]:
    """Return the keywords an issue's parameter paths name.

    Parameters
    ----------
    parameter : object
        The issue's parameter: a path, a list of paths, or ``None``.

    Returns
    -------
    list of str
        The first step of each path: the keyword whose row shows the issue.
    """
    if parameter is None:
        return []
    paths = parameter if isinstance(parameter, list | tuple) else [parameter]
    names = []
    for path in paths:
        text = str(path)
        for separator in (".", "["):
            text = text.split(separator, 1)[0]
        names.append(text)
    return names


class Section(QWidget):
    """A titled, collapsible part of a settings panel, as COMSOL's settings have.

    Parameters
    ----------
    title : str
        Its title.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """Make the section, open and empty."""
        super().__init__(parent)
        self.header = QToolButton()
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(True)
        self.header.setAutoRaise(True)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setArrowType(Qt.ArrowType.DownArrow)
        self.header.setStyleSheet("QToolButton { font-weight: bold; border: none; }")
        self.header.toggled.connect(self.set_open)
        self.body = QWidget()
        self.rows = QVBoxLayout(self.body)
        self.rows.setContentsMargins(12, 0, 0, 4)
        self.rows.setSpacing(0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.body)

    def set_open(self, open_: bool) -> None:
        """Show or fold the section's body.

        Parameters
        ----------
        open_ : bool
            Whether to show it.
        """
        self.header.setArrowType(
            Qt.ArrowType.DownArrow if open_ else Qt.ArrowType.RightArrow
        )
        self.body.setVisible(open_)

    def add(self, widget: QWidget) -> None:
        """Add a widget to the section's body.

        Parameters
        ----------
        widget : PySide6.QtWidgets.QWidget
            The widget.
        """
        self.rows.addWidget(widget)


class ParameterPage(QWidget):
    """One step's settings: the rows of the keywords in its groups, in sections.

    The page is a narrow panel at the window's left; a page of several groups
    shows each in a collapsible section.

    Parameters
    ----------
    page : solprop_gui.registry.pages.Page
        The page.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    edited : Signal
        Emitted with (name, table, value) when a row's value is changed.
    roots_wanted : Signal
        Emitted with a pinned keyword when the user asks for its pin's roots.
    """

    edited = Signal(str, str, object)
    roots_wanted = Signal(str)
    scan_wanted = Signal(str, object)

    def __init__(self, page: Page, parent: QWidget | None = None) -> None:
        """Make the page, with no rows until the route is described."""
        super().__init__(parent)
        self.page = page
        self.rows: dict[tuple[str, str], ParameterRow] = {}
        self.sections: dict[str, Section] = {}
        self.title = QLabel(f"<b>{page.title}</b>")
        self.help = QLabel(page.help)
        self.help.setWordWrap(True)
        self.help.setStyleSheet("color: palette(placeholder-text);")
        self.advanced = QCheckBox("Advanced")
        self.advanced.setToolTip("Show the expert settings too.")
        self.advanced.toggled.connect(self._show_levels)
        self.empty = QLabel(
            "<p><b>Select a setup first.</b></p><p>Choose a geometry class or one of "
            "solprop's examples on the Start step, or open a setup file.</p>"
        )
        self.empty.setObjectName("select-setup-first")
        self.empty.setWordWrap(True)
        self.empty.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.body = QWidget()
        self.form = QVBoxLayout(self.body)
        self.form.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.form.setContentsMargins(6, 4, 6, 4)
        self.form.setSpacing(2)
        self.top = QVBoxLayout()
        self.top.setContentsMargins(0, 0, 0, 0)
        self.form.addLayout(self.top)
        for group in page.groups:
            section = Section(GROUP_TITLES.get(group, group.capitalize()))
            section.setVisible(False)
            self.sections[group] = section
            self.form.addWidget(section)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self.body)
        header = QHBoxLayout()
        header.addWidget(self.title, 1)
        header.addWidget(self.advanced)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(header)
        layout.addWidget(self.help)
        layout.addWidget(self.empty, 1)
        layout.addWidget(scroll, 1)
        self.fields_area = scroll
        self.page_layout = layout
        self._show_empty(True)

    def add_footer(self, title: str, widget: QWidget) -> Section:
        """Put a section below the page's fields, always in view.

        Parameters
        ----------
        title : str
            Its title.
        widget : PySide6.QtWidgets.QWidget
            What it holds.

        Returns
        -------
        Section
            The section.
        """
        section = Section(title)
        section.add(widget)
        self.page_layout.addWidget(section)
        return section

    def _show_empty(self, empty: bool) -> None:
        """Show the page's fields, or that a setup must be chosen first.

        Parameters
        ----------
        empty : bool
            Whether there is no setup yet.
        """
        self.empty.setVisible(empty)
        self.fields_area.setVisible(not empty)
        self.advanced.setVisible(not empty)

    def add_section(self, title: str, widget: QWidget) -> Section:
        """Put a section of the window's own above the keywords' sections.

        Parameters
        ----------
        title : str
            Its title.
        widget : PySide6.QtWidgets.QWidget
            What it holds.

        Returns
        -------
        Section
            The section.
        """
        section = Section(title)
        section.add(widget)
        self.top.addWidget(section)
        return section

    def build(
        self,
        parameters: Iterable[Mapping[str, object]],
        table: str = "build",
        catalog: Catalog | None = None,
    ) -> None:
        """Add a row for each keyword of the page's groups.

        Parameters
        ----------
        parameters : Iterable of Mapping
            The route's keyword metadata, in its order.
        table : str, optional
            The recipe table they belong to.
        catalog : Catalog or None, optional
            The registered types the keywords can name, for their own editors.
        """
        for meta in parameters:
            if meta.get("group") not in self.page.groups:
                continue
            if not any(form.get("recipe", True) for form in meta.get("forms", [])):  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
                continue
            row = ParameterRow(meta, table, catalog=catalog)
            row.edited.connect(self.edited)
            row.scan_wanted.connect(self.scan_wanted)
            if row.pin_targets:
                row.pin.roots_wanted.connect(
                    lambda name=row.name: self.roots_wanted.emit(name)
                )
            self.rows[row.name, table] = row
            section = self.sections[str(meta.get("group"))]
            section.add(row)
            section.setVisible(True)
        self._show_empty(False)
        # One group needs no heading of its own: the page's title says it.
        for section in self.sections.values():
            section.header.setVisible(len(self.sections) > 1)
        self._show_levels(self.advanced.isChecked())

    def clear(self) -> None:
        """Remove every row."""
        for row in self.rows.values():
            row.setParent(None)
            row.deleteLater()
        self.rows = {}
        for section in self.sections.values():
            section.setVisible(False)

    def _show_levels(self, advanced: bool) -> None:
        """Show the basic rows, and the advanced ones when asked.

        Parameters
        ----------
        advanced : bool
            Whether to show the advanced rows.
        """
        for row in self.rows.values():
            row.setVisible(advanced or row.level == "basic" or row.is_set)

    def show_values(self, model: RecipeModel) -> None:
        """Show the recipe's values in the rows.

        Parameters
        ----------
        model : solprop_gui.model.recipe.RecipeModel
            The recipe.
        """
        advanced = self.advanced.isChecked()
        for (name, table), row in self.rows.items():
            value = model.get(name, table)
            pin = model.get(name, "pins") if row.pin_targets else MISSING
            if isinstance(pin, dict):
                row.set_pin(pin)
            else:
                row.set_value(value)
            # An advanced keyword the recipe sets is shown: nothing set is hidden.
            row.setVisible(advanced or row.level == "basic" or row.is_set)

    def show_per_width(self, per_width: bool) -> None:
        """Say on the energy and the peak power whether they are per metre of y.

        Parameters
        ----------
        per_width : bool
            Whether the setup is a slab mode set, uniform in y.
        """
        for (name, _), row in self.rows.items():
            if name in PER_WIDTH:
                row.set_per_width(per_width)

    def show_pinned(self, pinned: Mapping[str, object]) -> None:
        """Show the values solprop solved the pins to.

        Parameters
        ----------
        pinned : Mapping
            By pinned keyword, its solved value in the keyword's unit.
        """
        for (name, _), row in self.rows.items():
            if row.pin_targets:
                row.pin.set_solved(pinned.get(name), row.unit)

    def set_mode_labels(self, labels: Sequence[str]) -> None:
        """Offer the setup's mode labels to its tables of modes.

        Parameters
        ----------
        labels : Sequence of str
            The labels, as the mode preview gives them.
        """
        for row in self.rows.values():
            if "table" in row.modes:
                row.launch.set_labels(labels)

    def show_resolved(self, resolved: Mapping[str, Sequence[object]]) -> None:
        """Show what unset defaults resolve to.

        Parameters
        ----------
        resolved : Mapping
            By keyword, the value and the rule.
        """
        for (name, _), row in self.rows.items():
            row.set_resolved(resolved.get(name))

    def reveal(self, name: str) -> ParameterRow | None:
        """Show a keyword's row, opening its section, and focus its field.

        Parameters
        ----------
        name : str
            The keyword.

        Returns
        -------
        ParameterRow or None
            The row, or ``None`` if the page has none for it.
        """
        for (row_name, _), row in self.rows.items():
            if row_name == name:
                for section in self.sections.values():
                    if section.isAncestorOf(row):
                        section.header.setChecked(True)
                row.setVisible(True)
                row.editor.setFocus()
                return row
        return None

    def show_issues(self, issues: Sequence[Mapping[str, object]]) -> str | None:
        """Show the issues at their rows.

        Parameters
        ----------
        issues : Sequence of Mapping
            The issues solprop found: ``message``, ``parameter`` and ``severity``.

        Returns
        -------
        str or None
            The worst severity shown on this page, or ``None`` for none.
        """
        for row in self.rows.values():
            row.set_issue(None)
        worst: str | None = None
        for issue in issues:
            for name in issue_keyword(issue.get("parameter")):
                for (row_name, _), row in self.rows.items():
                    if row_name != name:
                        continue
                    severity = str(issue.get("severity", "error"))
                    row.set_issue(str(issue.get("message")), severity)
                    row.setVisible(True)
                    if worst != "error":
                        worst = severity
        return worst
