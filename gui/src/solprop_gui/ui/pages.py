# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The start screen and the setup pages built from solprop's metadata."""

from collections.abc import Iterable, Mapping, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.model.recipe import MISSING, RecipeModel
from solprop_gui.model.workspace import RunEntry
from solprop_gui.registry.pages import Page
from solprop_gui.registry.routes import CARDS
from solprop_gui.ui.fields import ParameterRow

__all__ = ["ParameterPage", "StartPage", "issue_keyword"]


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


class ParameterPage(QWidget):
    """One setup page: the rows of the keywords in its groups.

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
    """

    edited = Signal(str, str, object)

    def __init__(self, page: Page, parent: QWidget | None = None) -> None:
        """Make the page, with no rows until the route is described."""
        super().__init__(parent)
        self.page = page
        self.rows: dict[tuple[str, str], ParameterRow] = {}
        self.title = QLabel(f"<h2>{page.title}</h2>")
        self.help = QLabel(page.help)
        self.help.setWordWrap(True)
        self.advanced = QCheckBox("Show advanced settings")
        self.advanced.toggled.connect(self._show_levels)
        self.body = QWidget()
        self.form = QVBoxLayout(self.body)
        self.form.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.extra = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.body)
        header = QHBoxLayout()
        header.addWidget(self.title, 1)
        header.addWidget(self.advanced)
        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.help)
        layout.addWidget(scroll, 1)
        layout.addLayout(self.extra)

    def build(
        self,
        parameters: Iterable[Mapping[str, object]],
        table: str = "build",
    ) -> None:
        """Add a row for each keyword of the page's groups.

        Parameters
        ----------
        parameters : Iterable of Mapping
            The route's keyword metadata, in its order.
        table : str, optional
            The recipe table they belong to.
        """
        for meta in parameters:
            if meta.get("group") not in self.page.groups:
                continue
            if not any(form.get("recipe", True) for form in meta.get("forms", [])):  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
                continue
            row = ParameterRow(meta, table)
            row.edited.connect(self.edited)
            self.rows[row.name, table] = row
            self.form.addWidget(row)
        self._show_levels(self.advanced.isChecked())

    def clear(self) -> None:
        """Remove every row."""
        for row in self.rows.values():
            row.setParent(None)
            row.deleteLater()
        self.rows = {}

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
            row.set_value(value)
            # An advanced keyword the recipe sets is shown: nothing set is hidden.
            row.setVisible(advanced or row.level == "basic" or value is not MISSING)

    def show_resolved(self, resolved: Mapping[str, Sequence[object]]) -> None:
        """Show what unset defaults resolve to.

        Parameters
        ----------
        resolved : Mapping
            By keyword, the value and the rule.
        """
        for (name, _), row in self.rows.items():
            row.set_resolved(resolved.get(name))

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


class StartPage(QWidget):
    """Where a setup begins: a route card, an example, a file or a past run.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    card_chosen : Signal
        Emitted with a card's key.
    example_chosen : Signal
        Emitted with an example's name.
    run_chosen : Signal
        Emitted with a run folder's results file.
    """

    card_chosen = Signal(str)
    example_chosen = Signal(str)
    run_chosen = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the screen; the examples and runs are filled in later."""
        super().__init__(parent)
        cards = QGroupBox("New setup")
        cards_layout = QVBoxLayout(cards)
        self.card_buttons: dict[str, QPushButton] = {}
        for card in CARDS:
            button = QPushButton(card.title)
            button.setObjectName(f"card-{card.key}")
            button.setToolTip(card.reason or card.summary)
            button.setEnabled(card.enabled)
            button.clicked.connect(
                lambda _checked=False, key=card.key: self.card_chosen.emit(key)
            )
            cards_layout.addWidget(button)
            self.card_buttons[card.key] = button
        cards_layout.addStretch(1)

        examples = QGroupBox("Start from one of solprop's examples")
        self.examples = QListWidget()
        self.examples.setObjectName("examples")
        self.examples.itemActivated.connect(
            lambda item: self.example_chosen.emit(item.data(Qt.ItemDataRole.UserRole))
        )
        QVBoxLayout(examples).addWidget(self.examples)

        runs = QGroupBox("Runs in the workspace")
        self.runs = QListWidget()
        self.runs.setObjectName("runs")
        self.runs.itemActivated.connect(
            lambda item: self.run_chosen.emit(item.data(Qt.ItemDataRole.UserRole))
        )
        QVBoxLayout(runs).addWidget(self.runs)

        title = QLabel(
            "<h1>solprop</h1><p>Set up, run and read an ultrafast propagation "
            "simulation. Every setup is a solprop recipe, and can leave as a recipe "
            "file or a Python script.</p>"
        )
        title.setWordWrap(True)
        columns = QHBoxLayout()
        columns.addWidget(cards, 1)
        columns.addWidget(examples, 2)
        columns.addWidget(runs, 2)
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(columns, 1)

    def show_examples(
        self, examples: Sequence[Mapping[str, object]], routes: Iterable[str]
    ) -> None:
        """List solprop's examples, those of routes the window offers enabled.

        Parameters
        ----------
        examples : Sequence of Mapping
            ``name``, ``title`` and ``route`` of each.
        routes : Iterable of str
            The routes the window can show.
        """
        allowed = set(routes)
        self.examples.clear()
        for example in examples:
            item = QListWidgetItem(f"{example.get('title') or example['name']}")
            item.setData(Qt.ItemDataRole.UserRole, example["name"])
            item.setToolTip(f"{example['name']} ({example['route']})")
            if example["route"] not in allowed:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setToolTip(f"The {example['route']} route comes later.")
            self.examples.addItem(item)

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
            self.runs.addItem(item)
