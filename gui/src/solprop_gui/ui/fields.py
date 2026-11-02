# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""One keyword's row on a setup page: its label, its field and its issue.

The row is made from the keyword's metadata as the kernel describes it (SEP 41, S9):
a boolean or a single choice is a list to pick from, with solprop's default first;
anything else is a line of text, read by :func:`solprop_gui.model.values.read_value`
as the user leaves it. An empty field, or the default picked, unsets the keyword, so
solprop's default applies and follows solprop's improvements.
"""

from collections.abc import Mapping, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.help.guide import guide_url
from solprop_gui.model.recipe import MISSING
from solprop_gui.model.values import FieldError, inline, read_value
from solprop_gui.registry.pages import label

__all__ = ["ParameterRow"]

_SEVERITY_STYLE = {
    "error": "color: #c62828;",
    "warning": "color: #b26a00;",
}


class ParameterRow(QWidget):
    """The row of one keyword.

    Parameters
    ----------
    meta : Mapping
        The keyword's metadata, as the kernel's ``describe`` gives it.
    table : str, optional
        The recipe table it belongs to: ``"build"`` or ``"execution"``.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    edited : Signal
        Emitted with (name, table, value) when the user changes the value; the value
        is :data:`~solprop_gui.model.recipe.MISSING` when it is unset.
    """

    edited = Signal(str, str, object)

    def __init__(
        self,
        meta: Mapping[str, object],
        table: str = "build",
        parent: QWidget | None = None,
    ) -> None:
        """Make the row's widgets from the metadata."""
        super().__init__(parent)
        self.meta = meta
        self.name = str(meta["name"])
        self.table = table
        self.forms: list[Mapping[str, object]] = list(meta.get("forms", []))  # pyright: ignore[reportArgumentType]
        self.level = str(meta.get("level", "basic"))
        self._value: object = MISSING
        self._updating = False

        self.label = QLabel(label(self.name))
        self.label.setMinimumWidth(170)
        self.issue = QLabel()
        self.issue.setWordWrap(True)
        self.issue.hide()
        kinds = {str(form.get("kind")) for form in self.forms}
        self.choices: list[str] = []
        if kinds == {"boolean"}:
            self.editor: QComboBox | QLineEdit = QComboBox()
            self.choices = ["on", "off"]
        elif kinds == {"choice"} and len(self.forms) == 1:
            self.editor = QComboBox()
            self.choices = [str(c) for c in self.forms[0].get("choices", [])]  # pyright: ignore[reportGeneralTypeIssues]
        else:
            self.editor = QLineEdit()
            self.editor.editingFinished.connect(self._text_finished)
        if isinstance(self.editor, QComboBox):
            self.editor.addItem(self._default_text())
            self.editor.addItems(self.choices)
            self.editor.currentIndexChanged.connect(self._choice_changed)
        else:
            self.editor.setPlaceholderText(self._default_text())
        self.editor.setObjectName(f"field-{self.name}")
        self.label.setBuddy(self.editor)
        self._tooltip()

        line = QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        line.addWidget(self.label)
        line.addWidget(self.editor, 1)
        unit = meta.get("display")
        self.unit = QLabel(str(unit) if unit else "")
        self.unit.setMinimumWidth(36)
        line.addWidget(self.unit)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.addLayout(line)
        layout.addWidget(self.issue)

    @property
    def is_set(self) -> bool:
        """Return whether the recipe sets this keyword.

        Returns
        -------
        bool
            False while solprop's default applies.
        """
        return self._value is not MISSING

    # ------------------------------------------------------------------ text

    def _default_text(self, resolved: str | None = None) -> str:
        """Return what the field says while the keyword is unset.

        Parameters
        ----------
        resolved : str or None, optional
            The value a rule resolves the default to, with the rule.

        Returns
        -------
        str
            ``required``, ``default: <value>`` or ``default``.
        """
        if self.meta.get("required"):
            return "required"
        if resolved is not None:
            return f"default: {resolved}"
        default = self.meta.get("default")
        if default is None:
            return "default (solprop chooses)"
        return f"default: {default}"

    def _tooltip(self) -> None:
        """Write the row's tooltip from the metadata."""
        parts = [str(self.meta.get("summary", ""))]
        unit = self.forms[0].get("unit") if self.forms else None
        if unit:
            parts.append(f"Unit: {unit} (a bare number is read in it).")
        default = self.meta.get("default")
        if default is not None:
            parts.append(f"Default: {default}.")
        parts.append(f"Keyword: {self.name}")
        anchor = self.meta.get("anchor")
        if anchor:
            parts.append(f"Manual: {guide_url(str(anchor))}")
        text = "\n".join(p for p in parts if p)
        self.label.setToolTip(text)
        self.editor.setToolTip(text)

    # ------------------------------------------------------------------ editing

    def _text_finished(self) -> None:
        """Read the text the user left, and announce a changed value."""
        if self._updating or not isinstance(self.editor, QLineEdit):
            return
        text = self.editor.text().strip()
        if not text:
            value: object = MISSING
        else:
            try:
                value = read_value(text, self.forms)
            except FieldError as error:
                self.set_issue(str(error), "error")
                return
        self.set_issue(None)
        if value != self._value:
            self._value = value
            self.edited.emit(self.name, self.table, value)

    def _choice_changed(self, index: int) -> None:
        """Announce the value picked from the list.

        Parameters
        ----------
        index : int
            The item picked: 0 is solprop's default.
        """
        if self._updating:
            return
        if index <= 0:
            value: object = MISSING
        elif self.choices == ["on", "off"]:
            value = index == 1
        else:
            value = self.choices[index - 1]
        if value != self._value:
            self._value = value
            self.edited.emit(self.name, self.table, value)

    # ------------------------------------------------------------------ showing

    def set_value(self, value: object) -> None:
        """Show the recipe's value, without announcing it.

        Parameters
        ----------
        value : object
            The value as the recipe holds it, or
            :data:`~solprop_gui.model.recipe.MISSING`.
        """
        self._updating = True
        try:
            self._value = value
            if isinstance(self.editor, QComboBox):
                self.editor.setCurrentIndex(self._index_of(value))
            else:
                self.editor.setText("" if value is MISSING else inline(value))
        finally:
            self._updating = False

    def _index_of(self, value: object) -> int:
        """Return the list item that shows a value.

        Parameters
        ----------
        value : object
            The value.

        Returns
        -------
        int
            Its item; 0, the default, for a value the list does not hold.
        """
        if value is MISSING:
            return 0
        if self.choices == ["on", "off"] and isinstance(value, bool):
            return 1 if value else 2
        text = str(value)
        return self.choices.index(text) + 1 if text in self.choices else 0

    def set_resolved(self, resolved: Sequence[object] | None) -> None:
        """Show what an unset keyword's default resolves to.

        Parameters
        ----------
        resolved : Sequence or None
            The value and the rule's sentence, or ``None``.
        """
        text = None
        if resolved is not None:
            value, rule = resolved[0], resolved[1]
            text = f"{inline(value)} ({rule})"
        if isinstance(self.editor, QComboBox):
            self.editor.setItemText(0, self._default_text(text))
        else:
            self.editor.setPlaceholderText(self._default_text(text))

    def set_issue(self, message: str | None, severity: str = "error") -> None:
        """Show, or clear, the issue solprop or the field found.

        Parameters
        ----------
        message : str or None
            The issue; ``None`` clears it.
        severity : {'error', 'warning'}, optional
            Its severity.
        """
        if message is None:
            self.issue.clear()
            self.issue.hide()
            return
        self.issue.setText(message)
        self.issue.setStyleSheet(_SEVERITY_STYLE.get(severity, ""))
        self.issue.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.issue.show()
