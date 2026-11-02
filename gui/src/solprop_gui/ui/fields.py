# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""One keyword's row on a setup page: its label, its field and its issue.

The row is made from the keyword's metadata as the kernel describes it (SEP 41, S9).
Each form the keyword takes that a widget can hold gets one:

- a boolean or a choice is a list with solprop's default selected and marked;
- a number is a number box, with the units of its dimension for a quantity;
- a pair of quantities is two number boxes sharing one unit;
- anything else (a table, a profile, an array) is a line of inline TOML.

Where a keyword takes several forms (a constant pressure or a gradient), a small
picker chooses the form. Leaving a field empty, or picking the default, unsets the
keyword so that solprop's default applies and follows solprop's improvements; the
reset button does the same for any form.
"""

from collections.abc import Mapping, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.help.guide import guide_url
from solprop_gui.model.recipe import MISSING
from solprop_gui.model.values import FieldError, inline, read_value
from solprop_gui.registry.pages import label
from solprop_gui.ui.quantity import QuantityField, RangeField

__all__ = ["ParameterRow", "modes_of"]

_SEVERITY_STYLE = {
    "error": "color: #c62828;",
    "warning": "color: #b26a00;",
}

_MODE_TITLES = {
    "choice": "choice",
    "number": "value",
    "range": "from … to",
    "text": "as text",
}


def modes_of(forms: Sequence[Mapping[str, object]]) -> list[str]:
    """Return the editors a keyword's forms call for, in order of preference.

    Parameters
    ----------
    forms : Sequence of Mapping
        The keyword's forms: ``kind`` and ``unit``.

    Returns
    -------
    list of str
        Of ``"choice"``, ``"number"``, ``"range"`` and ``"text"``; ``"text"`` is
        there whenever a form needs it.

    Examples
    --------
    >>> modes_of([{"kind": "number", "unit": "bar"}, {"kind": "pair", "unit": "bar"},
    ...           {"kind": "object"}])
    ['number', 'range', 'text']
    >>> modes_of([{"kind": "boolean"}])
    ['choice']
    """
    modes: list[str] = []
    for form in forms:
        kind = form.get("kind")
        if kind in ("boolean", "choice"):
            mode = "choice"
        elif kind in ("number", "integer"):
            mode = "number"
        elif kind == "pair" and form.get("unit") not in (None, "1"):
            mode = "range"
        else:
            mode = "text"
        if mode not in modes:
            modes.append(mode)
    order = ("choice", "number", "range", "text")
    return sorted(modes, key=order.index)


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

    def __init__(  # noqa: PLR0915 - one widget per form
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
        self.forms: list[Mapping[str, object]] = [
            form
            for form in meta.get("forms", [])  # pyright: ignore[reportGeneralTypeIssues]
            if form.get("recipe", True)
        ]
        self.level = str(meta.get("level", "basic"))
        self.default = meta.get("default_value")
        self.default_text = meta.get("default")
        self._value: object = MISSING
        self._updating = False
        self.resolved: Sequence[object] | None = None

        self.label = QLabel(label(self.name))
        self.label.setMinimumWidth(170)
        self.issue = QLabel()
        self.issue.setWordWrap(True)
        self.issue.hide()

        self.modes = modes_of(self.forms)
        self.stack = QStackedWidget()
        self.editors: dict[str, QWidget] = {}
        self.choices: list[object] = []
        if "choice" in self.modes:
            self.choice = QComboBox()
            self.choices = self._choice_values()
            self._fill_choices()
            self.choice.currentIndexChanged.connect(self._choice_changed)
            self.editors["choice"] = self.choice
        if "number" in self.modes:
            form = next(f for f in self.forms if f.get("kind") in ("number", "integer"))
            unit = form.get("unit")
            self.number = QuantityField(
                None if unit is None else str(unit),
                None if meta.get("display") is None else str(meta.get("display")),
                integer=form.get("kind") == "integer",
            )
            self.number.changed.connect(self._emit)
            self.number.refused.connect(lambda reason: self.set_issue(reason, "error"))
            self.editors["number"] = self.number
        if "range" in self.modes:
            form = next(f for f in self.forms if f.get("kind") == "pair")
            self.range = RangeField(
                str(form.get("unit")),
                None if meta.get("display") is None else str(meta.get("display")),
            )
            self.range.changed.connect(self._emit)
            self.range.refused.connect(lambda reason: self.set_issue(reason, "error"))
            self.editors["range"] = self.range
        if "text" in self.modes or not self.modes:
            if "text" not in self.modes:
                self.modes.append("text")
            self.text = QLineEdit()
            self.text.editingFinished.connect(self._text_finished)
            self.editors["text"] = self.text
        for mode in self.modes:
            self.stack.addWidget(self.editors[mode])
        fixed = QSizePolicy.Policy.Fixed
        self.stack.setSizePolicy(QSizePolicy.Policy.Expanding, fixed)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, fixed)

        self.mode = QComboBox()
        self.mode.setObjectName(f"mode-{self.name}")
        for mode in self.modes:
            self.mode.addItem(_MODE_TITLES[mode], mode)
        self.mode.setVisible(len(self.modes) > 1)
        self.mode.setToolTip("The form of the value.")
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.reset = QToolButton()
        self.reset.setText("↺")
        self.reset.setObjectName(f"reset-{self.name}")
        self.reset.setToolTip("Back to solprop's default.")
        self.reset.clicked.connect(lambda: self._emit(None))
        self.editor = self.editors[self.modes[0]]
        self.editor.setObjectName(f"field-{self.name}")
        self.label.setBuddy(self.editor)
        self._tooltip()

        line = QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        line.addWidget(self.label)
        line.addWidget(self.mode)
        line.addWidget(self.stack, 1)
        line.addWidget(self.reset)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.addLayout(line)
        layout.addWidget(self.issue)
        self._show_mode(self.modes[0])
        self.set_value(MISSING)

    # ------------------------------------------------------------------ choices

    def _choice_values(self) -> list[object]:
        """Return the values the list offers.

        Returns
        -------
        list
            ``True`` and ``False`` for a boolean; the choices otherwise.
        """
        values: list[object] = []
        for form in self.forms:
            if form.get("kind") == "boolean":
                values.extend([True, False])
            elif form.get("kind") == "choice":
                values.extend(form.get("choices", []))  # pyright: ignore[reportArgumentType]
        return values

    def _default_choice(self) -> int | None:
        """Return the list item of solprop's default, if the list holds it.

        Returns
        -------
        int or None
            Its index among :attr:`choices`.
        """
        default = self.default
        if self.resolved is not None:
            default = self.resolved[0]
        for index, value in enumerate(self.choices):
            if value == default and type(value) is type(default):
                return index
        return None

    def _fill_choices(self) -> None:
        """Write the list's items, the default marked; without one, a first item."""
        self._updating = True
        try:
            self.choice.clear()
            default = self._default_choice()
            if default is None:
                rule = f" ({self.resolved[1]})" if self.resolved is not None else ""
                self.choice.addItem(f"solprop chooses{rule}", MISSING)
            for index, value in enumerate(self.choices):
                text = {True: "on", False: "off"}.get(value, str(value))  # pyright: ignore[reportArgumentType, reportCallIssue]
                if index == default:
                    text += " (solprop's default)"
                self.choice.addItem(text, value)
        finally:
            self._updating = False

    def _choice_changed(self, index: int) -> None:
        """Announce the value picked: the default, or no default, unsets it.

        Parameters
        ----------
        index : int
            The item picked.
        """
        if self._updating or index < 0:
            return
        value = self.choice.itemData(index)
        default = self._default_choice()
        if value is MISSING or (default is not None and value == self.choices[default]):
            value = MISSING
        self._announce(value)

    def _choice_index(self, value: object) -> int:
        """Return the item that shows a value.

        Parameters
        ----------
        value : object
            The value, or :data:`~solprop_gui.model.recipe.MISSING`.

        Returns
        -------
        int
            The item; the default's (or the first) for an unset value.
        """
        offset = 1 if self._default_choice() is None else 0
        if value is MISSING:
            default = self._default_choice()
            return 0 if default is None else default
        for index, choice in enumerate(self.choices):
            if choice == value and type(choice) is type(value):
                return index + offset
        return 0

    # ------------------------------------------------------------------ modes

    def _fits(self, mode: str, value: object) -> bool:
        """Return whether an editor can show a value.

        Parameters
        ----------
        mode : str
            The editor.
        value : object
            The value.

        Returns
        -------
        bool
            Whether it holds that kind of value.
        """
        if mode == "choice":
            return any(c == value and type(c) is type(value) for c in self.choices)
        if mode == "number":
            return isinstance(value, int | float | str) and not isinstance(value, bool)
        if mode == "range":
            return (
                isinstance(value, list)
                and len(value) == 2
                and all(
                    isinstance(v, int | float | str) and not isinstance(v, bool)
                    for v in value
                )
            )
        return True

    def _show_mode(self, mode: str) -> None:
        """Show one editor, and pick it in the form picker.

        Parameters
        ----------
        mode : str
            The editor.
        """
        self.stack.setCurrentWidget(self.editors[mode])
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(self.modes.index(mode))
        self.mode.blockSignals(False)

    def _mode_changed(self, index: int) -> None:
        """Show the editor of the form picked; the value stays until it is edited.

        Parameters
        ----------
        index : int
            The form picked.
        """
        mode = self.modes[index]
        self.stack.setCurrentWidget(self.editors[mode])
        self._show(self._value, mode)

    # ------------------------------------------------------------------ editing

    def _emit(self, value: object) -> None:
        """Announce a value from a structured editor; ``None`` unsets it.

        Parameters
        ----------
        value : object
            The value, or ``None``.
        """
        if self._updating:
            return
        self.set_issue(None)
        self._announce(MISSING if value is None else value)

    def _announce(self, value: object) -> None:
        """Announce a changed value.

        Parameters
        ----------
        value : object
            The value, or :data:`~solprop_gui.model.recipe.MISSING`.
        """
        if value != self._value or type(value) is not type(self._value):
            self._value = value
            self.edited.emit(self.name, self.table, value)
        self.reset.setEnabled(self.is_set)

    def _text_finished(self) -> None:
        """Read the text the user left, and announce a changed value."""
        if self._updating:
            return
        text = self.text.text().strip()
        if not text:
            value: object = MISSING
        else:
            try:
                value = read_value(text, self.forms)
            except FieldError as error:
                self.set_issue(str(error), "error")
                return
        self.set_issue(None)
        self._announce(value)

    @property
    def is_set(self) -> bool:
        """Return whether the recipe sets this keyword.

        Returns
        -------
        bool
            False while solprop's default applies.
        """
        return self._value is not MISSING

    # ------------------------------------------------------------------ showing

    def _tooltip(self) -> None:
        """Write the row's tooltip from the metadata."""
        parts = [str(self.meta.get("summary", ""))]
        unit = self.forms[0].get("unit") if self.forms else None
        if unit:
            parts.append(f"Unit: {unit}; any unit of its dimension may be typed.")
        if self.default_text is not None:
            parts.append(f"solprop's default: {self.default_text}.")
        elif self.meta.get("required"):
            parts.append("Required.")
        parts.append(f"Keyword: {self.name}")
        anchor = self.meta.get("anchor")
        if anchor:
            parts.append(f"Manual: {guide_url(str(anchor))}")
        text = "\n".join(p for p in parts if p)
        self.label.setToolTip(text)
        for editor in self.editors.values():
            editor.setToolTip(text)

    def _placeholder(self) -> str:
        """Return what an empty text field says.

        Returns
        -------
        str
            ``required``, the default, or that solprop chooses.
        """
        if self.meta.get("required"):
            return "required"
        if self.resolved is not None:
            return f"default: {inline(self.resolved[0])} ({self.resolved[1]})"
        if self.default_text is not None:
            return f"default: {self.default_text}"
        return "default: solprop chooses"

    def _show(self, value: object, mode: str) -> None:
        """Show a value in one editor, without announcing it.

        Parameters
        ----------
        value : object
            The value, or :data:`~solprop_gui.model.recipe.MISSING`.
        mode : str
            The editor.
        """
        unset = value is MISSING
        default = self.resolved[0] if self.resolved is not None else self.default
        self._updating = True
        try:
            if mode == "choice":
                self.choice.setCurrentIndex(self._choice_index(value))
            elif mode == "number":
                shown = None if unset or not self._fits("number", value) else value
                fallback = default if self._fits("number", default) else None
                self.number.set_value(shown, fallback)
                if self.meta.get("required"):
                    self.number.number.setPlaceholderText("required")
            elif mode == "range":
                shown = None if unset or not self._fits("range", value) else value
                fallback = default if self._fits("range", default) else None
                self.range.set_value(shown, fallback)
            else:
                self.text.setText("" if unset else inline(value))
                self.text.setPlaceholderText(self._placeholder())
        finally:
            self._updating = False

    def set_value(self, value: object) -> None:
        """Show the recipe's value in the editor that fits it, without announcing it.

        Parameters
        ----------
        value : object
            The value as the recipe holds it, or
            :data:`~solprop_gui.model.recipe.MISSING`.
        """
        self._value = value
        current = self.modes[self.mode.currentIndex()]
        if value is MISSING or self._fits(current, value):
            mode = current
        else:
            mode = next((m for m in self.modes if self._fits(m, value)), current)
        self._show_mode(mode)
        self._show(value, mode)
        self.reset.setEnabled(self.is_set)

    def set_resolved(self, resolved: Sequence[object] | None) -> None:
        """Show what an unset keyword's default resolves to.

        Parameters
        ----------
        resolved : Sequence or None
            The value and the rule's sentence, or ``None``.
        """
        resolved = None if resolved is None else (resolved[0], resolved[1])
        if resolved == self.resolved:
            return
        self.resolved = resolved
        if "choice" in self.modes:
            self._fill_choices()
        mode = self.modes[self.mode.currentIndex()]
        self._show(self._value, mode)

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
