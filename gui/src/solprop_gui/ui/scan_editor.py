# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The scan editor: the recipe's ``[scan]`` table (design 12.7).

A scan varies parameter paths over a grid of points: each variable a path (a keyword,
or a nested one such as ``pulses[0].energy``) with a list of values or a range
(``{start, stop, num, spacing}``), read in the keyword's unit, and a ``name`` that names
the scan's files and seeds its points. Turned on, Run runs the scan, point by point,
and its results open in the scan browser; pins are solved again at every point.
"""

from collections.abc import Mapping, Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from solprop.service.explore import sweep_values

__all__ = ["ScanEditor", "point_count"]


def point_count(table: Mapping[str, object]) -> int:
    """Return the number of points a ``[scan]`` table makes.

    Parameters
    ----------
    table : Mapping
        The table: ``name``, and each path's list or range.

    Returns
    -------
    int
        The product of the variables' lengths; 0 without a variable.

    Examples
    --------
    >>> point_count({"name": "s", "pressure": [1, 2], "energy": {"num": 3}})
    6
    >>> point_count({"name": "s"})
    0
    """
    total = 1
    found = False
    for key, value in table.items():
        if key in ("name", "base_seed"):
            continue
        found = True
        if isinstance(value, Mapping):
            total *= int(value.get("num", 0))  # pyright: ignore[reportArgumentType]
        elif isinstance(value, Sequence) and not isinstance(value, str):
            total *= len(value)
    return total if found else 0


class _Variable(QWidget):
    """One variable: its path, and a list of values or a range.

    Parameters
    ----------
    paths : Sequence of str
        The paths offered (the route's numeric keywords); any other may be typed.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    changed : Signal
        Emitted when any part changes.
    """

    changed = Signal()

    def __init__(self, paths: Sequence[str], parent: QWidget | None = None) -> None:
        """Make the path, the form and the fields of both forms."""
        super().__init__(parent)
        self.path = QComboBox()
        self.path.setEditable(True)
        self.path.addItems(list(paths))
        self.path.setObjectName("scan-path")
        self.form = QComboBox()
        self.form.addItem("values", "list")
        self.form.addItem("range", "range")
        self.values = QLineEdit()
        self.values.setPlaceholderText("e.g. 0.5, 1, 2 bar")
        self.start = QLineEdit()
        self.start.setPlaceholderText("from, e.g. 0.5 bar")
        self.stop = QLineEdit()
        self.stop.setPlaceholderText("to, e.g. 2 bar")
        self.num = QSpinBox()
        self.num.setRange(2, 10_000)
        self.num.setValue(5)
        self.num.setSuffix(" points")
        self.spacing = QComboBox()
        self.spacing.addItems(["linear", "log"])
        self.remove = QToolButton()
        self.remove.setText("−")  # noqa: RUF001 - a minus sign
        self.remove.setAutoRaise(True)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self.path, 0, 0, 1, 2)
        grid.addWidget(self.form, 0, 2)
        grid.addWidget(self.remove, 0, 3)
        grid.addWidget(self.values, 1, 0, 1, 3)
        grid.addWidget(self.start, 2, 0)
        grid.addWidget(self.stop, 2, 1)
        grid.addWidget(self.num, 3, 0)
        grid.addWidget(self.spacing, 3, 1)
        self.path.currentTextChanged.connect(self.changed)
        self.form.currentIndexChanged.connect(self._form_changed)
        for line in (self.values, self.start, self.stop):
            line.editingFinished.connect(self.changed)
        self.num.valueChanged.connect(self.changed)
        self.spacing.currentIndexChanged.connect(self.changed)
        self._form_changed()

    def _form_changed(self, *_args: object) -> None:
        """Show the fields of the form picked.

        Parameters
        ----------
        *_args : object
            What the signal that calls it carries, unused.
        """
        ranged = self.form.currentData() == "range"
        self.values.setVisible(not ranged)
        for widget in (self.start, self.stop, self.num, self.spacing):
            widget.setVisible(ranged)
        self.changed.emit()

    def entry(self) -> tuple[str, object] | None:
        """Return the variable as the table holds it.

        Returns
        -------
        tuple or None
            The path and its list or range; ``None`` while incomplete.
        """
        path = self.path.currentText().strip()
        if not path:
            return None
        if self.form.currentData() == "range":
            start, stop = self.start.text().strip(), self.stop.text().strip()
            if not start or not stop:
                return None
            return path, {
                "start": start,
                "stop": stop,
                "num": self.num.value(),
                "spacing": self.spacing.currentText(),
            }
        values = sweep_values(self.values.text())
        return (path, _numbers(values)) if values else None

    def set_entry(self, path: str, value: object) -> None:
        """Show a variable of a table.

        Parameters
        ----------
        path : str
            Its path.
        value : object
            Its list or range.
        """
        self.path.setCurrentText(path)
        if isinstance(value, Mapping):
            self.form.setCurrentIndex(1)
            self.start.setText(str(value.get("start", "")))
            self.stop.setText(str(value.get("stop", "")))
            self.num.setValue(int(value.get("num", 5)))  # pyright: ignore[reportArgumentType]
            self.spacing.setCurrentText(str(value.get("spacing", "linear")))
        else:
            self.form.setCurrentIndex(0)
            items = value if isinstance(value, list) else [value]
            self.values.setText(", ".join(str(item) for item in items))


def _numbers(values: Sequence[str]) -> list[object]:
    """Return a list's values, a bare number as a number.

    Parameters
    ----------
    values : Sequence of str
        The values as typed.

    Returns
    -------
    list
        Each a number where it is one, else the text (a quantity or a name).

    Examples
    --------
    >>> _numbers(["1", "2.5", "3 bar", "He"])
    [1, 2.5, '3 bar', 'He']
    """
    out: list[object] = []
    for value in values:
        try:
            number = float(value)
        except ValueError:
            out.append(value)
            continue
        out.append(int(number) if value.lstrip("+-").isdigit() else number)
    return out


class ScanEditor(QWidget):
    """The ``[scan]`` table: a switch, its name and its variables.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    changed : Signal
        Emitted with the table, or ``None`` when the scan is off.
    """

    changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the switch, the name, and an empty list of variables."""
        super().__init__(parent)
        self.paths: list[str] = []
        self.variables: list[_Variable] = []
        self._updating = False
        self.enabled = QCheckBox("Run as a scan")
        self.enabled.setObjectName("scan-enabled")
        self.enabled.toggled.connect(self._toggled)
        self.name = QLineEdit()
        self.name.setObjectName("scan-name")
        self.name.setPlaceholderText("name, e.g. pressure-scan")
        self.name.editingFinished.connect(self._changed)
        self.column = QVBoxLayout()
        self.column.setContentsMargins(0, 0, 0, 0)
        self.add_button = QToolButton()
        self.add_button.setText("+ variable")
        self.add_button.setAutoRaise(True)
        self.add_button.clicked.connect(self.add_variable)
        self.count = QLabel("")
        self.count.setObjectName("scan-count")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.enabled)
        layout.addWidget(self.name)
        layout.addLayout(self.column)
        layout.addWidget(self.add_button)
        layout.addWidget(self.count)
        self._show_parts(False)

    def set_paths(self, paths: Sequence[str]) -> None:
        """Offer a route's numeric keywords as paths.

        Parameters
        ----------
        paths : Sequence of str
            The keywords.
        """
        self.paths = list(paths)

    def _show_parts(self, on: bool) -> None:
        """Show the name and the variables only while the scan is on.

        Parameters
        ----------
        on : bool
            Whether it is.
        """
        for widget in (self.name, self.add_button, self.count, *self.variables):
            widget.setVisible(on)

    def _variable(self) -> _Variable:
        """Append an empty variable.

        Returns
        -------
        _Variable
            It.
        """
        variable = _Variable(self.paths)
        variable.changed.connect(self._changed)
        variable.remove.clicked.connect(lambda: self.remove_variable(variable))
        self.column.addWidget(variable)
        self.variables.append(variable)
        return variable

    def scan(self, path: str, value: object) -> None:
        """Turn the scan on with a keyword as a variable, starting from its value.

        Parameters
        ----------
        path : str
            The keyword.
        value : object
            Its value now, or ``None``; the list starts with it.
        """
        # Announced once, complete: the window's echo of a partial table would
        # rebuild the editor under the variable being filled.
        self._updating = True
        try:
            self.enabled.setChecked(True)
            self._show_parts(True)
            empty = (v for v in self.variables if not v.path.currentText().strip())
            variable = next(empty, None) or self._variable()
            variable.set_entry(path, [] if value is None else [value])
        finally:
            self._updating = False
        self._changed()

    def add_variable(self) -> None:
        """Add a variable, and announce the table."""
        self._variable()
        self._changed()

    def remove_variable(self, variable: _Variable) -> None:
        """Remove a variable, and announce the table.

        Parameters
        ----------
        variable : _Variable
            It.
        """
        self.variables.remove(variable)
        variable.setParent(None)
        variable.deleteLater()
        self._changed()

    def table(self) -> dict[str, object] | None:
        """Return the ``[scan]`` table the editor holds.

        Returns
        -------
        dict or None
            ``name`` and each complete variable; ``None`` when the scan is off.
        """
        if not self.enabled.isChecked():
            return None
        table: dict[str, object] = {"name": self.name.text().strip() or "scan"}
        for variable in self.variables:
            entry = variable.entry()
            if entry is not None:
                table[entry[0]] = entry[1]
        return table

    def set_value(self, table: object) -> None:
        """Show a recipe's ``[scan]`` table, without announcing it.

        Parameters
        ----------
        table : object
            The table, or anything else for none.
        """
        self._updating = True
        try:
            while self.variables:
                variable = self.variables.pop()
                variable.setParent(None)
                variable.deleteLater()
            on = isinstance(table, Mapping)
            self.enabled.setChecked(on)
            if isinstance(table, Mapping):
                self.name.setText(str(table.get("name", "")))
                for key, value in table.items():
                    if key in ("name", "base_seed"):
                        continue
                    self._variable().set_entry(str(key), value)
            self._show_parts(on)
            self._count(table if isinstance(table, Mapping) else None)
        finally:
            self._updating = False

    def _count(self, table: Mapping[str, object] | None) -> None:
        """Say how many points the scan makes.

        Parameters
        ----------
        table : Mapping or None
            The table.
        """
        points = 0 if table is None else point_count(table)
        self.count.setText(f"{points} points" if points else "No variable yet.")

    def _toggled(self, on: bool) -> None:
        """Show or hide the scan's parts, and announce the table.

        Parameters
        ----------
        on : bool
            Whether the scan is on.
        """
        self._show_parts(on)
        if on and not self.variables and not self._updating:
            self._variable()  # a first, empty variable to fill in
        self._changed()

    def _changed(self, *_args: object) -> None:
        """Announce the table.

        Parameters
        ----------
        *_args : object
            What the signal that calls it carries, unused.
        """
        if self._updating:
            return
        table = self.table()
        self._count(table)
        self.changed.emit(table)
