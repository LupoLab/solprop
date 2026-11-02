# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The launch table: which modes the pulse enters, with what share and phase (9.2).

Each row names a mode by solprop's label (``HE1,1``), its share (energy fractions,
normalised by solprop) and its phase in radians. The labels offered are the ones the
mode preview gives for the setup; a label typed by hand is checked by solprop. One
table launches every pulse alike; a list of tables, one per pulse, is edited as text.
"""

from collections.abc import Mapping, Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

__all__ = ["LaunchField", "launch_rows", "launch_value"]


def launch_rows(value: object) -> list[tuple[str, float, float]] | None:
    """Return a launch mapping's rows, if it is one mapping.

    Parameters
    ----------
    value : object
        The recipe's ``launch``.

    Returns
    -------
    list of tuple or None
        Each mode's label, share and phase; ``None`` for a value that is not one
        mapping of shares (a list per pulse, say).

    Examples
    --------
    >>> launch_rows({"HE1,1": 0.9, "HE1,2": {"share": 0.1, "phase": 1.57}})
    [('HE1,1', 0.9, 0.0), ('HE1,2', 0.1, 1.57)]
    >>> launch_rows([{"HE1,1": 1.0}]) is None
    True
    """
    if not isinstance(value, Mapping):
        return None
    rows: list[tuple[str, float, float]] = []
    for name, entry in value.items():
        if isinstance(entry, int | float) and not isinstance(entry, bool):
            rows.append((str(name), float(entry), 0.0))
        elif isinstance(entry, Mapping) and isinstance(entry.get("share"), int | float):
            rows.append(
                (str(name), float(entry["share"]), float(entry.get("phase", 0.0)))  # pyright: ignore[reportArgumentType]
            )
        else:
            return None
    return rows


def launch_value(rows: Sequence[tuple[str, float, float]]) -> dict[str, object]:
    """Return the launch mapping of rows: a bare share where the phase is zero.

    Parameters
    ----------
    rows : Sequence of tuple
        Each mode's label, share and phase.

    Returns
    -------
    dict
        The recipe's ``launch``.

    Examples
    --------
    >>> launch_value([("HE1,1", 0.9, 0.0), ("HE1,2", 0.1, 1.57)])
    {'HE1,1': 0.9, 'HE1,2': {'share': 0.1, 'phase': 1.57}}
    """
    return {
        name: share if phase == 0.0 else {"share": share, "phase": phase}
        for name, share, phase in rows
    }


class LaunchField(QWidget):
    """A table of the launched modes.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    changed : Signal
        Emitted with the launch mapping, or ``None`` for an empty table.
    refused : Signal
        Emitted with the reason a cell does not read.
    """

    changed = Signal(object)
    refused = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the empty table and its buttons."""
        super().__init__(parent)
        self.labels: list[str] = []
        self._updating = False
        self.table = QTableWidget(0, 3)
        self.table.setObjectName("launch-table")
        self.table.setHorizontalHeaderLabels(["Mode", "Share", "Phase (rad)"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(130)
        self.table.itemChanged.connect(self._edited)
        self.add = QToolButton()
        self.add.setText("+")
        self.add.setToolTip("Launch into another mode.")
        self.add.clicked.connect(self.add_row)
        self.remove = QToolButton()
        self.remove.setText("−")  # noqa: RUF001 - a minus sign
        self.remove.setToolTip("Remove the selected mode.")
        self.remove.clicked.connect(self.remove_row)
        buttons = QHBoxLayout()
        buttons.addWidget(self.add)
        buttons.addWidget(self.remove)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.table)
        layout.addLayout(buttons)

    def set_labels(self, labels: Sequence[str]) -> None:
        """Know the setup's mode labels, for new rows.

        Parameters
        ----------
        labels : Sequence of str
            The labels, as the mode preview gives them.
        """
        self.labels = [label.split(":")[0] for label in labels]

    def _row(self, name: str, share: float, phase: float) -> None:
        """Append a row.

        Parameters
        ----------
        name : str
            The mode's label.
        share : float
            Its share.
        phase : float
            Its phase, rad.
        """
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, text in enumerate((name, f"{share:g}", f"{phase:g}")):
            self.table.setItem(row, column, QTableWidgetItem(text))

    def rows(self) -> list[tuple[str, float, float]]:
        """Return the table's rows.

        Returns
        -------
        list of tuple
            Each mode's label, share and phase.

        Raises
        ------
        ValueError
            If a share or a phase is not a number.
        """
        rows = []
        for row in range(self.table.rowCount()):
            cells = [self.table.item(row, column) for column in range(3)]
            name = cells[0].text().strip() if cells[0] else ""
            if not name:
                continue
            share = float(cells[1].text()) if cells[1] else 0.0
            phase = float(cells[2].text() or 0.0) if cells[2] else 0.0
            rows.append((name, share, phase))
        return rows

    def set_value(self, value: object) -> bool:
        """Show a launch mapping, without announcing it.

        Parameters
        ----------
        value : object
            The recipe's ``launch``, or ``None``.

        Returns
        -------
        bool
            Whether the table can show it (one mapping of shares).
        """
        rows = [] if value is None else launch_rows(value)
        if rows is None:
            return False
        self._updating = True
        try:
            self.table.setRowCount(0)
            for row in rows:
                self._row(*row)
        finally:
            self._updating = False
        return True

    def add_row(self) -> None:
        """Add a row for the first mode not yet in the table."""
        used = {name for name, _, _ in self._names()}
        name = next((label for label in self.labels if label not in used), "")
        self._updating = True
        try:
            self._row(name, 1.0 if not used else 0.0, 0.0)
        finally:
            self._updating = False
        self._edited()

    def _names(self) -> list[tuple[str, int, int]]:
        """Return the table's labels.

        Returns
        -------
        list of tuple
            Each row's label, with two placeholders.
        """
        names = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            names.append((item.text() if item else "", row, 0))
        return names

    def remove_row(self) -> None:
        """Remove the selected row, or the last."""
        row = self.table.currentRow()
        if self.table.rowCount() == 0:
            return
        self.table.removeRow(row if row >= 0 else self.table.rowCount() - 1)
        self._edited()

    def _edited(self, *_args: object) -> None:
        """Announce the table's launch.

        Parameters
        ----------
        *_args : object
            What the signal that calls it carries, unused.
        """
        if self._updating:
            return
        try:
            rows = self.rows()
        except ValueError as error:
            self.refused.emit(f"a share and a phase are numbers: {error}")
            return
        self.changed.emit(launch_value(rows) if rows else None)
