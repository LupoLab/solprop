# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A pinned value's editor: a design target in place of a number (design 7.5).

A pinnable keyword's row offers "pinned" beside its value. Pinned, the keyword is set
by solprop from a target (a zero-dispersion wavelength, a dispersive-wave wavelength
or a soliton order) within a search range, and the editor writes the recipe's
``[pins]`` entry, with the root it takes (the lowest or the highest in the range);
the value solprop solves for is shown beside it.
"""

from collections.abc import Mapping, Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from solprop.service.explore import PIN_KEYS, PIN_SEARCH, PIN_TITLES

from solprop_gui.model.values import inline
from solprop_gui.ui.quantity import QuantityField, RangeField

__all__ = ["PinField"]

_ROOTS = (("lowest root", "lowest"), ("highest root", "highest"))


class PinField(QWidget):
    """The editor of one keyword's pin.

    Parameters
    ----------
    name : str
        The keyword.
    targets : Sequence of Mapping
        The targets it can be pinned to, as the kernel describes them: ``target``,
        ``stage`` and ``required``.
    unit : str or None
        The keyword's unit, for the search range.
    display : str or None
        The unit it is shown in.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    changed : Signal
        Emitted with the ``[pins]`` entry when it is complete and changed.
    refused : Signal
        Emitted with the reason a part of it does not read.
    """

    changed = Signal(object)
    refused = Signal(str)

    def __init__(
        self,
        name: str,
        targets: Sequence[Mapping[str, object]],
        unit: str | None,
        display: str | None,
        parent: QWidget | None = None,
    ) -> None:
        """Make one page of fields per target, and the search and root pickers."""
        super().__init__(parent)
        self.name = name
        self.targets = list(targets)
        self._updating = False
        self._entry: dict[str, object] | None = None
        self.target = QComboBox()
        self.target.setObjectName(f"pin-target-{name}")
        self.keys = QStackedWidget()
        self.fields: dict[tuple[str, str], QuantityField] = {}
        for target in self.targets:
            self.target.addItem(PIN_TITLES[str(target["target"])], target["target"])
            page = QWidget()
            grid = QGridLayout(page)
            grid.setContentsMargins(0, 0, 0, 0)
            required: list[str] = list(target["required"])  # pyright: ignore[reportArgumentType]
            for row, key in enumerate(required):
                title, key_unit, key_display = PIN_KEYS[key]
                field = QuantityField(key_unit, key_display)
                field.changed.connect(self._changed)
                field.refused.connect(self.refused)
                self.fields[str(target["target"]), key] = field
                grid.addWidget(QLabel(title), row, 0)
                grid.addWidget(field, row, 1)
            self.keys.addWidget(page)
        self.target.currentIndexChanged.connect(self._target_changed)
        self.search = RangeField(unit or "1", display)
        self.search.changed.connect(self._changed)
        self.search.refused.connect(self.refused)
        self.search_label = QLabel("Search")
        self.root = QComboBox()
        self.root.setObjectName(f"pin-root-{name}")
        for text, value in _ROOTS:
            self.root.addItem(text, value)
        self.root.currentIndexChanged.connect(self._changed)
        self.solved = QLabel("")
        self.solved.setObjectName(f"pin-solved-{name}")
        self.solved.setWordWrap(True)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self.search_label, 0, 0)
        grid.addWidget(self.search, 0, 1, 1, 2)
        grid.addWidget(self.root, 1, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.target)
        layout.addWidget(self.keys)
        layout.addLayout(grid)
        layout.addWidget(self.solved)
        self._target_changed(0)

    # ------------------------------------------------------------------ reading

    def _current(self) -> Mapping[str, object]:
        """Return the target picked.

        Returns
        -------
        Mapping
            Its description.
        """
        return self.targets[max(self.target.currentIndex(), 0)]

    def _target_changed(self, index: int) -> None:
        """Show the picked target's fields, and the search range if it needs one.

        Parameters
        ----------
        index : int
            The target picked.
        """
        self.keys.setCurrentIndex(max(index, 0))
        guide = self._current()["stage"] == "guide"
        for widget in (self.search_label, self.search, self.root):
            widget.setVisible(guide)
        if guide and self.search.read() is None and self.name in PIN_SEARCH:
            self.search.set_value(PIN_SEARCH[self.name])
        self._changed()

    def entry(self) -> dict[str, object] | None:
        """Return the ``[pins]`` entry the fields make, if complete.

        Returns
        -------
        dict or None
            ``target``, its keys, ``search`` (on the guide) and ``root``; ``None``
            while a required key or the search range is empty.
        """
        target = self._current()
        entry: dict[str, object] = {"target": target["target"]}
        for key in target["required"]:  # pyright: ignore[reportGeneralTypeIssues]
            value = self.fields[str(target["target"]), key].read()
            if value is None:
                return None
            entry[key] = value
        if target["stage"] == "guide":
            search = self.search.read()
            if search is None:
                return None
            entry["search"] = search
            root = self.root.currentData()
            if root != "lowest":
                entry["root"] = root
        return entry

    def _changed(self, *_args: object) -> None:
        """Announce the entry when it is complete and differs from the last.

        Parameters
        ----------
        *_args : object
            What the signal that calls it carries, unused.
        """
        if self._updating:
            return
        try:
            entry = self.entry()
        except ValueError as error:
            self.refused.emit(str(error))
            return
        if entry is not None and entry != self._entry:
            self._entry = entry
            self.changed.emit(entry)

    # ------------------------------------------------------------------ showing

    def set_value(self, entry: Mapping[str, object]) -> None:
        """Show a recipe's ``[pins]`` entry, without announcing it.

        Parameters
        ----------
        entry : Mapping
            The entry.
        """
        self._updating = True
        try:
            index = self.target.findData(entry.get("target"))
            self.target.setCurrentIndex(max(index, 0))
            self.keys.setCurrentIndex(max(index, 0))
            target = self._current()
            for key in target["required"]:  # pyright: ignore[reportGeneralTypeIssues]
                self.fields[str(target["target"]), key].set_value(entry.get(key))
            search = entry.get("search")
            if search is not None:
                self.search.set_value(search)
            root = entry.get("root", "lowest")
            found = self.root.findData(root)
            if found < 0:
                self.root.addItem(f"root {root}", root)
                found = self.root.count() - 1
            self.root.setCurrentIndex(found)
        finally:
            self._updating = False
        self._entry = dict(entry)
        guide = self._current()["stage"] == "guide"
        for widget in (self.search_label, self.search, self.root):
            widget.setVisible(guide)

    def set_solved(self, value: object, unit: str | None) -> None:
        """Show the value solprop solved the pin to.

        Parameters
        ----------
        value : object
            The value in the keyword's unit, or ``None`` when it is not known.
        unit : str or None
            The keyword's unit.
        """
        if value is None:
            self.solved.setText("")
            return
        text = f"{value:.6g}" if isinstance(value, float) else inline(value)
        self.solved.setText(f"Solved: {text} {unit or ''}".rstrip())
