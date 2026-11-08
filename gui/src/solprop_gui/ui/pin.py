# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A pinned value's editor: a design target in place of a number (design 7.5).

A pinnable keyword's row offers "pinned" beside its value. Pinned, the keyword is set
by solprop from a target (a zero-dispersion wavelength, a dispersive-wave wavelength
or a soliton order) within a search range, and the editor writes the recipe's
``[pins]`` entry; the value solprop solves for is shown beside it, and "Roots" asks
for every root in the range, so another can be chosen than the lowest.
"""

from collections.abc import Mapping, Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QStackedWidget,
    QToolButton,
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
    roots_wanted : Signal
        Emitted when the user asks for every root.
    """

    changed = Signal(object)
    refused = Signal(str)
    roots_wanted = Signal()

    def __init__(  # noqa: PLR0915 - one field per target key
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
        self.find_button = QToolButton()
        self.find_button.setText("Roots")
        self.find_button.setToolTip(
            "Find every value in the search range that meets the target."
        )
        self.find_button.clicked.connect(self.roots_wanted)
        self.solved = QLabel("")
        self.solved.setObjectName(f"pin-solved-{name}")
        self.solved.setWordWrap(True)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self.search_label, 0, 0)
        grid.addWidget(self.search, 0, 1, 1, 2)
        grid.addWidget(self.root, 1, 1)
        grid.addWidget(self.find_button, 1, 2)
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
        for widget in (self.search_label, self.search, self.root, self.find_button):
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
        for widget in (self.search_label, self.search, self.root, self.find_button):
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

    def set_roots(self, found: Mapping[str, object], unit: str | None) -> None:
        """Offer every root solprop found, or say what the range reaches.

        Parameters
        ----------
        found : Mapping
            The kernel's ``pin_roots`` reply: ``roots``, ``quantity`` and
            ``attainable``.
        unit : str or None
            The keyword's unit.
        """
        roots: list[float] = list(found.get("roots") or [])  # pyright: ignore[reportArgumentType]
        self._updating = True
        try:
            while self.root.count() > len(_ROOTS):
                self.root.removeItem(self.root.count() - 1)
            for index, root in enumerate(roots):
                self.root.addItem(f"root {index}: {root:.6g} {unit or ''}", index)
        finally:
            self._updating = False
        if roots:
            self.solved.setText(f"{len(roots)} root(s) in the search range.")
        else:
            reach = found.get("attainable")
            self.solved.setText(
                f"No root: the {found.get('quantity')} reaches {reach} in this range."
                if reach
                else "No root, and the target is not reached in this range."
            )
