# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Editors of a registered type's table: one object, or a list of them (design 12.1).

A recipe names an object by a ``type`` key and its keywords
(``lattice = {type = "TubeLattice", wall_thickness = "180 nm", ...}``). The kernel
describes every type a route's keywords can name, with the same metadata a route
keyword has (``parameters(type)``), so each type's form is a column of the same rows a
page has: a picker of the types the keyword allows, and below it the chosen type's
keywords, the basic ones shown and the advanced ones behind "more". A list (the pulses,
a waveguide's modes) is a column of such editors with buttons to add and remove one.
"""

from collections.abc import Callable, Mapping, Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.model.recipe import MISSING
from solprop_gui.registry.pages import type_label

__all__ = ["Catalog", "ObjectField", "ObjectListField", "type_title"]

Catalog = Mapping[str, Sequence[Mapping[str, object]]]
"""Each registered type's keyword metadata, by its name, as the kernel describes it."""

RowFactory = Callable[[Mapping[str, object]], QWidget]
"""Makes a keyword's row (a :class:`~solprop_gui.ui.fields.ParameterRow`)."""


def type_title(name: str) -> str:
    """Return a type's name as the picker shows it.

    Parameters
    ----------
    name : str
        The type, as a recipe names it.

    Returns
    -------
    str
        The name with a class method's dot read as a space.

    Examples
    --------
    >>> type_title("PressureGradient.two_point"), type_title("RectMode")
    ('PressureGradient two_point', 'RectMode')
    """
    return name.replace(".", " ")


class _TypeForm(QWidget):
    """The rows of one type's keywords, and what they hold.

    Parameters
    ----------
    name : str
        The type.
    keywords : Sequence of Mapping
        Its keywords' metadata.
    make_row : callable
        Makes a keyword's row.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    changed : Signal
        Emitted when a keyword is changed.
    """

    changed = Signal()

    def __init__(
        self,
        name: str,
        keywords: Sequence[Mapping[str, object]],
        make_row: RowFactory,
        parent: QWidget | None = None,
    ) -> None:
        """Make a row per keyword, the advanced ones hidden until asked for."""
        super().__init__(parent)
        self.name = name
        self.values: dict[str, object] = {}
        self.rows: dict[str, QWidget] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        for meta in keywords:
            if not any(form.get("recipe", True) for form in meta.get("forms", [])):  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
                continue
            row = make_row({**meta, "label": type_label(name, str(meta["name"]))})
            row.edited.connect(self._edited)  # pyright: ignore[reportAttributeAccessIssue]
            self.rows[str(meta["name"])] = row
            layout.addWidget(row)
        self.more = QToolButton()
        self.more.setObjectName(f"more-{name}")
        self.more.setCheckable(True)
        self.more.setAutoRaise(True)
        self.more.toggled.connect(self._levels)
        layout.addWidget(self.more)
        self._levels(False)

    def _advanced(self, name: str) -> bool:
        """Return whether a keyword is shown only with "more".

        Parameters
        ----------
        name : str
            The keyword.

        Returns
        -------
        bool
            Whether it is advanced, not required and not set.
        """
        row = self.rows[name]
        meta = row.meta  # pyright: ignore[reportAttributeAccessIssue]
        return (
            meta.get("level") == "advanced"
            and not meta.get("required")
            and name not in self.values
        )

    def _levels(self, more: bool) -> None:
        """Show the basic rows, and the rest when asked.

        Parameters
        ----------
        more : bool
            Whether to show every row.
        """
        hidden = [name for name in self.rows if self._advanced(name)]
        for name, row in self.rows.items():
            row.setVisible(more or name not in hidden)
        self.more.setVisible(bool(hidden))
        self.more.setText("fewer" if more else f"more ({len(hidden)})")

    def _edited(self, name: str, _table: str, value: object) -> None:
        """Hold a keyword's new value, or drop it when unset.

        Parameters
        ----------
        name : str
            The keyword.
        _table : str
            The row's table, unused.
        value : object
            Its value, or :data:`~solprop_gui.model.recipe.MISSING`.
        """
        if value is MISSING:
            self.values.pop(name, None)
        else:
            self.values[name] = value
        self.changed.emit()

    def set_values(self, values: Mapping[str, object]) -> None:
        """Show a table's keywords, without announcing them.

        Parameters
        ----------
        values : Mapping
            The keywords the table sets, its ``type`` left out.
        """
        self.values = {k: v for k, v in values.items() if k in self.rows}
        for name, row in self.rows.items():
            row.set_value(self.values.get(name, MISSING))  # pyright: ignore[reportAttributeAccessIssue]
        self._levels(self.more.isChecked())

    def table(self) -> dict[str, object]:
        """Return the table the rows make.

        Returns
        -------
        dict
            ``type`` and every keyword set.
        """
        return {"type": self.name, **self.values}


class ObjectField(QWidget):
    """One registered type's table: a picker of the types, and the chosen one's rows.

    Parameters
    ----------
    types : Sequence of str
        The types the keyword allows, as the catalog has them.
    catalog : Catalog
        Every type's keywords.
    make_row : callable
        Makes a keyword's row.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    changed : Signal
        Emitted with the table, as a recipe writes it.
    """

    changed = Signal(object)

    def __init__(
        self,
        types: Sequence[str],
        catalog: Catalog,
        make_row: RowFactory,
        parent: QWidget | None = None,
    ) -> None:
        """Make the picker, and the first type's form: the others as they are picked."""
        super().__init__(parent)
        self.types = [name for name in types if name in catalog]
        self._updating = False
        self._catalog = catalog
        self._make_row = make_row
        self.picker = QComboBox()
        self.picker.setObjectName("object-type")
        self.forms = QStackedWidget()
        self.by_name: dict[str, _TypeForm] = {}
        for name in self.types:
            self.picker.addItem(type_title(name), name)
        if self.types:
            self.form(self.types[0])
        self.picker.setVisible(len(self.types) > 1)
        self.picker.currentIndexChanged.connect(self._picked)
        self.header = QHBoxLayout()
        self.header.setContentsMargins(0, 0, 0, 0)
        self.header.addWidget(self.picker, 1)
        frame = QFrame()
        frame.setObjectName("object-frame")
        frame.setFrameShape(QFrame.Shape.NoFrame)
        frame.setStyleSheet(
            "#object-frame { border-left: 2px solid palette(mid); padding-left: 6px; }"
        )
        inner = QVBoxLayout(frame)
        inner.setContentsMargins(6, 0, 0, 0)
        inner.addWidget(self.forms)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(self.header)
        layout.addWidget(frame)

    def fits(self, value: object) -> bool:
        """Return whether a value is a table of a type this field offers.

        Parameters
        ----------
        value : object
            The value.

        Returns
        -------
        bool
            Whether it is a mapping whose ``type`` is one of :attr:`types`.
        """
        return isinstance(value, Mapping) and value.get("type") in self.types

    def form(self, name: str) -> _TypeForm:
        """Return a type's form, made the first time it is wanted.

        A keyword may allow a dozen types, each with its rows: making them all at
        once made a sheet slow to open.

        Parameters
        ----------
        name : str
            The type.

        Returns
        -------
        _TypeForm
            Its form.
        """
        found = self.by_name.get(name)
        if found is None:
            found = _TypeForm(name, self._catalog[name], self._make_row)
            found.changed.connect(self._changed)
            self.by_name[name] = found
            self.forms.addWidget(found)
        return found

    def current(self) -> _TypeForm:
        """Return the form of the type picked.

        Returns
        -------
        _TypeForm
            The form.
        """
        return self.form(str(self.picker.currentData()))

    def set_value(self, value: object) -> None:
        """Show a table, or the first type's empty form, without announcing it.

        Parameters
        ----------
        value : object
            A table of one of :attr:`types`, or anything else for an empty form.
        """
        self._updating = True
        try:
            if isinstance(value, Mapping) and self.fits(value):
                name = str(value["type"])
                values = {k: v for k, v in value.items() if k != "type"}
            else:
                name, values = str(self.picker.currentData()), {}
            self.picker.setCurrentIndex(self.picker.findData(name))
            form = self.form(name)
            self.forms.setCurrentWidget(form)
            form.set_values(values)
        finally:
            self._updating = False

    def table(self) -> dict[str, object]:
        """Return the table the field holds.

        Returns
        -------
        dict
            ``type`` and the keywords set.
        """
        return self.current().table()

    def _picked(self, index: int) -> None:
        """Show the type picked, and announce its table.

        Parameters
        ----------
        index : int
            The type's item.
        """
        name = self.picker.itemData(max(index, 0))
        if name is not None:
            self.forms.setCurrentWidget(self.form(str(name)))
        self._changed()

    def _changed(self) -> None:
        """Announce the table."""
        if not self._updating:
            self.changed.emit(self.table())


class ObjectListField(QWidget):
    """A list of registered types' tables, each in its own :class:`ObjectField`.

    Parameters
    ----------
    types : Sequence of str
        The types an item may be.
    catalog : Catalog
        Every type's keywords.
    make_row : callable
        Makes a keyword's row.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    changed : Signal
        Emitted with the list, or ``None`` for an empty one; a list loaded as one
        table and still of one item is written back as that table.
    """

    changed = Signal(object)

    def __init__(
        self,
        types: Sequence[str],
        catalog: Catalog,
        make_row: RowFactory,
        parent: QWidget | None = None,
    ) -> None:
        """Make the empty list and its add button."""
        super().__init__(parent)
        self.types = [name for name in types if name in catalog]
        self.catalog = catalog
        self.make_row = make_row
        self.items: list[ObjectField] = []
        self.single = False
        self._updating = False
        self.column = QVBoxLayout()
        self.column.setContentsMargins(0, 0, 0, 0)
        self.column.setSpacing(4)
        self.add_button = QToolButton()
        self.add_button.setObjectName("object-add")
        self.add_button.setText("+ add")
        self.add_button.setAutoRaise(True)
        self.add_button.clicked.connect(self.add_item)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(self.column)
        layout.addWidget(self.add_button)

    def fits(self, value: object) -> bool:
        """Return whether a value is a table, or a list of tables, of these types.

        Parameters
        ----------
        value : object
            The value.

        Returns
        -------
        bool
            Whether every table names one of :attr:`types`.
        """
        tables = [value] if isinstance(value, Mapping) else value
        return (
            isinstance(tables, list)
            and bool(tables)
            and all(
                isinstance(item, Mapping) and item.get("type") in self.types
                for item in tables
            )
        )

    def _item(self, value: object) -> ObjectField:
        """Append an item showing a table.

        Parameters
        ----------
        value : object
            The table, or anything else for the first type's empty form.

        Returns
        -------
        ObjectField
            The item.
        """
        field = ObjectField(self.types, self.catalog, self.make_row)
        field.set_value(value)
        field.changed.connect(self._changed)
        remove = QToolButton()
        remove.setObjectName("object-remove")
        remove.setText("−")  # noqa: RUF001 - a minus sign
        remove.setToolTip("Remove this one.")
        remove.setAutoRaise(True)
        if not field.picker.isVisibleTo(field):
            field.header.addStretch(1)
        field.header.addWidget(remove)
        remove.clicked.connect(lambda: self.remove_item(field))
        self.column.addWidget(field)
        self.items.append(field)
        return field

    def set_value(self, value: object) -> None:
        """Show a list of tables (or one table), without announcing it.

        Parameters
        ----------
        value : object
            The value; anything that does not fit shows an empty list.
        """
        self._updating = True
        try:
            while self.items:
                self._drop(self.items[-1])
            self.single = isinstance(value, Mapping)
            if self.fits(value):
                for table in [value] if isinstance(value, Mapping) else value:  # pyright: ignore[reportGeneralTypeIssues]
                    self._item(table)
        finally:
            self._updating = False

    def add_item(self) -> None:
        """Add an item of the first type, and announce the list."""
        self._item(None)
        self._changed()

    def _drop(self, field: ObjectField) -> None:
        """Remove an item without announcing it.

        Parameters
        ----------
        field : ObjectField
            The item.
        """
        self.items.remove(field)
        field.setParent(None)
        field.deleteLater()

    def remove_item(self, field: ObjectField) -> None:
        """Remove an item, and announce the list.

        Parameters
        ----------
        field : ObjectField
            The item.
        """
        self._drop(field)
        self._changed()

    def value(self) -> object:
        """Return what the list holds, as a recipe writes it.

        Returns
        -------
        object
            The list of tables; one table where one was loaded and one is left;
            ``None`` for none.
        """
        tables = [field.table() for field in self.items]
        if not tables:
            return None
        if self.single and len(tables) == 1:
            return tables[0]
        return tables

    def _changed(self, *_args: object) -> None:
        """Announce the list.

        Parameters
        ----------
        *_args : object
            What the signal that calls it carries, unused.
        """
        if not self._updating:
            self.changed.emit(self.value())
