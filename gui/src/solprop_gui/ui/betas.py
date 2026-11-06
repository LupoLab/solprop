# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Taylor coefficients of a dispersion, one row per order (the GNLSE's ``betas``).

``betas = [0, 0, "-0.01183 ps^2/m"]`` gives beta_n for n = 0, 1, 2, ..., each in its own
unit, ``s^n/m`` (``1/m`` for n = 0). Each order is a quantity field offering the units
of its dimension (fs^2/mm, ps^2/km and the rest), and rows are added or removed at the
end, the highest order.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from solprop.recipe.errors import RecipeError

from solprop_gui.ui.quantity import QuantityField

__all__ = ["BetasField", "order_unit"]

_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def order_unit(order: int) -> tuple[str, str]:
    """Return the unit of one Taylor order, and the unit it is shown in at first.

    Parameters
    ----------
    order : int
        The order, at least 0.

    Returns
    -------
    tuple of str
        The coefficient's SI unit and its usual display unit.

    Examples
    --------
    >>> order_unit(0), order_unit(1), order_unit(2), order_unit(3)
    (('1/m', '1/m'), ('s/m', 'ps/m'), ('s^2/m', 'ps^2/km'), ('s^3/m', 'ps^3/km'))
    """
    if order == 0:
        return "1/m", "1/m"
    if order == 1:
        return "s/m", "ps/m"
    return f"s^{order}/m", f"ps^{order}/km"


def _is_number(text: str) -> bool:
    """Return whether a field's text is a bare number.

    Parameters
    ----------
    text : str
        The text.

    Returns
    -------
    bool
        Whether it reads as a float.

    Examples
    --------
    >>> _is_number("0"), _is_number("-1e-3"), _is_number("1 ps/m")
    (True, True, False)
    """
    try:
        float(text)
    except ValueError:
        return False
    return True


class BetasField(QWidget):
    """A column of Taylor coefficients.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    changed : Signal
        Emitted with the list, or ``None`` for an empty one.
    refused : Signal
        Emitted with the reason a coefficient does not read.
    """

    changed = Signal(object)
    refused = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the empty column and its buttons."""
        super().__init__(parent)
        self.fields: list[QuantityField] = []
        self._updating = False
        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.add_button = QToolButton()
        self.add_button.setText("+ order")
        self.add_button.setAutoRaise(True)
        self.add_button.clicked.connect(self.add_order)
        self.remove_button = QToolButton()
        self.remove_button.setText("− order")  # noqa: RUF001 - a minus sign
        self.remove_button.setAutoRaise(True)
        self.remove_button.clicked.connect(self.remove_order)
        buttons = QHBoxLayout()
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(self.grid)
        layout.addLayout(buttons)

    @staticmethod
    def fits(value: object) -> bool:
        """Return whether a value is a list of coefficients.

        Parameters
        ----------
        value : object
            The value.

        Returns
        -------
        bool
            Whether it is a list of numbers and quantity strings.
        """
        return isinstance(value, list) and all(
            isinstance(item, int | float | str) and not isinstance(item, bool)
            for item in value
        )

    def _order(self, value: object) -> None:
        """Append the next order's row.

        Parameters
        ----------
        value : object
            Its coefficient, or ``None``.
        """
        order = len(self.fields)
        unit, display = order_unit(order)
        field = QuantityField(unit, display)
        field.set_value(value, 0)
        field.changed.connect(self._changed)
        field.refused.connect(self.refused)
        self.grid.addWidget(QLabel(f"β{str(order).translate(_SUBSCRIPTS)}"), order, 0)
        self.grid.addWidget(field, order, 1)
        self.fields.append(field)
        self.remove_button.setEnabled(True)

    def set_value(self, value: object) -> None:
        """Show a list of coefficients, without announcing it.

        Parameters
        ----------
        value : object
            The list; anything else shows none.
        """
        self._updating = True
        try:
            while self.fields:
                self._drop()
            if self.fits(value):
                for item in value:  # pyright: ignore[reportGeneralTypeIssues]
                    self._order(item)
        finally:
            self._updating = False
        self.remove_button.setEnabled(bool(self.fields))

    def _drop(self) -> None:
        """Remove the highest order's row."""
        field = self.fields.pop()
        order = len(self.fields)
        for column in (0, 1):
            item = self.grid.itemAtPosition(order, column)
            widget = None if item is None else item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        del field
        self.remove_button.setEnabled(bool(self.fields))

    def add_order(self) -> None:
        """Add the next order, at zero, and announce the list."""
        self._order(None)
        self._changed()

    def remove_order(self) -> None:
        """Remove the highest order, and announce the list."""
        if self.fields:
            self._drop()
            self._changed()

    def value(self) -> list[object] | None:
        """Return the coefficients as a recipe writes them.

        Returns
        -------
        list or None
            Each order's quantity, an empty field as 0; ``None`` for no orders.

        Raises
        ------
        RecipeError
            If a coefficient does not read.
        """
        if not self.fields:
            return None
        values: list[object] = []
        for field in self.fields:
            value = field.read()
            text = field.number.text().strip()
            # A zero is written bare, as recipes write the orders a dispersion skips.
            zero = value is None or (_is_number(text) and float(text) == 0.0)
            values.append(0 if zero else value)
        return values

    def _changed(self, *_args: object) -> None:
        """Announce the list.

        Parameters
        ----------
        *_args : object
            What the signal that calls it carries, unused.
        """
        if self._updating:
            return
        try:
            value = self.value()
        except RecipeError as error:
            self.refused.emit(str(error))
            return
        self.changed.emit(value)
