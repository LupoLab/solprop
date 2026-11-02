# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Number and quantity fields: a number box, a unit picker, and a pair of them.

A number box holds text, so that an unset keyword can show solprop's default in grey;
the arrow keys and the wheel step it by a tenth of its leading digit. A quantity field
adds the units of its dimension; changing the unit converts the number shown, exactly,
through solprop's parser. A range is two numbers sharing one unit.
"""

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QWidget
from solprop.recipe.errors import RecipeError

from solprop_gui.model.quantities import number_in, split, unit_of, write
from solprop_gui.registry.units import pretty, unit_choices

__all__ = ["NumberEdit", "QuantityField", "RangeField", "step"]


def step(text: str, steps: int, *, integer: bool = False) -> str | None:
    """Return a number stepped by a tenth of its leading digit's place, or by one.

    Parameters
    ----------
    text : str
        The number's text.
    steps : int
        How many steps, signed.
    integer : bool, optional
        Whether it is an integer, stepped by one.

    Returns
    -------
    str or None
        The stepped number; ``None`` when the text is not a number.

    Examples
    --------
    >>> step("125", 1), step("0.5", -1), step("3", 2, integer=True)
    ('135', '0.49', '5')
    """
    try:
        value = float(text)
    except ValueError:
        return None
    if integer:
        return str(int(value) + steps)
    size = 10.0 ** (math.floor(math.log10(abs(value))) - 1) if value else 0.1
    return f"{value + steps * size:.6g}"


class NumberEdit(QLineEdit):
    """A line for one number, stepped with the arrow keys and the wheel.

    Parameters
    ----------
    integer : bool, optional
        Whether it holds an integer.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.
    """

    def __init__(self, *, integer: bool = False, parent: QWidget | None = None) -> None:
        """Make the line, right-aligned like a spin box."""
        super().__init__(parent)
        self.integer = integer
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.setMinimumWidth(90)

    def _step(self, steps: int) -> None:
        """Step the number shown, or the default, and finish the edit.

        Parameters
        ----------
        steps : int
            How many steps, signed.
        """
        stepped = step(
            self.text() or self.placeholderText(), steps, integer=self.integer
        )
        if stepped is not None:
            self.setText(stepped)
            self.editingFinished.emit()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt's name
        """Step on the up and down arrows.

        Parameters
        ----------
        event : PySide6.QtGui.QKeyEvent
            The key.
        """
        if event.key() == Qt.Key.Key_Up:
            self._step(1)
        elif event.key() == Qt.Key.Key_Down:
            self._step(-1)
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt's name
        """Step on the wheel while the line has the focus.

        Parameters
        ----------
        event : PySide6.QtGui.QWheelEvent
            The wheel.
        """
        if not self.hasFocus():
            event.ignore()
            return
        self._step(1 if event.angleDelta().y() > 0 else -1)
        event.accept()


class _UnitBox(QComboBox):
    """The units of one dimension, shown with µ and superscripts.

    Parameters
    ----------
    choices : tuple of str
        The units, in ASCII.
    """

    def __init__(self, choices: tuple[str, ...]) -> None:
        """List the units; a single one cannot be changed."""
        super().__init__()
        for unit in choices:
            self.addItem(pretty(unit), unit)
        self.setEnabled(len(choices) > 1)

    def unit(self) -> str:
        """Return the unit picked.

        Returns
        -------
        str
            Its ASCII name.
        """
        return str(self.currentData())

    def set_unit(self, unit: str) -> None:
        """Pick a unit without announcing it.

        Parameters
        ----------
        unit : str
            Its ASCII name; an unknown one picks the first.
        """
        self.blockSignals(True)
        self.setCurrentIndex(max(self.findData(unit), 0))
        self.blockSignals(False)


class QuantityField(QWidget):
    """A number with its unit, or a plain number.

    Parameters
    ----------
    keyword_unit : str or None
        The keyword's unit; ``None`` for a number without one.
    display : str or None, optional
        The unit solprop displays it in, picked at first.
    integer : bool, optional
        Whether it holds an integer.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    changed : Signal
        Emitted with the value typed: a quantity string, a number, or ``None`` for an
        empty field.
    refused : Signal
        Emitted with solprop's reason when what was typed cannot be read.
    """

    changed = Signal(object)
    refused = Signal(str)

    def __init__(
        self,
        keyword_unit: str | None,
        display: str | None = None,
        *,
        integer: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        """Make the number box and, for a quantity, its units."""
        super().__init__(parent)
        dimensionless = keyword_unit in (None, "1")
        self.keyword_unit = None if dimensionless else keyword_unit
        self.integer = integer
        self.number = NumberEdit(integer=integer)
        self.number.editingFinished.connect(self._finished)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.number, 1)
        self.units: _UnitBox | None = None
        self._value: object = None
        self._default: object = None
        if self.keyword_unit is not None:
            choices = unit_choices(self.keyword_unit, display)
            self.units = _UnitBox(choices)
            self.units.set_unit(display or self.keyword_unit)
            self.units.currentIndexChanged.connect(self._unit_changed)
            layout.addWidget(self.units)

    def unit(self) -> str | None:
        """Return the unit picked.

        Returns
        -------
        str or None
            The unit; ``None`` for a plain number.
        """
        return None if self.units is None else self.units.unit()

    def _shown(self, value: object) -> str:
        """Return a value's number in the unit picked.

        Parameters
        ----------
        value : object
            The value.

        Returns
        -------
        str
            The number's text; the value as written where it cannot be converted.
        """
        if value is None:
            return ""
        unit = self.unit()
        if unit is None or self.keyword_unit is None:
            return str(value)
        try:
            return number_in(value, self.keyword_unit, unit)
        except RecipeError:
            return str(value)

    def set_value(
        self, value: object, default: object = None, *, keep_unit: bool = False
    ) -> None:
        """Show a value, or the default in grey when there is none.

        Parameters
        ----------
        value : object
            The recipe's value, or ``None`` when it sets none.
        default : object, optional
            The default solprop applies, shown while the value is ``None``.
        keep_unit : bool, optional
            Show it in the unit picked, rather than in the value's own.
        """
        self._value = value
        self._default = default
        if self.units is not None and value is not None and not keep_unit:
            self.units.set_unit(unit_of(value, self._choices(), self.units.unit()))
        self.number.setText(self._shown(value))
        self.number.setPlaceholderText(self._shown(default))

    def _choices(self) -> tuple[str, ...]:
        """Return the units offered.

        Returns
        -------
        tuple of str
            Their ASCII names.
        """
        units = self.units
        if units is None:
            return ()
        return tuple(str(units.itemData(i)) for i in range(units.count()))

    def read(self) -> object:
        """Return what the box holds, as the recipe will hold it.

        Returns
        -------
        object
            A quantity string, a number, or ``None`` for an empty box.

        Raises
        ------
        RecipeError
            If it cannot be read, in solprop's words.
        """
        text = self.number.text().strip()
        if not text:
            return None
        if self.keyword_unit is not None:
            return write(text, self.keyword_unit, self.unit() or self.keyword_unit)
        number, unit = split(text)
        if not number or unit:
            msg = f"{text!r} is not a number"
            raise RecipeError(msg)
        value = float(number)
        if self.integer:
            if not value.is_integer():
                msg = f"{text!r} is not a whole number"
                raise RecipeError(msg)
            return int(value)
        return int(number) if number.lstrip("+-").isdigit() else value

    def _finished(self) -> None:
        """Read the box as the person leaves it."""
        try:
            value = self.read()
        except RecipeError as error:
            self.refused.emit(str(error))
            return
        if isinstance(value, str) and self.units is not None:
            # A whole quantity typed in another unit: show it in that unit.
            number, unit = split(value)
            if unit in self._choices():
                self.units.set_unit(unit)
                self.number.setText(number)
        if value != self._value:
            self._value = value
            self.changed.emit(value)

    def _unit_changed(self) -> None:
        """Show the same value in the unit just picked."""
        self.number.setText(self._shown(self._value))
        self.number.setPlaceholderText(self._shown(self._default))
        if self._value is not None and self.keyword_unit is not None:
            value = write(self.number.text(), self.keyword_unit, self.unit() or "")
            self._value = value
            self.changed.emit(value)


class RangeField(QWidget):
    """Two numbers of one quantity, from and to, sharing one unit.

    Parameters
    ----------
    keyword_unit : str
        The keyword's unit.
    display : str or None, optional
        The unit solprop displays it in.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    changed : Signal
        Emitted with the pair as the recipe will hold it, or ``None`` when both
        boxes are empty.
    refused : Signal
        Emitted with the reason what was typed cannot be read.
    """

    changed = Signal(object)
    refused = Signal(str)

    def __init__(
        self,
        keyword_unit: str,
        display: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Make the two boxes and the units."""
        super().__init__(parent)
        self.keyword_unit = keyword_unit
        self.low = QuantityField(keyword_unit, display)
        self.high = QuantityField(keyword_unit, display)
        # One unit for both: the second box's picker is the shared one.
        assert self.low.units is not None and self.high.units is not None
        self.low.units.hide()
        self.units = self.high.units
        self.units.currentIndexChanged.connect(self._unit_changed)
        for field in (self.low, self.high):
            field.number.editingFinished.disconnect()
            field.number.editingFinished.connect(self._finished)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.low, 1)
        layout.addWidget(QLabel("to"))
        layout.addWidget(self.high, 1)
        self._value: object = None

    def set_value(self, value: object, default: object = None) -> None:
        """Show a pair, or the default pair in grey.

        Parameters
        ----------
        value : object
            The recipe's pair, or ``None``.
        default : object, optional
            The default pair solprop applies.
        """
        self._value = value
        pair = value if isinstance(value, list | tuple) and len(value) == 2 else None
        fallback = (
            default if isinstance(default, list | tuple) and len(default) == 2 else None
        )
        if pair is not None:
            self.units.set_unit(
                unit_of(pair[0], self.low._choices(), self.units.unit())
            )
        self.low.units.set_unit(self.units.unit())  # pyright: ignore[reportOptionalMemberAccess]
        self.low.set_value(
            None if pair is None else pair[0],
            None if fallback is None else fallback[0],
            keep_unit=True,
        )
        self.high.set_value(
            None if pair is None else pair[1],
            None if fallback is None else fallback[1],
            keep_unit=True,
        )

    def read(self) -> object:
        """Return the pair the boxes hold.

        Returns
        -------
        object
            Two quantity strings in the unit picked, or ``None`` when both are
            empty.

        Raises
        ------
        RecipeError
            If one box is empty and the other not, or one cannot be read.
        """
        unit = self.units.unit()
        texts = [field.number.text().strip() for field in (self.low, self.high)]
        if not any(texts):
            return None
        if not all(texts):
            msg = "give both ends of the range"
            raise RecipeError(msg)
        return [write(text, self.keyword_unit, unit) for text in texts]

    def _finished(self) -> None:
        """Read the pair as the person leaves either box."""
        try:
            value = self.read()
        except RecipeError as error:
            self.refused.emit(str(error))
            return
        # The halves hold their ends, so that a change of unit converts them.
        ends = value if isinstance(value, list) else [None, None]
        self.low._value, self.high._value = ends[0], ends[1]
        if value != self._value:
            self._value = value
            self.changed.emit(value)

    def _unit_changed(self) -> None:
        """Show both ends in the unit just picked."""
        assert self.low.units is not None
        self.low.units.set_unit(self.units.unit())
        self.low._unit_changed()
        self.high._unit_changed()
        self._value = self.read() if self._value is not None else None
        if self._value is not None:
            self.changed.emit(self._value)
