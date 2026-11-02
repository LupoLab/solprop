# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A quantity as a field shows it: a number in a chosen unit.

A recipe holds a quantity as a string in any unit of its keyword's dimension
(``"125 um"``), or as a bare number in the keyword's own unit. A field shows the
number in the unit picked beside it, and writes back ``"<number> <unit>"``, so the
unit the person chose is the one the recipe keeps. Every conversion is solprop's
quantity parser and writer, exact to the double.
"""

import re

from solprop.recipe.errors import RecipeError
from solprop.recipe.quantity import Unit, format_quantity, parse_quantity

__all__ = ["number_in", "split", "unit_of", "write"]

_SPLIT = re.compile(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*")


def split(text: str) -> tuple[str, str]:
    """Split a quantity's text into its number and its unit.

    Parameters
    ----------
    text : str
        The text, as typed or as a recipe holds it.

    Returns
    -------
    tuple of str
        The number's text and the unit's (empty when none is written); both empty
        when the text does not start with a number.

    Examples
    --------
    >>> split("125 um"), split(" 2e-3 "), split("um")
    (('125', 'um'), ('2e-3', ''), ('', ''))
    """
    match = _SPLIT.fullmatch(text)
    if match is None:
        return "", ""
    return match.group(1), match.group(2)


def unit_of(value: object, choices: tuple[str, ...], fallback: str) -> str:
    """Return the unit a field should show a value in.

    Parameters
    ----------
    value : object
        The value as the recipe holds it.
    choices : tuple of str
        The units the field offers.
    fallback : str
        The unit for a value written in none of them.

    Returns
    -------
    str
        The value's own unit where the field offers it, else `fallback`.

    Examples
    --------
    >>> unit_of("0.125 mm", ("um", "mm"), "um"), unit_of(1e-4, ("um",), "um")
    ('mm', 'um')
    """
    if isinstance(value, str):
        unit = split(value)[1]
        if unit in choices:
            return unit
    return fallback


def number_in(value: object, keyword_unit: str, unit: str) -> str:
    """Return a value's number in a unit, as the field shows it.

    Parameters
    ----------
    value : object
        A quantity string or a bare number in the keyword's unit.
    keyword_unit : str
        The keyword's unit.
    unit : str
        The unit to show it in, of the same dimension.

    Returns
    -------
    str
        The fewest digits that read back exactly.

    Raises
    ------
    RecipeError
        If the value is not a quantity of the keyword's dimension.

    Examples
    --------
    >>> number_in("125 um", "m", "mm"), number_in(1.2, "bar", "mbar")
    ('0.125', '1200')
    """
    keyword = Unit(keyword_unit)
    exact = parse_quantity(value, keyword)
    return split(format_quantity(exact, keyword, unit))[0]


def write(text: str, keyword_unit: str, unit: str) -> str:
    """Return what was typed into a field as the quantity string a recipe holds.

    Parameters
    ----------
    text : str
        The number typed, or a whole quantity in any unit of the dimension.
    keyword_unit : str
        The keyword's unit.
    unit : str
        The unit picked beside the field, used for a bare number.

    Returns
    -------
    str
        ``"<number> <unit>"`` in the unit typed, or the one picked.

    Raises
    ------
    RecipeError
        If the text is not a quantity of the keyword's dimension, in solprop's
        words.

    Examples
    --------
    >>> write("125", "m", "um"), write("0.1 mm", "m", "um")
    ('125 um', '0.1 mm')
    """
    number, typed_unit = split(text)
    if not number:
        msg = f"{text.strip()!r} is not a number"
        raise RecipeError(msg)
    quantity = f"{number} {typed_unit or unit}"
    parse_quantity(quantity, Unit(keyword_unit))  # refuses a wrong dimension
    return quantity
