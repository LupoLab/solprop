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

from solprop.recipe.errors import RecipeError
from solprop.recipe.quantity import Unit, format_quantity, parse_quantity
from solprop.service.recipes import split

__all__ = ["number_in", "rounded", "unit_of", "write"]


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


def rounded(value: float, keyword_unit: str, unit: str, digits: int = 6) -> str:
    """Return a solved value as a recipe literal, rounded, in a unit.

    A value solprop solves for (a design's pressure, say) has more digits than
    anyone measures it to; written as a literal it keeps `digits` significant ones.

    Parameters
    ----------
    value : float
        The value in the keyword's unit.
    keyword_unit : str
        The keyword's unit.
    unit : str
        The unit to write it in, of the same dimension.
    digits : int, optional
        The significant digits kept.

    Returns
    -------
    str
        ``"<number> <unit>"``; a bare number for a dimensionless keyword.

    Examples
    --------
    >>> rounded(1.2779114785253451, "bar", "bar"), rounded(1.2317608e-4, "J", "uJ")
    ('1.27791 bar', '123.176 uJ')
    >>> rounded(3.14159265, "1", "1")
    '3.14159'
    """
    if keyword_unit in ("", "1"):
        return f"{value:.{digits}g}"
    number = float(number_in(value, keyword_unit, unit))
    return f"{number:.{digits}g} {unit}"
