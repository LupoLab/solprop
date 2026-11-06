# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Between what a field shows and what a recipe holds.

A keyword's value is shown as text: a quantity string as it is (``125 um``), any other
value as inline TOML (``["1.2 bar", "0 bar"]``, ``{ type = "StatsSpec" }``). What the
user types is read back by the keyword's forms from solprop's metadata: a number with
a unit is checked by solprop's own quantity parser, here in the window's process, so a
wrong dimension is caught on the keystroke; everything else is read as TOML and left
to solprop's check in the kernel.
"""

import math
import tomllib
from collections.abc import Mapping, Sequence

from solprop.recipe.errors import RecipeError
from solprop.recipe.quantity import Unit, parse_quantity

__all__ = ["FieldError", "inline", "read_value"]


class FieldError(ValueError):
    """What a field cannot read, in a sentence to show beside it."""


def inline(value: object) -> str:
    """Write a recipe value as the text a field shows.

    Parameters
    ----------
    value : object
        A value as a recipe holds it.

    Returns
    -------
    str
        A string as it is; anything else as inline TOML.

    Examples
    --------
    >>> inline("125 um"), inline(True), inline(["1.2 bar", 0.0])
    ('125 um', 'true', '["1.2 bar", 0.0]')
    >>> inline({"type": "StatsSpec", "names": ["energy"]})
    '{ type = "StatsSpec", names = ["energy"] }'
    """
    if isinstance(value, str):
        return value
    return _toml(value)


def _toml(value: object) -> str:  # noqa: PLR0911 - one return per kind
    """Write a value as inline TOML.

    Parameters
    ----------
    value : object
        A boolean, number, string, list or table.

    Returns
    -------
    str
        Its inline TOML.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, Mapping):
        items = ", ".join(f"{_key(k)} = {_toml(v)}" for k, v in value.items())
        return "{ " + items + " }" if items else "{}"
    if isinstance(value, Sequence):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    return _toml(str(value))


def _key(key: object) -> str:
    """Write a table key, quoted where it is not bare.

    Parameters
    ----------
    key : object
        The key.

    Returns
    -------
    str
        The key.
    """
    text = str(key)
    if text and all(c.isalnum() or c in "_-" for c in text) and text.isascii():
        return text
    return _toml(text)


def _toml_value(text: str) -> object:
    """Read inline TOML.

    Parameters
    ----------
    text : str
        The text.

    Returns
    -------
    object
        The value.

    Raises
    ------
    FieldError
        If it is not a TOML value.
    """
    try:
        return tomllib.loads(f"value = {text}")["value"]
    except tomllib.TOMLDecodeError as error:
        msg = f"not a value: {error}"
        raise FieldError(msg) from error


def read_value(text: str, forms: Sequence[Mapping[str, object]]) -> object:
    """Read what was typed into a keyword's field.

    Parameters
    ----------
    text : str
        The text, stripped of surrounding space; empty means the keyword is unset.
    forms : Sequence of Mapping
        The keyword's forms, as the kernel describes them: ``kind`` and ``unit``
        at least.

    Returns
    -------
    object
        The value as the recipe will hold it.

    Raises
    ------
    FieldError
        If no form of the keyword reads the text, saying why the first one that
        could have refused it.

    Examples
    --------
    >>> number = [{"kind": "number", "unit": "m"}]
    >>> read_value("125 um", number), read_value("1e-4", number)
    ('125 um', 0.0001)
    >>> read_value("on", [{"kind": "choice", "unit": None}])
    'on'
    """
    text = text.strip()
    kinds = [str(form.get("kind")) for form in forms]
    refusal: str | None = None
    for form in forms:
        kind = str(form.get("kind"))
        unit = form.get("unit")
        if kind == "number":
            try:
                number = _toml_value(text)
            except FieldError:
                number = None
            if isinstance(number, int | float) and not isinstance(number, bool):
                return number
            if unit is not None:
                try:
                    parse_quantity(text, Unit(str(unit)))
                except RecipeError as error:
                    refusal = refusal or str(error)
                    continue
                return text
        elif kind == "integer":
            try:
                number = _toml_value(text)
            except FieldError:
                continue
            if isinstance(number, int) and not isinstance(number, bool):
                return number
    try:
        value = _toml_value(text)
    except FieldError as error:
        if any(kind in ("string", "choice") for kind in kinds):
            return text
        raise FieldError(refusal or str(error)) from error
    return value
