# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Read and trim a recipe's text, for the service and the window alike.

These helpers import only the light parts of :mod:`solprop.recipe` (its document and
nothing that builds), so the window's process may use them as freely as the kernel.
"""

import re
from collections.abc import Mapping
from typing import Final

from solprop.recipe.document import dump_text, parse_text

__all__ = ["BEAM_ROUTES", "data_files", "split", "without_table"]

BEAM_ROUTES: Final = frozenset({"radial", "free", "planar"})
"""The routes of a beam in free space, whose files solprop reads by its spatial readers
alone: the spectral and power readers of a guided run's field refuse them."""

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


def without_table(text: str, table: str) -> str:
    r"""Return a recipe's text with one of its tables left out.

    Parameters
    ----------
    text : str
        The recipe.
    table : str
        The table, as ``"scan"``.

    Returns
    -------
    str
        The text, canonical TOML; the same text where the recipe has no such table.

    Examples
    --------
    >>> text = '[recipe]\nformat = 1\nroute = "capillary"\n\n[build]\nradius = 1e-4\n'
    >>> left = without_table(text + '\n[scan]\nname = "s"\nradius = [1e-4]\n', "scan")
    >>> "scan" in parse_text(left), parse_text(left)["build"]["radius"]
    (False, 0.0001)
    >>> without_table(text, "scan") == text
    True
    """
    document = parse_text(text)
    if table not in document:
        return text
    tables = {name: values for name, values in document.items() if name != table}
    return dump_text(tables)


def data_files(text: str) -> list[str]:
    r"""Return the data files a recipe reads, as it names them.

    A data file is written in a recipe as a table of its ``file`` (relative to the
    recipe's folder) and its ``sha256``; a run on another machine needs a copy of each
    beside its recipe (design 13.4).

    Parameters
    ----------
    text : str
        The recipe.

    Returns
    -------
    list of str
        Each file once, in the order the recipe gives them.

    Examples
    --------
    >>> text = '[recipe]\nformat = 1\nroute = "gnlse"\n\n[build]\n'
    >>> text += 'pulses = [{ type = "load_spectrum", path = '
    >>> text += '{ file = "spectrum.txt", sha256 = "%s" } }]\n' % ("0" * 64)
    >>> data_files(text)
    ['spectrum.txt']
    """
    found: list[str] = []

    def _walk(value: object) -> None:
        """Add the references inside a value, depth first."""
        if isinstance(value, Mapping):
            file = value.get("file")
            if "sha256" in value and isinstance(file, str) and file not in found:
                found.append(file)
            for item in value.values():
                _walk(item)
        elif isinstance(value, list):
            for item in value:
                _walk(item)

    _walk(dict(parse_text(text)))
    return found
