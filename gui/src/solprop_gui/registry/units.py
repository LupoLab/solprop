# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The units a quantity field offers (design 12.2).

solprop's metadata gives each keyword its unit and the unit it is displayed in; which
other units a person may want to type in is presentation, and lives here. Every unit
listed is one solprop's quantity parser reads, so a value written in it converts
exactly; a keyword whose dimension is not listed offers its display unit alone.
"""

from typing import Final

from solprop.recipe.quantity import Unit

__all__ = ["CHOICES", "pretty", "unit_choices"]

CHOICES: Final = (
    ("nm", "um", "mm", "cm", "m"),
    ("as", "fs", "ps", "ns", "s"),
    ("nJ", "uJ", "mJ", "J"),
    ("W", "kW", "MW", "GW", "TW"),
    ("mbar", "bar", "kPa", "Pa"),
    ("Hz", "GHz", "THz", "PHz"),
    ("W/cm^2", "W/m^2"),
    ("K",),
)
"""The units offered, by dimension, smallest first."""


def unit_choices(unit: str, display: str | None = None) -> tuple[str, ...]:
    """Return the units a keyword's field offers.

    Parameters
    ----------
    unit : str
        The keyword's own unit.
    display : str or None, optional
        The unit solprop displays it in.

    Returns
    -------
    tuple of str
        The units of its dimension from :data:`CHOICES`, the display unit included;
        the display unit (or the keyword's) alone for a dimension not listed.

    Examples
    --------
    >>> unit_choices("m", "um")
    ('nm', 'um', 'mm', 'cm', 'm')
    >>> unit_choices("s^2/m", "fs^2/mm")
    ('fs^2/mm',)
    """
    dimension = Unit(unit).dimension
    shown = display or unit
    for group in CHOICES:
        if Unit(group[0]).dimension == dimension:
            return group if shown in group else (*group, shown)
    return (shown,)


def pretty(unit: str) -> str:
    """Return a unit as it is shown: ``u`` as ``µ``, powers as superscripts.

    Parameters
    ----------
    unit : str
        The unit, in ASCII.

    Returns
    -------
    str
        The unit for a label.

    Examples
    --------
    >>> pretty("uJ"), pretty("W/cm^2"), pretty("fs^2/mm")
    ('µJ', 'W/cm²', 'fs²/mm')
    """
    text = unit.replace("^2", "²").replace("^3", "³").replace("^-1", "⁻¹")
    return "µ" + text[1:] if text.startswith("u") else text
