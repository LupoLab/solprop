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

__all__ = ["CHOICES", "pretty", "si", "unit_choices"]

CHOICES: Final = (
    ("nm", "um", "mm", "cm", "m"),
    ("as", "fs", "ps", "ns", "s"),
    ("pJ", "nJ", "uJ", "mJ", "J"),
    ("W", "kW", "MW", "GW", "TW"),
    ("mbar", "bar", "kPa", "Pa"),
    ("Hz", "GHz", "THz", "PHz"),
    ("W/cm^2", "W/m^2"),
    ("K",),
    # The Taylor coefficients of a dispersion, order by order (the GNLSE's betas).
    ("1/mm", "1/m"),
    ("fs/mm", "ps/m", "s/m"),
    ("fs^2/mm", "ps^2/km", "ps^2/m", "s^2/m"),
    ("fs^3/mm", "ps^3/km", "ps^3/m", "s^3/m"),
    ("fs^4/mm", "ps^4/km", "ps^4/m", "s^4/m"),
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
    >>> unit_choices("s^2/m", "ps^2/km")
    ('fs^2/mm', 'ps^2/km', 'ps^2/m', 's^2/m')
    >>> unit_choices("m^3", "m^3")
    ('m^3',)
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


_PREFIXES: Final = (
    (1e12, "T"),
    (1e9, "G"),
    (1e6, "M"),
    (1e3, "k"),
    (1.0, ""),
    (1e-3, "m"),
    (1e-6, "µ"),
    (1e-9, "n"),
    (1e-12, "p"),
    (1e-15, "f"),
)
"""The SI prefixes a summary number may take, largest first."""


def si(value: float, unit: str, digits: int = 4) -> str:
    """Return a number with the SI prefix that keeps it between 1 and 1000.

    Parameters
    ----------
    value : float
        The number, in `unit`.
    unit : str
        Its unit, unprefixed (``"J"``, ``"W"``).
    digits : int, optional
        The significant digits shown.

    Returns
    -------
    str
        ``"<number> <prefix><unit>"``.

    Examples
    --------
    >>> si(1.936e-11, "J"), si(96.79, "W"), si(9.4e9, "W"), si(0.0, "J")
    ('19.36 pJ', '96.79 W', '9.4 GW', '0 J')
    """
    size = abs(value)
    scale, prefix = next(
        ((s, p) for s, p in _PREFIXES if size >= s),
        (1.0, "") if size == 0 else _PREFIXES[-1],
    )
    return f"{value / scale:.{digits}g} {prefix}{unit}"
