# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Provide silver and aluminium optical constants for waveguide walls.

:func:`silver` and :func:`aluminium` each return optmat's stitched table with the
interpolant's exact derivative, as a :class:`~solprop.physdata.glasses.GlassIndex`, the
common index interface whose historical name covers metals too
(``docs/guide/media/materials.md``).
"""

from solprop.physdata.glasses import GlassIndex, _resolve
from solprop.physdata.selection import ModelChoice

__all__ = ["aluminium", "silver"]


def silver(*, model: ModelChoice | None = None) -> GlassIndex:
    """Return the silver index, with positive imaginary part for loss.

    Parameters
    ----------
    model : str or OptmatModel or None, optional
        ``None`` or ``"recommended"`` is optmat's ``table/Ag``, stitched from
        primary sets, 4.1 nm to 12.4 um, with the interpolant's exact derivative,
        holding its end values outside its range with a warning; an optmat model
        name or object selects another
        (:class:`~solprop.physdata.glasses.OptmatGlass`).

    Returns
    -------
    solprop.physdata.glasses.GlassIndex
        The complex index.

    Raises
    ------
    MaterialError
        If the optmat model is unknown or not silver.

    Examples
    --------
    >>> from solprop.physdata import silver
    >>> silver().source
    'table/Ag'
    """
    return _resolve("Ag", model)


def aluminium(*, model: ModelChoice | None = None) -> GlassIndex:
    """Return the aluminium index, with positive imaginary part for loss.

    Parameters
    ----------
    model : str or OptmatModel or None, optional
        ``None`` or ``"recommended"`` is optmat's ``table/Al``, 0.79 nm to 200 um,
        holding its end values outside its range with a warning; an optmat model
        name or object selects another
        (:class:`~solprop.physdata.glasses.OptmatGlass`).

    Returns
    -------
    solprop.physdata.glasses.GlassIndex
        The complex index.

    Raises
    ------
    MaterialError
        If the optmat model is unknown or not aluminium.
    """
    return _resolve("Al", model)
