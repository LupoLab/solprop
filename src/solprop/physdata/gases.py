# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The gas species solprop knows about.

:class:`Gas` names every built-in gas, and every material quantity of it comes from
optmat (SEP 3): the linear law, the equation of state and transport, the Kerr
coefficient, the Raman response and the ionisation rate all key off it. :func:`as_gas`
coerces a name, and :func:`is_molecule` and :func:`is_air` classify one. A model variant
is not a separate gas; it is chosen with a
:class:`~solprop.physdata.selection.MaterialSelection`.
"""

import math
import sys
from collections.abc import Sequence
from enum import StrEnum

from solprop.exceptions import MaterialError

__all__ = ["Gas", "as_gas", "is_air", "is_molecule"]


class Gas(StrEnum):
    """A gas species, spelled as its chemical formula.

    The members are :class:`str` subclasses, so ``Gas.He == "He"`` and a plain string
    may be used anywhere a gas is expected; :func:`as_gas` performs the conversion and
    raises :class:`~solprop.exceptions.MaterialError` for anything unknown.

    Examples
    --------
    >>> from solprop.physdata import Gas
    >>> Gas.N2O.value
    'N2O'
    >>> Gas("Ar") is Gas.Ar
    True
    """

    Air = "Air"
    He = "He"
    Ne = "Ne"
    Ar = "Ar"
    Kr = "Kr"
    Xe = "Xe"
    N2 = "N2"
    H2 = "H2"
    D2 = "D2"
    O2 = "O2"
    CH4 = "CH4"
    SF6 = "SF6"
    N2O = "N2O"


def as_gas(gas: Gas | str) -> Gas:
    """Coerce a gas name to a :class:`Gas` member.

    Parameters
    ----------
    gas : Gas or str
        A :class:`Gas` member or its exact name, e.g. ``"He"``. The lookup is
        case-sensitive: an explicit error beats a silent coercion.

    Returns
    -------
    Gas
        The matching member.

    Raises
    ------
    MaterialError
        If `gas` is not one of the known species.

    Examples
    --------
    >>> from solprop.physdata import as_gas
    >>> as_gas("Kr")
    <Gas.Kr: 'Kr'>
    """
    try:
        return Gas(gas)
    except ValueError as exc:
        known = ", ".join(member.value for member in Gas)
        raise MaterialError(
            f"unknown gas {gas!r}; solprop knows {known}", parameter="gas"
        ) from exc


#: The species whose particles are single atoms. Everything else in :class:`Gas` is
#: molecular, including ``Air``, which is a mixture of diatomic molecules.
_MONATOMIC = frozenset({Gas.He, Gas.Ne, Gas.Ar, Gas.Kr, Gas.Xe})


def is_molecule(gas: Gas | str) -> bool:
    """Return whether the species is molecular rather than a single atom.

    ``Air`` counts as molecular: it is a mixture of diatomic molecules, not a
    monatomic gas. The predicate is chemical only; which ionisation rate and which
    Raman response a gas has is optmat's (:mod:`solprop.nonlinear.optmat_rates`,
    :mod:`solprop.nonlinear.optmat_raman`).

    Parameters
    ----------
    gas : Gas or str
        A :class:`Gas` member or its exact name.

    Returns
    -------
    bool
        True for a molecular species, False for a noble gas.

    Raises
    ------
    MaterialError
        If `gas` is not one of the known species.

    Examples
    --------
    >>> from solprop.physdata import is_molecule
    >>> is_molecule("He"), is_molecule("N2"), is_molecule("Air")
    (False, True, True)
    """
    return as_gas(gas) not in _MONATOMIC


def is_air(material: Gas | str) -> bool:
    """Return whether a built-in gas's name means air.

    Unlike :func:`as_gas`, which is case-sensitive, this accepts the spelling the
    interface takes for air: ``"air"`` in any case, with surrounding whitespace
    ignored. Any other name, known or not, is simply not air; nothing is refused.

    Parameters
    ----------
    material : Gas or str
        The name given to the interface.

    Returns
    -------
    bool
        ``True`` for ``Gas.Air`` and for ``"air"`` in any case.

    Examples
    --------
    >>> from solprop.physdata import Gas, is_air
    >>> is_air(Gas.Air), is_air(" AIR "), is_air("Ar")
    (True, True, False)
    """
    if isinstance(material, Gas):
        return material is Gas.Air
    return material.strip().lower() == "air"


def _fractions(values: Sequence[float]) -> tuple[float, ...]:
    """Validate mole fractions without silently renormalising them.

    The one rule for a mixture's fill and a gas-flow composition, so that both accept
    the same sets.
    """
    result = tuple(float(v) for v in values)
    if not result or any(not math.isfinite(v) or v < 0 for v in result):
        raise MaterialError("fractions must be finite, non-negative and non-empty")
    # A supplied distribution can incur one rounding per entry and its sum.
    if abs(math.fsum(result) - 1.0) > 2 * len(result) * sys.float_info.epsilon:
        raise MaterialError(f"fractions must sum to one, got {math.fsum(result):.17g}")
    return result
