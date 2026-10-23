# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The gas species solprop knows about.

Every built-in gas is a name here, and every material quantity of it comes from optmat
(SEP 3): the linear law, the equation of state and transport, the Kerr coefficient,
the Raman response and the ionisation rate all key off :class:`Gas`. A model variant is
not a separate gas; it is chosen with a
:class:`~solprop.physdata.selection.MaterialSelection`.
"""

from enum import StrEnum

from solprop.exceptions import MaterialError

__all__ = ["Gas", "as_gas", "is_molecule"]


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
        raise MaterialError(f"unknown gas {gas!r}; solprop knows {known}") from exc


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
