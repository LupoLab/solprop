# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Know which arrays solprop holds: freeze its own in place, copy everyone else's.

Every array that crosses the seam, and every array a value type keeps, is read-only.
Freezing is done one way: an array solprop has just made is frozen where it stands
and marked as held (:func:`hold`); anything else is copied first
(:func:`frozen_copy`). The mark, not the write flag, says which is which: the owner of
an array can make it writeable again, so a read-only array a caller hands in is
copied all the same, while one solprop already holds is taken as it stands, however
many times it is handed on.

This module imports nothing from the rest of the package, so that any layer may use
it.
"""

import weakref

import numpy as np

__all__ = ["frozen_copy", "hold", "is_held"]

_HELD: weakref.WeakValueDictionary[int, np.ndarray] = weakref.WeakValueDictionary()
"""The arrays solprop froze as its own, by ``id``.

An array is in it because solprop made it -- a copy :func:`frozen_copy` took, or one
just built (:func:`hold`) -- never because of its flags, which the owner of an array,
or a writeable buffer beneath it, can change. Entries leave with their arrays.
"""


def hold(values: np.ndarray) -> np.ndarray:
    """Freeze an array solprop has just made, in place, and mark it as held.

    Only for an array no one else refers to: a copy, a quotient or a fresh build. It
    costs no copy, and :func:`frozen_copy` takes the array, or a view of it, as it
    stands.

    Parameters
    ----------
    values : numpy.ndarray
        The array.

    Returns
    -------
    numpy.ndarray
        The same array, read-only.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop._ownership import hold, is_held
    >>> made = hold(np.arange(3.0))
    >>> bool(made.flags.writeable), is_held(made)
    (False, True)
    """
    values.setflags(write=False)
    _HELD[id(values)] = values
    return values


def is_held(values: object) -> bool:
    """Return whether an array is read-only down to an array :func:`hold` marked.

    A read-only view of a held array cannot be made writeable, so it is held too. A
    read-only array that is not held is not, however it was frozen: whoever owns it
    can make it writeable again.

    Parameters
    ----------
    values : object
        Anything.

    Returns
    -------
    bool
        True for a held array, or a read-only view of one, while it is read-only.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop._ownership import is_held
    >>> mine = np.arange(3.0)
    >>> mine.setflags(write=False)
    >>> is_held(mine)
    False
    """
    array = values
    while isinstance(array, np.ndarray) and not array.flags.writeable:
        if _HELD.get(id(array)) is array:
            return True
        array = array.base
    return False


def frozen_copy(
    values: object, dtype: np.dtype | type[np.generic] | None = None
) -> np.ndarray:
    """Return the values read-only and C-contiguous, as an array solprop holds.

    It copies, so that freezing never reaches an array the caller still holds:
    ``numpy.ascontiguousarray`` returns its argument itself when that is already
    contiguous and of the dtype asked for, so clearing the write flag on what it
    returns would freeze the caller's own array, and a caller's read-only array can be
    made writeable again by its owner. Only an array solprop already holds
    (:func:`is_held`) is returned as it stands. An array just built takes
    :func:`hold` instead, which copies nothing.

    Parameters
    ----------
    values : array_like
        The values to freeze, of any shape.
    dtype : numpy.dtype or type or None, optional
        The element type of the result; None keeps the input's.

    Returns
    -------
    numpy.ndarray
        The same values and shape, C-contiguous, not writeable and held: `values`
        itself when it is already all of that, with `dtype`; otherwise a new copy.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop._ownership import frozen_copy
    >>> mine = np.arange(3.0)
    >>> frozen = frozen_copy(mine)
    >>> bool(mine.flags.writeable), bool(frozen.flags.writeable)
    (True, False)
    >>> frozen_copy(frozen) is frozen
    True
    """
    if (
        isinstance(values, np.ndarray)
        and (dtype is None or values.dtype == dtype)
        and values.flags.c_contiguous
        and is_held(values)
    ):
        return values
    return hold(np.array(values, dtype=dtype, order="C", copy=True))
