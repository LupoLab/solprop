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

__all__ = ["borrow", "frozen_copy", "hold", "hold_view", "is_held"]

_HELD: weakref.WeakValueDictionary[int, np.ndarray] = weakref.WeakValueDictionary()
"""The arrays solprop froze as its own, by ``id``.

An array is in it because solprop made it -- a copy :func:`frozen_copy` took, or one
just built (:func:`hold`) -- never because of its flags, which the owner of an array,
or a writeable buffer beneath it, can change. Only arrays that own their memory are
entered, and the views :func:`borrow` names. Entries leave with their arrays.
"""


def hold(values: np.ndarray) -> np.ndarray:
    """Freeze an array solprop has just made, in place, and mark it as held.

    Only for an array no one else refers to: a copy, a quotient or a fresh build. It
    costs no copy, and :func:`frozen_copy` takes the array, or a view of it, as it
    stands. A view is frozen but not held, since the memory beneath it could still be
    written: hold the array that owns the memory instead.

    - Returns: the same array, read-only.
    """
    values.setflags(write=False)
    if values.base is None:
        _HELD[id(values)] = values
    return values


def hold_view(values: np.ndarray) -> np.ndarray:
    """Freeze a view of memory solprop has just made, and hold the array that owns it.

    For a fresh build handed out as a view, such as a spectrum given its leading axes
    by indexing: :func:`hold` would freeze the view alone, and it would not be held.
    Only for a view whose memory no one else refers to.

    - Returns: the same array, read-only, and held when the memory beneath it is an
      array's.
    """
    owner = values
    while isinstance(owner.base, np.ndarray):
        owner = owner.base
    if owner.base is None:
        hold(owner)
    values.setflags(write=False)
    return values


def borrow(values: np.ndarray) -> np.ndarray:
    """Freeze a view of memory solprop does not own, and treat it as held.

    The one exception to :func:`hold`'s rule, for a caller's array solprop reads
    within a call and does not keep: the caller's memory stays writeable, and nothing
    copies the view. Whoever borrows promises the memory is not written while the
    view lives.

    - `values`: a view, created for the borrowing.
    - Returns: the same view, read-only and held.
    """
    values.setflags(write=False)
    _HELD[id(values)] = values
    return values


def is_held(values: object) -> bool:
    """Return whether an array is read-only down to memory solprop holds.

    The array, and every array beneath it, must be read-only, and the array that owns
    the memory must be one :func:`hold` marked (or the chain must reach a view
    :func:`borrow` named). A read-only view of a held array cannot be made writeable,
    so it is held too; a read-only view of a writeable array is not, and nor is a
    read-only array that is not held, however it was frozen, since whoever owns it
    can make it writeable again.

    - Returns: True for a held array, or a read-only view of one, while it is
      read-only.
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

    - `dtype`: the element type of the result; None keeps the input's.
    - Returns: the same values and shape, C-contiguous, not writeable and held: `values`
      itself when it is already all of that, with `dtype`; otherwise a new copy.
    """
    if (
        isinstance(values, np.ndarray)
        and (dtype is None or values.dtype == dtype)
        and values.flags.c_contiguous
        and is_held(values)
    ):
        return values
    return hold(np.array(values, dtype=dtype, order="C", copy=True))
