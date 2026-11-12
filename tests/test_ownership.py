# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Verify which arrays solprop freezes in place and which it copies."""

import numpy as np
import pytest

from solprop._ownership import frozen_copy, hold


def test_a_frozen_copy_leaves_the_caller_s_array_writeable():
    # A contiguous array of the requested dtype is what ``ascontiguousarray`` hands back
    # unchanged, so clearing the flag on its result froze the caller's own array.
    mine = np.arange(6.0).reshape(2, 3)
    frozen = frozen_copy(mine, np.float64)
    assert mine.flags.writeable
    assert not frozen.flags.writeable
    assert frozen.flags.c_contiguous
    assert not np.shares_memory(frozen, mine)
    np.testing.assert_array_equal(frozen, mine)
    mine[0, 0] = -1.0
    assert frozen[0, 0] == 0.0


def test_a_held_array_and_a_view_of_it_are_taken_as_they_stand():
    held = hold(np.arange(4.0))
    assert frozen_copy(held) is held
    assert frozen_copy(held, np.float64) is held
    view = held[1:]
    assert frozen_copy(view) is view


def test_a_caller_s_read_only_array_is_copied_all_the_same():
    # Its owner can make it writeable again, and a held array must not change.
    mine = np.arange(3.0)
    mine.setflags(write=False)
    frozen = frozen_copy(mine)
    assert frozen is not mine
    mine.setflags(write=True)
    mine[0] = 9.0
    assert frozen[0] == 0.0
    assert frozen_copy(frozen) is frozen


def test_a_read_only_view_of_a_writeable_array_is_copied():
    # Its own flag is clear, but writing the array beneath would change it.
    mine = np.arange(4.0)
    view = mine[:]
    view.setflags(write=False)
    frozen = frozen_copy(view)
    assert frozen is not view
    assert not np.shares_memory(frozen, mine)
    assert mine.flags.writeable


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        (np.arange(8.0)[::2], None),
        (np.asfortranarray(np.ones((2, 3))), None),
        (np.arange(3), np.float64),
        ([1.0, 2.0], np.complex128),
    ],
)
def test_a_frozen_copy_is_contiguous_and_of_the_dtype_asked_for(values, dtype):
    frozen = frozen_copy(values, dtype)
    expected = np.asarray(values, dtype=dtype)
    assert frozen.flags.c_contiguous
    assert not frozen.flags.writeable
    assert frozen.dtype == expected.dtype
    np.testing.assert_array_equal(frozen, expected)
    if isinstance(values, np.ndarray):
        assert values.flags.writeable


def test_a_read_only_array_over_a_foreign_buffer_is_copied():
    # A base that is not an array may still be written; its flags cannot be read.
    buffer = bytearray(16)
    foreign = np.frombuffer(buffer, dtype=np.float64)
    foreign.setflags(write=False)
    frozen = frozen_copy(foreign)
    assert frozen is not foreign
    assert not frozen.flags.writeable
