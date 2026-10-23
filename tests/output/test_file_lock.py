# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise the portable advisory lock on this platform and Windows' path anywhere."""

import os
import sys
import types

import pytest

from solprop.output import _file_lock
from solprop.output._file_lock import lock, unlock


@pytest.fixture
def descriptors(tmp_path):
    """Open two descriptors on one lock file, as two processes would."""
    path = tmp_path / "scan.h5.lock"
    first = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    second = os.open(path, os.O_RDWR)
    yield first, second
    os.close(first)
    os.close(second)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shared locks")
def test_shared_locks_coexist_and_exclude_a_writer(descriptors):
    first, second = descriptors
    lock(first, shared=True)
    lock(second, shared=True, blocking=False)
    unlock(second)
    with pytest.raises(BlockingIOError):
        lock(second, blocking=False)
    unlock(first)
    lock(second, blocking=False)
    unlock(second)


def test_an_exclusive_lock_refuses_a_second_holder_until_released(descriptors):
    first, second = descriptors
    lock(first)
    with pytest.raises(BlockingIOError):
        lock(second, shared=True, blocking=False)
    unlock(first)
    lock(second, shared=True, blocking=False)
    unlock(second)


class _FakeMsvcrt(types.SimpleNamespace):
    """Record byte-range calls as msvcrt.locking would see them, held by one handle."""

    LK_UNLCK = 0
    LK_NBLCK = 2

    def __init__(self, busy: int) -> None:
        super().__init__(calls=[], busy=busy)

    def locking(self, descriptor: int, mode: int, size: int) -> None:
        assert size == 1
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 0
        self.calls.append(mode)
        if mode == self.LK_NBLCK and self.busy > 0:
            self.busy -= 1
            raise OSError(36, "Resource deadlock avoided")


@pytest.mark.parametrize("shared", [True, False])
def test_windows_locks_the_first_byte_and_retries_until_free(
    tmp_path, monkeypatch, shared
):
    fake = _FakeMsvcrt(busy=2)
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(_file_lock, "_RETRY_SECONDS", 0.0)
    descriptor = os.open(tmp_path / "lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        lock(descriptor, shared=shared)
        unlock(descriptor)
    finally:
        os.close(descriptor)
    assert fake.calls == [fake.LK_NBLCK] * 3 + [fake.LK_UNLCK]


def test_windows_refuses_at_once_when_not_blocking(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "msvcrt", _FakeMsvcrt(busy=1))
    monkeypatch.setattr(sys, "platform", "win32")
    descriptor = os.open(tmp_path / "lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        with pytest.raises(BlockingIOError):
            lock(descriptor, blocking=False)
    finally:
        os.close(descriptor)
