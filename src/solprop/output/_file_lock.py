# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Hold kernel-owned advisory locks on POSIX and on Windows through one interface.

POSIX takes :func:`fcntl.flock`, shared or exclusive. Windows has no shared byte-range
lock in the standard library, so there every lock is exclusive on the descriptor's
first byte through :func:`msvcrt.locking`: readers then serialise with each other as
well as with the writer, which is safe and only less concurrent. Either way the
operating system owns the lock, so a killed process releases it.
"""

import os
import sys
import time

# The interval at which a blocking Windows lock retries: msvcrt's own blocking mode
# gives up after ten one-second attempts, and a reader must wait for a writer as long
# as the write session lasts.
_RETRY_SECONDS = 0.05


def lock(descriptor: int, *, shared: bool = False, blocking: bool = True) -> None:
    """Acquire the advisory lock on an open descriptor.

    Parameters
    ----------
    descriptor : int
        Open file descriptor of the lock file.
    shared : bool
        Whether a shared (reader) lock suffices; exclusive on Windows regardless.
    blocking : bool
        Whether to wait for the lock rather than fail at once.

    Raises
    ------
    BlockingIOError
        If ``blocking`` is false and another process holds a conflicting lock.
    """
    if sys.platform == "win32":
        _windows_lock(descriptor, blocking=blocking)
        return
    import fcntl  # noqa: PLC0415 - POSIX only

    mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    fcntl.flock(descriptor, mode if blocking else mode | fcntl.LOCK_NB)


def unlock(descriptor: int) -> None:
    """Release the advisory lock taken by :func:`lock`.

    Parameters
    ----------
    descriptor : int
        Open file descriptor of the lock file.
    """
    if sys.platform == "win32":
        import msvcrt  # noqa: PLC0415 - Windows only

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)  # pyright: ignore[reportAttributeAccessIssue]
        return
    import fcntl  # noqa: PLC0415 - POSIX only

    fcntl.flock(descriptor, fcntl.LOCK_UN)


def _windows_lock(descriptor: int, *, blocking: bool) -> None:
    """Lock the first byte exclusively, retrying until it is free when blocking.

    Parameters
    ----------
    descriptor : int
        Open file descriptor of the lock file.
    blocking : bool
        Whether to wait for the lock rather than fail at once.

    Raises
    ------
    BlockingIOError
        If ``blocking`` is false and another process holds the lock.
    """
    import msvcrt  # noqa: PLC0415 - Windows only

    while True:
        # locking() acts from the current position, and a byte beyond the end of an
        # empty file can still be locked on Windows.
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)  # pyright: ignore[reportAttributeAccessIssue]
        except OSError as error:
            if not blocking:
                raise BlockingIOError(*error.args) from error
            time.sleep(_RETRY_SECONDS)
        else:
            return
