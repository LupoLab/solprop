# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Small files replaced whole, so that a reader never sees half of one.

The scan queue's state, the front end's run records and the TDSE consistency cache are
each a small text file that another process may read at any moment. Each is written to a
temporary file beside it, flushed to the disk, and renamed over the old one in a single
step (:func:`write_atomic`); a scan queue also makes the rename itself durable
(:func:`sync_directory`), since its claims must survive a crash of the machine.

Like ``_process``, this module imports nothing from the rest of the package, so any
layer may use it.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Final

__all__ = ["REPLACE_TRIES", "sync_directory", "write_atomic"]

REPLACE_TRIES: Final = 50
"""How often a rename refused by another process's open handle is tried, 20 ms apart."""


def sync_directory(path: Path) -> None:
    """Make the names published in a directory durable (POSIX directory fsync)."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_atomic(path: Path, text: str, *, durable: bool = False) -> None:
    """Replace `path` with `text` in one step, its content on the disk first.

    ``durable=True`` also syncs the directory, so the new name survives a crash.
    Windows refuses to replace a file another process has open (a reader of a run's
    summary), so a refused rename is tried again for a moment rather than leaving the
    file stale.
    """
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        left = REPLACE_TRIES
        while True:
            try:
                temporary.replace(path)
            except PermissionError:
                left -= 1
                if not left:
                    raise
                time.sleep(0.02)
            else:
                break
        if durable:
            sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
