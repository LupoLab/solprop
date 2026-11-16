# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep exclusive writer ownership independently of an HDF5 handle's lifetime."""

import json
import os
from pathlib import Path

from solprop._file_lock import lock, unlock
from solprop.exceptions import OutputError

SNAPSHOT_SCHEMA = "solprop.snapshot.v1"
"""The schema a snapshot's ``manifest.json`` declares; writes under one are refused."""


class WriterHeldError(OutputError):
    """Another writer owns the file, so ownership cannot be taken without waiting."""


class WriterLock:
    """Own a persistent sidecar's advisory lock until explicitly released."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._descriptor: int | None = None

    def acquire(self) -> None:
        """Acquire ownership without waiting for another writer."""
        if self._descriptor is not None:
            return
        for parent in self.path.parents:
            manifest = parent / "manifest.json"
            if manifest.is_file():
                try:
                    record = json.loads(manifest.read_text(encoding="utf-8"))
                except ValueError, OSError:
                    continue
                if isinstance(record, dict) and record.get("schema") == SNAPSHOT_SCHEMA:
                    raise OutputError(
                        "immutable snapshot: restore_snapshot before writing"
                    )
        sidecar = self.path.with_name(self.path.name + ".writer.lock")
        try:
            descriptor = os.open(sidecar, os.O_CREAT | os.O_RDWR, 0o600)
        except OSError as error:
            raise OutputError(
                f"cannot open writer ownership for {self.path}: {error}"
            ) from error
        try:
            lock(descriptor, blocking=False)
        except BaseException as error:
            os.close(descriptor)
            if isinstance(error, BlockingIOError):
                raise WriterHeldError(f"another writer owns {self.path}") from error
            if isinstance(error, OSError):
                raise OutputError(
                    f"cannot lock writer ownership for {self.path}: {error}"
                ) from error
            raise
        self._descriptor = descriptor

    def release(self) -> None:
        """Release ownership; repeated calls are harmless."""
        descriptor, self._descriptor = self._descriptor, None
        if descriptor is not None:
            try:
                unlock(descriptor)
            finally:
                os.close(descriptor)
