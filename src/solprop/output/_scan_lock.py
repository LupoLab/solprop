# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Serialise collected scan readers against the controller's HDF5 write session."""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import h5py

from solprop.output._file_lock import lock, unlock

logger = logging.getLogger(__name__)


def lock_path(path: Path | str) -> Path:
    """Locate the persistent advisory lock beside a collected scan.

    Parameters
    ----------
    path : Path | str
        Queue directory or record filename used by this operation.

    Returns
    -------
    pathlib.Path
        Persistent sidecar filename beside the collected scan.
    """
    path = Path(path)
    return path.with_name(path.name + ".lock")


def _sidecar(path: Path | str) -> int:
    """Open the persistent sidecar, creating it so a reader can find the lock.

    Parameters
    ----------
    path : Path | str
        Queue directory or record filename used by this operation.

    Returns
    -------
    int
        Open file descriptor for the sidecar; the caller closes it.
    """
    return os.open(lock_path(path), os.O_CREAT | os.O_RDWR, 0o600)


@contextmanager
def write_scan(path: Path | str) -> Iterator[None]:
    """Hold the exclusive advisory lock for one controller's write session.

    Ownership is the kernel's, so a killed controller releases the file rather
    than stranding it. HDF5 is not opened here: the caller owns the file for as
    long as it holds this lock.

    Parameters
    ----------
    path : Path | str
        Queue directory or record filename used by this operation.

    Yields
    ------
    None
        Control while the exclusive advisory lock is held.
    """
    descriptor = _sidecar(path)
    try:
        lock(descriptor)
        yield
    finally:
        unlock(descriptor)
        os.close(descriptor)


@contextmanager
def read_scan(path: Path | str) -> Iterator[h5py.File]:
    """Read copied standalone files or wait for an active queue collector.

    The sidecar is created rather than merely honoured when present: the writer
    creates it too, and a reader that skipped the lock because no sidecar existed
    yet would open a live non-SWMR file under an active writer.

    Parameters
    ----------
    path : Path | str
        Queue directory or record filename used by this operation.

    Yields
    ------
    h5py.File
        Open read-only scan while a shared advisory lock is held.
    """
    if not Path(path).is_file():
        # Refused before the sidecar exists, so a mistyped path leaves nothing behind.
        raise FileNotFoundError(f"no scan file at {path}")
    try:
        descriptor = _sidecar(path)
    except PermissionError:
        # A read-only mount or a directory the reader cannot write: no controller
        # can be writing there either, so the unlocked read is safe.
        logger.debug("reading %s without a lock: its directory is not writable", path)
        descriptor = None
    try:
        if descriptor is not None:
            try:
                lock(descriptor, shared=True, blocking=False)
            except BlockingIOError:
                # Blocking is correct, but a silent block looks like a hang to
                # anyone reading a scan that a controller is still writing.
                logger.info("waiting for the writer of %s", path)
                lock(descriptor, shared=True)
        with h5py.File(path, "r") as file:
            yield file
    finally:
        if descriptor is not None:
            unlock(descriptor)
            os.close(descriptor)
