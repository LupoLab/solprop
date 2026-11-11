# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Share host BLAS limits after Julia has loaded its numerical libraries."""

from __future__ import annotations

import atexit
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solprop import _process
from solprop.backends.julia._ownership import require_main_thread

__all__ = ["BlasHold", "acquire_blas", "prepare_blas_cleanup"]


@dataclass(frozen=True)
class BlasHold:
    """Retain shared process limits for one Julia preparation's lifetime.

    Attributes
    ----------
    requested : int
        Requested positive width, within the process's CPU budget. A simultaneous
        narrower owner may reduce the actual width further.
    """

    requested: int
    # threadpoolctl has no stubs. Its untyped library controllers remain inside
    # this discovery/metadata boundary and the common process lease manager.
    _libraries: tuple[Any, ...]
    _restore: Callable[[], None]

    def close(self) -> None:
        """Release this owner's requests on the main thread, idempotently."""
        require_main_thread()
        try:
            self._restore()
        except BaseException:
            # The common lease deliberately retains a failed library's request.
            # Keep its Python-only owner available after registry callback cleanup
            # drops the Julia preparation; a later acquisition can retry safely.
            _FAILED_RELEASES[id(self)] = self
            raise
        else:
            _FAILED_RELEASES.pop(id(self), None)

    def metadata(self) -> dict[str, object]:
        """Read actual shared widths without changing any numerical library.

        Returns
        -------
        dict
            JSON-compatible requested width and each library's current width.
        """
        require_main_thread()
        return {
            "blas_threads_requested": self.requested,
            "blas_libraries": [
                {
                    "library": Path(library.filepath).name,
                    "threads": library.num_threads,
                }
                for library in self._libraries
            ],
        }


_FAILED_RELEASES: dict[int, BlasHold] = {}


def _release_pending() -> None:
    """Retry every failed Python-only lease release and report the first failure."""
    failure = None
    for hold in tuple(_FAILED_RELEASES.values()):
        try:
            hold.close()
        except BaseException as error:
            if failure is None:
                failure = error
    if failure is not None:
        raise failure


@dataclass
class _Cleanup:
    """Remember registration without changing runtime or library state."""

    registered: bool = False


_CLEANUP = _Cleanup()


def prepare_blas_cleanup() -> None:
    """Install Python-only lease cleanup before the numerical runtime starts.

    Registration is idempotent and runs on the main thread. Its ordering lets it
    retry a release that the later runtime-owner shutdown hook reports as failed.
    """
    require_main_thread()
    if not _CLEANUP.registered:
        atexit.register(_release_pending)
        _CLEANUP.registered = True


def acquire_blas() -> BlasHold:
    """Discover loaded host BLAS libraries and acquire shared limits for Julia.

    Call after runtime initialization: an existing construction scope's cached
    discovery predates Julia's BLAS. A fresh discovery joins the same process
    leases, so releasing either owner cannot restore another owner's active pool.
    The already-loaded native backend's private BLAS retains its own fitted width.

    Returns
    -------
    BlasHold
        Eagerly acquired lifetime hold, released with immutable preparation.

    Raises
    ------
    BackendError
        If called outside the process's main Python thread.
    ValueError
        If the BLAS thread environment contains an invalid count.
    """
    require_main_thread()
    prepare_blas_cleanup()
    _release_pending()
    import threadpoolctl  # noqa: PLC0415 - discover after Julia loads its libraries

    requested = _process.thread_count("SOLPROP_BLAS_THREADS")
    threads = _process.thread_budget().explicit(1 if requested is None else requested)
    native = sys.modules.get("solprop_native")
    build_info = getattr(native, "build_info", None)
    own = "" if build_info is None else os.path.realpath(build_info().blas_library)
    libraries = tuple(
        library
        for library in threadpoolctl.ThreadpoolController()
        .select(user_api="blas")
        .lib_controllers
        if os.path.realpath(library.filepath) != own
    )
    lease = _process.acquire_host_blas(libraries, threads)
    try:
        return BlasHold(threads, libraries, lease.restore_original_limits)
    except BaseException:
        lease.restore_original_limits()
        raise
