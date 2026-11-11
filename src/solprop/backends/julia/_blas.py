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
from solprop.backends.julia._accelerate import AccelerateHold, acquire_accelerate
from solprop.backends.julia._ownership import require_main_thread
from solprop.exceptions import BackendError

__all__ = ["BlasHold", "CPUBlasInfo", "acquire_blas", "prepare_blas_cleanup"]


@dataclass(frozen=True)
class CPUBlasInfo:
    """Retain verified BLAS forwarding facts without Julia wrappers.

    Attributes
    ----------
    provider : str
        Verified ``accelerate`` or ``openblas`` provider.
    library : str
        Actual primary BLAS/LAPACK library.
    interface : str
        Julia's integer ABI, ``ilp64`` or ``lp64``.
    fallback_library : str
        Retained OpenBLAS library for missing Accelerate symbols, or empty.
    thread_scope : str
        ``calling_thread`` for Accelerate or ``process`` for OpenBLAS.
    reason : str
        Why OpenBLAS was retained, or empty for Accelerate.
    """

    provider: str
    library: str
    interface: str
    fallback_library: str
    thread_scope: str
    reason: str


@dataclass
class _BlasResources:
    """Own both independent process pools and Apple's calling-thread mode."""

    host: _process._BlasLease | None = None
    apple: AccelerateHold | None = None

    def release(self) -> None:
        """Try both restorations, retaining failed resources for the owner's retry."""
        failure = None
        for resource in (self.host, self.apple):
            if resource is None:
                continue
            try:
                if isinstance(resource, AccelerateHold):
                    resource.close()
                else:
                    resource.restore_original_limits()
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    failure.add_note(f"Additional BLAS restoration failed: {error}")
        if failure is not None:
            raise failure


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
    _resources: _BlasResources | None = None
    _provider: CPUBlasInfo | None = None

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
        record: dict[str, object] = {
            "blas_threads_requested": self.requested,
            "blas_libraries": [
                {
                    "library": Path(library.filepath).name,
                    "threads": library.num_threads,
                }
                for library in self._libraries
            ],
        }
        if self._provider is not None:
            provider = self._provider
            record.update(
                blas_provider=provider.provider,
                blas_primary_library=Path(provider.library).name,
                blas_integer_interface=provider.interface,
                blas_fallback_library=Path(provider.fallback_library).name
                if provider.fallback_library
                else "",
                blas_thread_scope=provider.thread_scope,
                blas_provider_reason=provider.reason,
            )
        if self._resources is not None and self._resources.apple is not None:
            record.update(self._resources.apple.metadata())
        return record


_FAILED_RELEASES: dict[int, BlasHold] = {}


def _release_pending() -> None:
    """Retry every failed Python-only lease release and report the first failure."""
    failure = None
    try:
        _process._retry_blas_acquisitions()
    except BaseException as error:
        failure = error
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


def acquire_blas(*, provider: CPUBlasInfo | None = None) -> BlasHold:
    """Discover loaded host BLAS libraries and acquire shared limits for Julia.

    Call after runtime initialization: an existing construction scope's cached
    discovery predates Julia's BLAS. A fresh discovery joins the same process
    leases, so releasing either owner cannot restore another owner's active pool.
    The already-loaded native backend's independent BLAS retains its fitted width.
    Accelerate is shared by all callers on Python's main OS thread, including native
    and NumPy users of that system framework: its lifetime hold selects single mode.
    Julia worker threads are unaffected and currently make no BLAS calls.

    Parameters
    ----------
    provider : CPUBlasInfo or None, optional
        Julia's verified provider facts. An Accelerate provider requires the separate
        documented calling-thread lease; generic library controllers handle counts.

    Returns
    -------
    BlasHold
        Eagerly acquired lifetime hold, released with immutable preparation.

    Raises
    ------
    BackendError
        If called outside the process's main Python thread, or verified Accelerate
        threading controls cannot be acquired.
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
        and (
            provider is None
            or provider.provider != "accelerate"
            or os.path.realpath(library.filepath) != os.path.realpath(provider.library)
        )
    )
    resources = _BlasResources()
    hold = BlasHold(threads, libraries, resources.release, resources, provider)
    try:
        if provider is not None and provider.provider == "accelerate":
            resources.apple = acquire_accelerate(threads)
            if resources.apple is None:
                raise BackendError(
                    "the verified Julia Accelerate provider has no threading control"
                )
        resources.host = _process.acquire_host_blas(libraries, threads)
        return hold
    except BaseException as failure:
        try:
            hold.close()
        except BaseException as cleanup:
            failure.add_note(f"BLAS cleanup failed; retained for retry: {cleanup}")
        raise
