# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lease Accelerate's calling-thread mode without entering a numerical runtime."""

from __future__ import annotations

import ctypes
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from solprop import _process
from solprop.backends.julia._ownership import require_main_thread
from solprop.exceptions import BackendError

__all__ = ["AccelerateHold", "acquire_accelerate"]

_LIBRARY = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
_AUTOMATIC = 0
_SINGLE = 1


@dataclass(frozen=True)
class _AccelerateLibrary:
    """Adapt exact Apple enum restoration to the existing host lease manager."""

    filepath: str
    owner: object
    getter: Callable[[], int]
    setter: Callable[[int], int]
    thread: int

    def _check(self) -> None:
        """Refuse a thread that cannot own or restore this calling-thread setting."""
        require_main_thread()
        if threading.get_ident() != self.thread:
            raise BackendError("Accelerate limits belong to their acquiring thread")

    @property
    def num_threads(self) -> int:
        """Return the private lease token, an Apple enum rather than a count.

        Returns
        -------
        int
            Zero for automatic mode or one for single mode.
        """
        self._check()
        mode = self.getter()
        if mode not in (_AUTOMATIC, _SINGLE):
            raise BackendError(f"Accelerate returned an unknown threading mode: {mode}")
        return mode

    def set_num_threads(self, mode: int) -> None:
        """Apply only a single-mode request or an exact saved enum restoration.

        Parameters
        ----------
        mode : int
            One for every live request, or the saved zero/one enum on final release.
        """
        self._check()
        if mode not in (_AUTOMATIC, _SINGLE):
            raise BackendError("Accelerate lease requests must use single mode")
        if self.setter(mode) != 0:
            raise BackendError("Accelerate refused its calling-thread mode")
        if self.num_threads != mode:
            raise BackendError("Accelerate did not retain its calling-thread mode")


def _library() -> _AccelerateLibrary | None:
    """Bind the documented mode controls without importing a numerical runtime.

    Returns
    -------
    _AccelerateLibrary or None
        Typed C functions and their owner, or absent controls on this platform.
    """
    if sys.platform != "darwin":
        return None
    try:
        owner = ctypes.CDLL(_LIBRARY)
        getter = ctypes.CFUNCTYPE(ctypes.c_uint)(("BLASGetThreading", owner))
        setter = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint)(
            ("BLASSetThreading", owner)
        )
    except OSError, AttributeError:
        return None
    return _AccelerateLibrary(
        _LIBRARY,
        owner,
        cast(Callable[[], int], getter),
        cast(Callable[[int], int], setter),
        threading.get_ident(),
    )


@dataclass
class AccelerateHold:
    """Keep Accelerate single-threaded for one main-thread preparation's lifetime.

    Attributes
    ----------
    requested : int
        Positive requested BLAS width. The effective Accelerate mode is always
        single-threaded and may be narrower than this request.

    Notes
    -----
    Apple exposes a calling-thread toggle, not an exact arbitrary thread count.
    The shared lease manager holds an internal enum token: zero means automatic
    mode, not zero threads. This controller must not enter generic count metadata
    or receive ordinary requested widths. Every live request here is exactly one;
    the final release restores the original enum, including automatic mode.

    No setting applies to Julia's internal worker threads. The current engine's
    matrix maps use scalar kernels, and its ``mul!`` calls execute FFT plans.
    Admitting a worker-thread BLAS call would require its own calling-thread scope.

    References
    ----------
    .. [1] Apple, ``BLASSetThreading``: thread-local mode and return status.
       https://developer.apple.com/documentation/accelerate/blassetthreading(_:)
    .. [2] Apple, ``BLAS_THREADING``: automatic and single-threaded models.
       https://developer.apple.com/documentation/accelerate/blas_threading
    """

    requested: int
    _library: _AccelerateLibrary
    _lease: _process._BlasLease | None = None

    def close(self) -> None:
        """Restore the remaining single-mode hold or the original enum, idempotently.

        A failed restoration retains this lease in the common manager. Keep the
        owner and retry after resolving the failure; it never widens another live
        owner's limit. Call only from the acquiring main thread.
        """
        self._library._check()
        if self._lease is not None:
            self._lease.restore_original_limits()

    def metadata(self) -> dict[str, object]:
        """Report the actual calling-thread mode without inventing an automatic count.

        Returns
        -------
        dict
            Requested width, current mode, one effective thread in single mode
            or ``None`` in automatic mode, library and calling-thread scope.
        """
        mode = self._library.num_threads
        return {
            "accelerate_library": self._library.filepath,
            "accelerate_threads_requested": self.requested,
            "accelerate_threading_mode": "single" if mode == _SINGLE else "automatic",
            "accelerate_threads": 1 if mode == _SINGLE else None,
            "accelerate_thread_scope": "calling-thread",
        }


def acquire_accelerate(requested: int) -> AccelerateHold | None:
    """Acquire a shared single-mode hold after Julia verified its Accelerate provider.

    Parameters
    ----------
    requested : int
        Positive requested width already resolved against the process budget.
        Accelerate uses single mode even when this value is larger than one.

    Returns
    -------
    AccelerateHold or None
        Acquired Python-only lifetime owner, or ``None`` outside macOS or when the
        documented macOS 15 threading symbols are absent. A runtime reporting an
        Accelerate provider must refuse a missing hold rather than continue unbounded.

    Raises
    ------
    ValueError
        If the requested width is not a positive integer.
    BackendError
        For foreign-thread entry, an unknown mode, or a failed mode change.
    """
    require_main_thread()
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError("the Accelerate requested width must be a positive integer")
    library = _library()
    if library is None:
        return None
    # The pool only sees the single-mode token. Its saved original token is the
    # exact 0/1 enum; no automatic-mode CPU-count guess is needed for restoration.
    # Allocate the Python owner before changing native state. Publishing the
    # returned lease into its existing slot needs no new callback or owner object.
    hold = AccelerateHold(requested, library)
    hold._lease = _process.acquire_host_blas((library,), _SINGLE)
    return hold
