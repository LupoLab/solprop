# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Retain Julia wrappers until Python's main thread can release them safely."""

from __future__ import annotations

import os
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import Any

from solprop.backends.julia._errors import detached
from solprop.backends.session import SessionError
from solprop.exceptions import BackendError

__all__ = ["JuliaHandle", "OwnerRegistry", "require_main_thread"]


def require_main_thread() -> None:
    """Refuse Julia entry from a Python worker before touching its runtime.

    Raises
    ------
    BackendError
        If this is not Python's main thread in the current process.
    """
    if threading.current_thread() is not threading.main_thread():
        raise BackendError(
            "julia operations require Python's main thread in each process; "
            "use a spawned worker process, or request cancellation with StopToken"
        )


@dataclass(eq=False, repr=False)
class _Entry:
    """Retain a wrapper, its borrowed storage and Python-only cleanup together."""

    # JuliaCall's dynamic wrapper is confined to the owner-only bridge boundary.
    value: Any
    owners: tuple[object, ...]
    on_release: Callable[[], None] | None


class OwnerRegistry:
    """Own Julia wrappers and their borrowed buffers until main-thread release.

    The process runtime retains this registry. Handles carry integer identities;
    their finalizers only enqueue those identities, even on foreign threads.
    """

    def __init__(self) -> None:
        """Create an empty registry on the runtime's owning main thread."""
        require_main_thread()
        self._pid = os.getpid()
        self._entries: dict[int, _Entry] = {}
        self._pending: SimpleQueue[int] = SimpleQueue()
        self._next = 0
        self._closed = False
        self._releasing = False

    def _check(self) -> None:
        """Refuse foreign, forked, closed or recursive owner access."""
        require_main_thread()
        if self._pid != os.getpid():
            raise BackendError(
                "julia owners cannot run after fork; use a fresh process"
            )
        if self._closed:
            raise SessionError("the julia runtime owners are closed")
        if self._releasing:
            raise SessionError("julia entry during owner release is not allowed")

    def retain(
        self,
        value: Any,
        owners: tuple[object, ...] = (),
        *,
        on_release: Callable[[], None] | None = None,
    ) -> JuliaHandle:
        """Retain a wrapper and the Python buffers its Julia value borrows.

        Parameters
        ----------
        value : object
            A Julia wrapper created inside the owner-only bridge boundary.
        owners : tuple of object, optional
            Owners that must outlive Julia's access to borrowed storage.
        on_release : callable or None, optional
            Python-only cleanup before releasing borrowed storage, such as detaching
            a stop listener. It must not enter Julia. A foreign cleanup failure
            is reported after the wrapper is released; retryable Python resources
            must retain their own failed-cleanup ownership independently.

        Returns
        -------
        JuliaHandle
            An identity whose destruction never enters Julia.
        """
        self._check()
        self._next += 1
        key = self._next
        self._entries[key] = _Entry(value, owners, on_release)
        return JuliaHandle(self, key)

    def _borrow(self, key: int) -> Any:
        """Resolve an identity only inside the owning thread's bridge helper.

        Parameters
        ----------
        key : int
            Live owner identity.

        Returns
        -------
        object
            The borrowed Julia wrapper.
        """
        self._check()
        try:
            return self._entries[key].value
        except KeyError:
            raise SessionError("the julia handle is closed") from None

    def _release(self, key: int) -> None:
        """Release one identity on main, refusing recursive entry.

        Parameters
        ----------
        key : int
            Owner identity; an already released one is a no-op.
        """
        require_main_thread()
        if self._closed:
            return
        self._check()
        self._releasing = True
        try:
            self._release_entry(key)
        finally:
            self._releasing = False

    def drain(self) -> None:
        """Release abandoned handles on the main thread before another operation."""
        self._check()
        self._releasing = True
        try:
            while True:
                try:
                    key = self._pending.get_nowait()
                except Empty:
                    return
                self._release_entry(key)
        finally:
            self._releasing = False

    def _release_entry(self, key: int) -> None:
        """Detach and release an entry without leaking it through a failure.

        Parameters
        ----------
        key : int
            Owner identity, while the registry's release guard is held.
        """
        entry = self._entries.pop(key, None)
        failure = None
        try:
            if entry is not None and entry.on_release is not None:
                entry.on_release()
        except BaseException as error:
            failure = detached(error, "the julia owner cleanup failed")
        finally:
            # An interrupted detach must not retain a wrapper in an escaping frame.
            del entry
        if failure is not None:
            raise failure from None

    def close(self) -> None:
        """Release all wrappers on main before Julia's shutdown hook; idempotent."""
        require_main_thread()
        if self._pid != os.getpid():
            # The parent owns these pointers. A fork child cannot destroy them.
            return
        if not self._closed:
            if self._releasing:
                raise SessionError("julia entry during owner release is not allowed")
            self._closed = True
            failure = None
            for key in tuple(self._entries):
                try:
                    self._release_entry(key)
                except BaseException as error:
                    # Finish all owners before Julia's next shutdown hook, even
                    # when one cleanup is interrupted. Preserve the first failure.
                    if failure is None:
                        failure = error
            if failure is not None:
                raise failure from None


class JuliaHandle:
    """Hold a Julia identity without owning a wrapper on the calling thread.

    Parameters
    ----------
    registry : OwnerRegistry
        The process-owned registry that retains the wrapper and its borrowed buffers.
    key : int
        A monotonically assigned identity; never reused during the runtime's lifetime.
    """

    def __init__(self, registry: OwnerRegistry, key: int) -> None:
        """Attach an enqueue-only finalizer with no reference to the Julia wrapper."""
        self._registry = registry
        self._key = key
        self._finalizer = weakref.finalize(self, registry._pending.put, key)
        # The runtime's explicit hook releases all owners before Julia shuts down.
        # weakref's independent exit hook has no reliable ordering with JuliaCall.
        self._finalizer.atexit = False

    def borrow(self) -> Any:
        """Borrow the wrapper within an owner-only bridge call.

        Returns
        -------
        object
            The opaque Julia wrapper. Never return it through a public session API
            or retain it in a frame that can escape through a Python exception.

        Raises
        ------
        SessionError
            If this handle or the runtime's registry is closed.
        """
        return self._registry._borrow(self._key)

    def close(self) -> None:
        """Release the wrapper on main after its engine readers finish; idempotent."""
        self._registry._release(self._key)
        self._finalizer.detach()
