# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep atomic cancellation storage alive through captured foreign callbacks."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from solprop.backends.julia._runtime import JuliaRuntime
from solprop.backends.session import SessionError, StopToken

__all__ = ["StopLease"]


@dataclass(frozen=True, eq=False)
class _Listener:
    """Retain only owners safe to reclaim on a foreign stopping thread."""

    # No Julia wrapper can reach this object: its final reference may be dropped
    # on a previously unknown stopping thread after concurrent session close.
    # Even destruction must not adopt that thread or wait for Julia's GC.
    storage: ctypes.c_uint8
    setter: Callable[[int], None]
    address: int

    def __call__(self) -> None:
        """Store the monotone request without entering Julia or acquiring its locks."""
        self.setter(self.address)


class StopLease:
    """Attach an owned C atomic byte and its borrowed Julia view to a stop token.

    Parameters
    ----------
    runtime : JuliaRuntime
        The process's initialized engine, retained with the generated setter code.
    token : StopToken
        The run's stop token, which may already be set.
    """

    _listener: _Listener | None
    _detach: Callable[[], None] | None
    _flag: Any  # Opaque borrowed-pointer Julia wrapper, owned only by the lease.

    def __init__(self, runtime: JuliaRuntime, token: StopToken) -> None:
        """Allocate the flag and compile its setter before attaching the listener."""
        runtime.check()
        storage = ctypes.c_uint8(0)
        address = ctypes.addressof(storage)
        self._flag = runtime.engine.StopFlag(address)
        setter_address = int(runtime.engine.stop_setter_address())
        # CFUNCTYPE releases the GIL; its target is a bare atomic store, not a
        # Julia @cfunction trampoline. This cannot adopt a thread or wait for GC.
        setter = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(setter_address)
        # The process runtime permanently roots the compiled setter. A listener
        # retains only C/Python owners; a captured tuple can outlive this lease.
        self._listener = _Listener(storage, setter, address)
        self._detach = token.attach(self._listener)

    @property
    def flag(self) -> Any:
        """Return the opaque Julia view while the lease owns its byte.

        Returns
        -------
        object
            Julia's borrowed stop flag. Engine reads must finish before lease closure.

        Raises
        ------
        SessionError
            If the lease was closed.
        """
        if self._listener is None:
            raise SessionError("the julia stop lease is closed")
        return self._flag

    def close(self) -> None:
        """Detach once after engine reads end, retaining captured setter callbacks.

        A session must reject or serialize close while advancement is in flight;
        the borrowed Julia flag itself does not own the byte used by its readers.
        """
        if self._detach is not None:
            self._detach()
            self._detach = None
        self._listener = None
        self._flag = None
