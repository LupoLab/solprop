# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Hold NumPy's and SciPy's BLAS pools to one thread while a native backend lives.

The host side of a native run calls NumPy's BLAS only for small products (the power
weights' expansion at each saved plane, a 3 x 2561 ``dgemv`` on the averaged helium
case). NumPy's OpenBLAS is a separate, pthreads library with its own pool, sized from
``OPENBLAS_NUM_THREADS`` or every CPU: after each call its workers spin, waiting for the
next one, on the cores the native pool needs, and saved planes come often enough that
they never sleep. On 16 cores of a Xeon Gold 6240 that pool took 1.09 s of CPU per
worker in a 1.24 s window and doubled the window's time. So, as ``numba_cpu`` holds
its BLAS pool while it lives (``solprop.backends.numba_cpu.fft.apply_pools``), the
first native backend of a process limits every BLAS that threadpoolctl finds, except
this package's own (whose width is fitted at import), to one thread, or to
``SOLPROP_BLAS_THREADS`` within the thread budget; the last one to close gives the
pools back.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from ._budget import thread_count

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_holders = 0
_limiter: Any = None
_reached = "unset"


def _limit(own_library: str, budget: int, controller: Any = None) -> tuple[str, Any]:
    """Limit the BLAS pools other than `own_library`; say which, return the limit."""
    if controller is None:
        try:
            import threadpoolctl  # solprop's dependency, not this package's
        except ImportError:
            return "unavailable", None
        controller = threadpoolctl.ThreadpoolController()
    own = os.path.realpath(own_library) if own_library else ""
    others = [
        one.filepath
        for one in controller.select(user_api="blas").lib_controllers
        if os.path.realpath(one.filepath) != own
    ]
    if not others:
        return "none", None
    requested = thread_count("SOLPROP_BLAS_THREADS")
    threads = 1 if requested is None else min(requested, budget)
    libraries = controller.select(filepath=others)
    limiter = libraries.limit(limits=threads)
    names = ",".join(sorted({one.internal_api for one in libraries.lib_controllers}))
    return f"{names}: {threads} thread{'s' if threads > 1 else ''}", limiter


def hold(own_library: str, budget: int, controller: Any = None) -> str:
    """Take the host BLAS pools for one more backend; return what applies.

    Parameters
    ----------
    own_library : str
        This package's BLAS library, which keeps its own fitted width.
    budget : int
        The process's thread budget, which ``SOLPROP_BLAS_THREADS`` stays within.
    controller : threadpoolctl.ThreadpoolController, optional
        The library walk; a test passes its own.

    Returns
    -------
    str
        The libraries limited and their width, ``"none"`` when there is no other BLAS,
        or ``"unavailable"`` without threadpoolctl.
    """
    global _holders, _limiter, _reached
    with _LOCK:
        if _holders == 0:
            _reached, _limiter = _limit(own_library, budget, controller)
            logger.info("native: host BLAS pools %s", _reached)
        _holders += 1
        return _reached


def release() -> None:
    """Give one backend's hold back; the last one restores the pools."""
    global _holders, _limiter
    with _LOCK:
        if _holders == 0:
            return
        _holders -= 1
        if _holders == 0 and _limiter is not None:
            _limiter.restore_original_limits()
            _limiter = None
