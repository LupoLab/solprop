# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Translate failures before Julia wrappers can escape in Python tracebacks."""

from collections.abc import Callable

from solprop.backends.julia._errors import detached
from solprop.backends.julia._ownership import require_main_thread
from solprop.exceptions import BackendError

__all__ = ["invoke"]


class _CallState:
    """Refuse recursive calls while main already owns a bridge operation."""

    active: bool = False


_ENTRY = _CallState()


def invoke[T](operation: Callable[[], T], message: str) -> T:
    """Run a narrow owner-only bridge operation and detach its failure.

    Parameters
    ----------
    operation : callable
        An adapter helper that holds all temporary Julia wrappers in its own frame.
        Its captures and return value must be Python data or registry handles. Never
        pass a raw Julia function or return a borrowed Julia array or wrapper.
    message : str
        Context for a failure, including an actionable remedy where available.

    Returns
    -------
    object
        The helper's Python data or owned handle.

    Raises
    ------
    BackendError
        With a Python-only description of a failed bridge operation.
    KeyboardInterrupt
        A fresh interrupt, without the bridge's traceback or Julia exception owners.
    SystemExit
        A fresh exit request, retaining only a scalar exit status or text.
    """
    require_main_thread()
    if _ENTRY.active:
        raise BackendError("recursive julia bridge entry is not allowed")
    _ENTRY.active = True
    try:
        try:
            return operation()
        except BaseException as error:
            failure = detached(error, message)
    finally:
        _ENTRY.active = False
    # Raising inside the except suite would retain JuliaError in __context__, even
    # with `from None`. The new traceback must not retain the operation either.
    del operation
    raise failure from None
