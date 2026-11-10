# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Copy failures without retaining Julia exceptions or traceback owners."""

from traceback import clear_frames

from solprop.exceptions import BackendError

__all__ = ["detached"]


def detached(error: BaseException, message: str) -> BaseException:
    """Copy a failure on the owner thread before leaving the catching suite.

    Parameters
    ----------
    error : BaseException
        The original failure. Formatting can enter Julia, so callers must already
        enforce main-thread ownership and release their own raw locals separately.
    message : str
        Context for the Python-only replacement.

    Returns
    -------
    BaseException
        A fresh BackendError, KeyboardInterrupt or SystemExit. Raise it only after
        leaving the except suite, so its context cannot retain the original failure.
    """
    if isinstance(error, KeyboardInterrupt):
        failure: BaseException = KeyboardInterrupt()
    elif isinstance(error, SystemExit):
        code = error.code
        failure = SystemExit(
            code if code is None or isinstance(code, (int, str)) else _text(code)
        )
    else:
        failure = BackendError(f"{message}: {type(error).__name__}: {_text(error)}")
    _clear_graph(error)
    return failure


def _text(error: object) -> str:
    """Format a failure on main, retaining no failure of its formatter.

    Parameters
    ----------
    error : object
        The original exception or exit status.

    Returns
    -------
    str
        Its text, or an explicit formatting-failure description.
    """
    try:
        return str(error)
    except BaseException as formatting_error:
        _clear_graph(formatting_error)
        return "exception message could not be formatted"


def _clear_graph(error: BaseException) -> None:
    """Clear every chained exception frame while Julia destruction is safe.

    Parameters
    ----------
    error : BaseException
        Root of the exception graph to detach on main.
    """
    # Causes, contexts and grouped children can each retain a frame/exception cycle.
    # Clear the whole graph on main; a later Python GC may run on any worker thread.
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        clear_frames(current.__traceback__)
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None
