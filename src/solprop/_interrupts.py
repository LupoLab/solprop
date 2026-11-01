# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Deliver SIGTERM and Ctrl-C to a run between its commits, never inside one.

A run commits in short sections: a step's new state and position, a saved plane and
the march's snapshot beside it, a terminal checkpoint. An exception raised inside one
-- as a signal handler's is, at whatever line the interpreter has reached -- could
leave the state and its position apart, or a file half written. While a run is in
progress (:func:`handled`), SIGTERM raises :class:`solprop.exceptions.RunTerminated`
and SIGINT :class:`KeyboardInterrupt` at once, which abandons a trial in flight without
its result, unless a commit is being written (:func:`committing`): then the exception
is held and raised as the commit ends. The run loop catches it as any failure and
writes its terminal checkpoint.
"""

import signal
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import FrameType
from typing import Protocol, runtime_checkable

from solprop.exceptions import RunTerminated

__all__ = [
    "StopFlag",
    "StopRequested",
    "committing",
    "deferred_handlers",
    "handled",
    "poll",
]


@runtime_checkable
class StopFlag(Protocol):
    """A run's own stop flag: any object with an ``is_set()`` method (SEP 41, S17).

    :class:`threading.Event` qualifies. A front end sets it from any thread, on any
    platform, without signals.
    """

    def is_set(self) -> bool:
        """Return whether the run is asked to stop.

        Returns
        -------
        bool
            True once the caller has set the flag.
        """
        ...


class StopRequested(BaseException):
    """Raised inside a run when its own stop flag is set (SEP 41, S17).

    The run loop writes its state as for SIGTERM, and
    :meth:`solprop.driver.simulation.Simulation.run` returns the output when the flag
    that stopped it is its own: a requested stop is not an error. It derives from
    :class:`BaseException`, as :class:`KeyboardInterrupt` does, so no ``except
    Exception`` on the way swallows it.

    Parameters
    ----------
    flag : StopFlag
        The flag that was set.

    Examples
    --------
    >>> import threading
    >>> from solprop._interrupts import StopRequested, poll
    >>> flag = threading.Event()
    >>> poll(flag)
    >>> flag.set()
    >>> try:
    ...     poll(flag)
    ... except StopRequested as stop:
    ...     print(stop, stop.flag is flag)
    stopped by the caller's stop flag True
    """

    def __init__(self, flag: StopFlag) -> None:
        """Hold the flag that was set."""
        super().__init__("stopped by the caller's stop flag")
        self.flag = flag


def poll(flag: StopFlag | None) -> None:
    """Raise :class:`StopRequested` when a run's stop flag is set.

    Parameters
    ----------
    flag : StopFlag or None
        The run's flag; ``None`` never stops.

    Raises
    ------
    StopRequested
        If the flag is set.
    """
    if flag is not None and flag.is_set():
        raise StopRequested(flag)


class _Guard:
    """How deep the current commit is, and what signal waits for its end."""

    def __init__(self) -> None:
        """Start outside every commit with nothing pending."""
        self.depth = 0
        self.pending: BaseException | None = None


_GUARD = _Guard()


class committing:  # noqa: N801 - used as a function, ``with committing():``
    """Hold SIGTERM and Ctrl-C while the block writes a commit.

    A class rather than a generator: it is entered at every accepted step.

    Parameters
    ----------
    deliver : bool, optional
        Raise a signal that arrived during the block when it ends; ``False`` drops it,
        for a commit written while the run is already stopping.

    Raises
    ------
    solprop.exceptions.RunTerminated
        If SIGTERM arrived during the block and `deliver` holds.
    KeyboardInterrupt
        If Ctrl-C arrived during the block and `deliver` holds.

    Examples
    --------
    >>> from solprop._interrupts import committing
    >>> with committing():
    ...     state, position = "new state", 1.0
    """

    __slots__ = ("_deliver",)

    def __init__(self, *, deliver: bool = True) -> None:
        """Hold whether the block delivers a pending signal."""
        self._deliver = deliver

    def __enter__(self) -> None:
        """Enter the commit."""
        _GUARD.depth += 1

    def __exit__(self, kind: type[BaseException] | None, *_: object) -> None:
        """Leave the commit, raising a signal that waited for it.

        Parameters
        ----------
        kind : type or None
            The exception leaving the block, if any; a pending signal does not
            replace it.
        *_ : object
            The exception and its traceback, unused.

        Raises
        ------
        BaseException
            The pending signal's exception, when the block ended normally.
        """
        _GUARD.depth -= 1
        if _GUARD.depth or _GUARD.pending is None:
            return
        pending, _GUARD.pending = _GUARD.pending, None
        if self._deliver and kind is None:
            raise pending


def _handle(signal_number: int, frame: FrameType | None) -> None:
    """Raise the signal's exception now, or when the current commit ends.

    Parameters
    ----------
    signal_number : int
        The signal received.
    frame : types.FrameType or None
        The interrupted frame, unused.

    Raises
    ------
    solprop.exceptions.RunTerminated
        For SIGTERM, outside a commit.
    KeyboardInterrupt
        For SIGINT, outside a commit.
    """
    del frame
    error: BaseException = (
        KeyboardInterrupt()
        if signal_number == signal.SIGINT
        else RunTerminated(signal_number)
    )
    if _GUARD.depth:
        _GUARD.pending = error
        return
    raise error


@contextmanager
def deferred_handlers() -> Iterator[None]:
    """Defer application signal handlers until a file transaction finishes.

    The run's own handlers use :class:`committing`. Temporarily queue callable
    application handlers too, then restore and invoke them after the transaction.
    Ignored signals stay ignored. A failing transaction keeps its primary error.
    Outside the main thread only the existing commit guard applies.

    Yields
    ------
    None
        With TERM and Ctrl-C deferred until all working files are open again.
    """
    pending: tuple[int, FrameType | None] | None = None
    previous: dict[int, Callable[[int, FrameType | None], object]] = {}

    def defer(number: int, frame: FrameType | None) -> None:
        """Remember the last signal without running application code.

        Parameters
        ----------
        number : int
            Signal number.
        frame : types.FrameType or None
            Interrupted Python frame to pass to the original handler.
        """
        nonlocal pending
        pending = number, frame

    with committing():
        if threading.current_thread() is threading.main_thread():
            for number in (signal.SIGTERM, signal.SIGINT):
                handler = signal.getsignal(number)
                if callable(handler) and handler is not _handle:
                    previous[number] = handler
                    signal.signal(number, defer)
        try:
            yield
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
        if pending is not None:
            number, frame = pending
            previous[number](number, frame)


@contextmanager
def handled() -> Iterator[None]:
    """Route SIGTERM and Ctrl-C through the commit guard for the block's duration.

    Installed only in the main thread, which is where Python delivers signals, and only
    over Python's own defaults: an application that set its own handler keeps it.
    Restored on exit.

    Yields
    ------
    None
        With the handlers installed where they could be.

    Examples
    --------
    >>> from solprop._interrupts import handled
    >>> with handled():
    ...     pass
    """
    defaults: dict[int, Callable[..., object] | int | None] = {
        signal.SIGTERM: signal.SIG_DFL,
        signal.SIGINT: signal.default_int_handler,
    }
    installed: dict[int, Callable[..., object] | int | None] = {}
    if threading.current_thread() is threading.main_thread():
        for number, default in defaults.items():
            if signal.getsignal(number) is default:
                installed[number] = signal.signal(number, _handle)
    try:
        yield
    finally:
        for number, previous in installed.items():
            signal.signal(number, previous)
        if installed:
            _GUARD.pending = None
