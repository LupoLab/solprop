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

from solprop.exceptions import RunTerminated, StopRequested
from solprop.log import StopFlag

__all__ = [
    "committing",
    "deferred_handlers",
    "handled",
    "interruptible",
    "poll",
    "raise_held",
]


def poll(flag: StopFlag | None) -> None:
    """Raise :class:`StopRequested` when a run's stop flag is set.

    - ``flag``: the run's flag; ``None`` never stops.
    """
    if flag is not None and flag.is_set():
        raise StopRequested(flag)


class _Guard:
    """How deep the current commit is, and what signal waits for its end."""

    def __init__(self) -> None:
        """Start outside every commit with nothing pending."""
        self.depth = 0
        self.pending: BaseException | None = None
        # How many interruptible scopes the main thread is in (:func:`interruptible`).
        self.interruptible = 0


_GUARD = _Guard()


class committing:  # noqa: N801 - used as a function, ``with committing():``
    """Hold SIGTERM and Ctrl-C while the block writes a commit.

    A class rather than a generator: it is entered at every accepted step.

    - ``deliver``: raise a signal that arrived during the block when it ends; ``False``
      drops it, for a commit written while the run is already stopping.
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

        - ``kind``: the exception leaving the block, if any; a pending signal does not
          replace it.
        """
        _GUARD.depth -= 1
        if _GUARD.depth or _GUARD.pending is None:
            return
        pending, _GUARD.pending = _GUARD.pending, None
        if self._deliver and kind is None:
            raise pending


@contextmanager
def interruptible() -> Iterator[None]:
    """Let a long wait inside a commit take the signal the commit holds.

    A native ``advance`` runs inside :class:`committing`, and a host source it calls may
    wait minutes for an atomic call (SEP 49 section 6.7). Its session marks that wait
    with this scope, and the wait calls :func:`raise_held` between polls, so that a
    Ctrl-C or SIGTERM stops it there, before the trial commits. Only the main thread's
    scope counts, since signals reach the main thread alone; any other commit -- the
    terminal checkpoint's ``committing(deliver=False)`` among them -- keeps holding.
    """
    main = threading.current_thread() is threading.main_thread()
    if main:
        _GUARD.interruptible += 1
    try:
        yield
    finally:
        if main:
            _GUARD.interruptible -= 1


def raise_held() -> None:
    """Raise, now, a signal the commit holds, inside an :func:`interruptible` scope.

    Does nothing outside such a scope, on any thread but the main one, or when nothing
    is held.
    """
    if not _GUARD.interruptible or threading.current_thread() is not (
        threading.main_thread()
    ):
        return
    pending, _GUARD.pending = _GUARD.pending, None
    if pending is not None:
        raise pending


def _handle(signal_number: int, frame: FrameType | None) -> None:
    """Raise the signal's exception now, or when the current commit ends."""
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
    """
    pending: tuple[int, FrameType | None] | None = None
    previous: dict[int, Callable[[int, FrameType | None], object]] = {}

    def defer(number: int, frame: FrameType | None) -> None:
        """Remember the last signal without running application code."""
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

    - Yields: with the handlers installed where they could be.
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
