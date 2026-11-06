# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A bounded, coalescing queue of run events, between the march and the pipe.

The worker's observer runs in the march's thread and must return quickly; it hands
each event to an :class:`Outbox`, and a sender thread takes them off and writes them to
the pipe (design 13.3). The outbox never blocks the producer:

- a replaceable event (progress) replaces an undelivered one of its kind;
- a terminal event is always kept;
- any other event is dropped when the outbox is full, and counted: the rows a client
  misses are replayed from the worker's history on request.
"""

import threading
from collections import deque

from solprop_gui.service.messages import Event

__all__ = ["Outbox"]


class Outbox:
    """A bounded, thread-safe queue of events that never blocks its producer.

    Parameters
    ----------
    capacity : int
        The most events held, terminal ones aside.

    Attributes
    ----------
    dropped : int
        The events refused because the outbox was full.

    Examples
    --------
    >>> from solprop_gui.service.messages import Event
    >>> box = Outbox(capacity=1)
    >>> box.put(Event("r", "progress", {"z": 0.1}))
    True
    >>> box.put(Event("r", "progress", {"z": 0.2}))
    True
    >>> box.get(timeout=0).data
    {'z': 0.2}
    """

    def __init__(self, capacity: int = 1024) -> None:
        """Make an empty outbox."""
        if capacity < 1:
            msg = f"capacity must be at least 1, not {capacity}"
            raise ValueError(msg)
        self._capacity = capacity
        self._events: deque[Event] = deque()
        self._ready = threading.Condition()
        self._closed = False
        self.dropped = 0

    def __len__(self) -> int:
        """Return the number of events waiting.

        Returns
        -------
        int
            The undelivered events.
        """
        with self._ready:
            return len(self._events)

    def put(self, event: Event) -> bool:
        """Hand over one event, without waiting.

        Parameters
        ----------
        event : Event
            The event.

        Returns
        -------
        bool
            Whether it was kept: False when the outbox was full, or closed.
        """
        with self._ready:
            if self._closed:
                return False
            if event.replaceable:
                for index, waiting in enumerate(self._events):
                    if waiting.kind == event.kind and waiting.run_id == event.run_id:
                        self._events[index] = event
                        return True
            waiting = sum(not e.terminal for e in self._events)
            if not event.terminal and waiting >= self._capacity:
                self.dropped += 1
                return False
            self._events.append(event)
            self._ready.notify()
            return True

    def get(self, timeout: float | None = None) -> Event | None:
        """Take the oldest event, waiting for one if need be.

        Parameters
        ----------
        timeout : float or None, optional
            The most seconds to wait; ``None`` waits until an event arrives or the
            outbox is closed.

        Returns
        -------
        Event or None
            The event; ``None`` when none arrived in time, or the outbox is closed
            and empty.
        """
        with self._ready:
            if not self._events and not self._closed:
                self._ready.wait(timeout)
            if self._events:
                return self._events.popleft()
            return None

    def close(self) -> None:
        """Refuse further events; those waiting can still be taken."""
        with self._ready:
            self._closed = True
            self._ready.notify_all()

    @property
    def closed(self) -> bool:
        """Return whether the outbox refuses further events.

        Returns
        -------
        bool
            True after :meth:`close`.
        """
        with self._ready:
            return self._closed
