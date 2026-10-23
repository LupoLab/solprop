# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Dispatch compatible scan members through their selected lockstep backend."""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import cast

from solprop.exceptions import DriverError

__all__ = ["Batch"]


@dataclass
class Batch:
    """Run fixed-schedule scan batches in the calling process.

    The builder selects the backend. Compatible adjacent points share a fixed
    march, while each retains its input, material tables, noise and statistics.
    Smaller groups are used at layout boundaries and when memory requires them.
    Adaptive simulations must first discover and calibrate a replay schedule.

    Attributes
    ----------
    size : int
        Positive maximum number of simultaneously advancing members.
    memory_limit : int or None
        Optional positive host-memory limit in bytes, further restricted by the
        physical or scheduler allocation. Twenty percent remains as headroom.
    budgets : list of dict
        Admission calculations recorded for this invocation, in dispatch order.
        Resolved backend policies and the actual batch size are stored per point.
    """

    size: int
    memory_limit: int | None = None
    budgets: list[dict[str, int | None]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Reject invalid bounds before constructing any simulations."""
        for name, value in (("size", self.size), ("memory_limit", self.memory_limit)):
            if value is None and name == "memory_limit":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise DriverError(f"Batch {name} must be a positive integer")

    def run[T, R](self, points: Iterable[T], work: Callable[[T], R]) -> Iterator[R]:
        """Execute scan work through its backend-preserving batch operation.

        Parameters
        ----------
        points : iterable
            Globally indexed scan jobs, consumed with bounded lookahead.
        work : callable
            Scan worker exposing the common driver's batch operation.

        Yields
        ------
        object
            Independently validated results, including failures and skipped points.

        Raises
        ------
        DriverError
            If work does not implement scan batching or a contract cannot execute.
        """
        operation = getattr(work, "run_batch", None)
        if not callable(operation):
            raise DriverError("Batch requires run_scan work with a batch operation")
        self.budgets.clear()
        yield from cast(Iterator[R], operation(points, self))
