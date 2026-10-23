# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Execute scan points sequentially in the calling process."""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

__all__ = ["Local"]


@dataclass(frozen=True)
class Local:
    """Run one point at a time as the scan execution reference."""

    def run[T, R](self, points: Iterable[T], work: Callable[[T], R]) -> Iterator[R]:
        """Apply the builder in the supplied point order.

        Parameters
        ----------
        points : iterable
            Independent point descriptions.
        work : callable
            Point operation; closures are supported in this executor.

        Yields
        ------
        object
            One result per input point.
        """
        for point in points:
            yield work(point)
