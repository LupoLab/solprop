# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Execute independent scan points under an explicit resource policy."""

from collections.abc import Callable, Iterable, Iterator
from typing import Protocol, runtime_checkable

from solprop.driver.executors.batch import Batch
from solprop.driver.executors.local import Local
from solprop.driver.executors.pool import Pool
from solprop.driver.executors.queue import Queue
from solprop.driver.executors.slurm import Slurm

__all__ = ["Batch", "Executor", "Local", "Pool", "Queue", "Slurm"]


@runtime_checkable
class Executor(Protocol):
    """Run point work while preserving the identity carried by every result."""

    def run[T, R](self, points: Iterable[T], work: Callable[[T], R]) -> Iterator[R]:
        """Execute work for each point.

        Parameters
        ----------
        points : iterable
            Descriptions of independent work.
        work : callable
            Turn one description into a result containing its global index.

        Yields
        ------
        object
            Result, possibly in completion order. Storage uses its global index.
        """
        ...
