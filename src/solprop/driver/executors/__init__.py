# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Execute independent scan points under an explicit resource policy.

``Executor`` is the protocol: ``run(points, work)`` yields one result per point,
possibly in completion order, each carrying its global index. Four policies satisfy it,
one per module: :class:`~solprop.driver.executors.local.Local` (sequential, in the
calling process), :class:`~solprop.driver.executors.pool.Pool` (spawned workers sized by
CPUs and memory), :class:`~solprop.driver.executors.queue.Queue` (a durable file queue
drained by controllers sharing a filesystem) and
:class:`~solprop.driver.executors.slurm.Slurm` (an array job of queue controllers). The
design is ``docs/dev/scans.md``.
"""

from collections.abc import Callable, Iterable, Iterator
from typing import Protocol, runtime_checkable

from solprop.driver.executors.local import Local
from solprop.driver.executors.pool import Pool
from solprop.driver.executors.queue import Queue
from solprop.driver.executors.slurm import Slurm

__all__ = ["Executor", "Local", "Pool", "Queue", "Slurm"]


@runtime_checkable
class Executor(Protocol):
    """Run point work while preserving the identity carried by every result.

    :func:`~solprop.driver.scan.run_scan` needs only :meth:`run`, and calls a
    ``validate(work)`` method before it writes anything where an executor has one. The
    rest of what it does is decided by the four executors' classes, a closed set: an
    executor of any other class runs as ``Local`` and ``Pool`` do, its points' results
    written by the calling process, without the backend ``Local`` keeps between
    points, or the stop refusal, durable preparation and collection of ``Queue`` and
    ``Slurm``.
    """

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
