# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fixed-schedule lockstep batches of ``mlx`` members (SEP 33 section 6.3, Q10).

:class:`~solprop.backends.batch.LockstepRunner` asks each member for its trial and its
host norm in turn, so a member whose trial synchronises could not queue the others
first. :class:`MlxBatchRunner` keeps the public ``select_batch`` interface and every
member's own state, ledgers and statistics, and splits each trial: it builds every
member's step graph, evaluates them together, then reads each member's norm.
"""

from collections.abc import Sequence
from typing import cast

import mlx.core as mx  # pyright: ignore[reportMissingImports] - macOS arm64 only

from solprop.backends.batch import LockstepRunner
from solprop.backends.mlx_gpu.backend import MlxBackend

__all__ = ["MlxBatchRunner"]


class MlxBatchRunner(LockstepRunner):
    """Advance loaded ``mlx`` members through one fixed schedule, evaluated together.

    Parameters
    ----------
    members : sequence of MlxBackend
        Loaded members with matching shapes, fixed schedules and Runge-Kutta
        tableaus. Ownership transfers on successful construction.
    """

    name = "mlx"

    def __init__(self, members: Sequence[MlxBackend]) -> None:
        """Adopt the members."""
        super().__init__(members)  # pyright: ignore[reportArgumentType]
        self._mlx_members = tuple(members)

    def _trial_step(self, z: float, dz: float) -> tuple[float, tuple[object, ...]]:
        """Build every member's trial, evaluate them together, then score each.

        Parameters
        ----------
        z : float
            Start position, m.
        dz : float
            Positive fixed step length, m.

        Returns
        -------
        tuple
            The largest member norm, a diagnostic, and the member trial states.
        """
        graphs = [member._trial_graph(z, dz) for member in self._mlx_members]
        mx.eval(*(array for graph in graphs for array in graph))
        norms = tuple(
            member._device_norm(error, state, advanced)
            for member, (state, error, advanced) in zip(
                self._mlx_members, graphs, strict=True
            )
        )
        self._member_norms = norms
        return max(norms), tuple(cast(object, graph[2]) for graph in graphs)

    def metadata(self) -> dict[str, object]:
        """Return the batch's settings.

        Returns
        -------
        dict
            The runner's, with the trials evaluated together.
        """
        return super().metadata() | {
            "batch_strategy": "member_graphs_evaluated_together"
        }
