# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Kill a monitored run hard, outside pytest's process.

``python -m tests.monitor_worker STAGE PATH``: ``first`` runs the walking pulse and
exits at its fourth plane; ``second`` resumes it and exits at its third step, before it
reaches a plane; ``third`` resumes it to the end. A hard exit writes no closing record,
so each execution leaves the file as SIGKILL would.
"""

import os
import sys
import warnings
from pathlib import Path

from solprop.exceptions import BoundaryWarning
from solprop.log import NullReporter
from solprop.monitor import BoundaryCheck
from test_monitor import TRANGE, walker

RELAXED = BoundaryCheck(budget=1.0, stop_factor=1.0)
"""A check that neither warns nor stops on the walking pulse."""


class _ExitAtPlane(NullReporter):
    """Exit the process at the `n`-th saved plane."""

    def __init__(self, n: int) -> None:
        self.n, self.seen = n, 0

    def plane(self, z: float, summary: object) -> None:
        del z, summary
        self.seen += 1
        if self.seen == self.n:
            os._exit(9)


class _ExitAtStep(NullReporter):
    """Exit the process at the `n`-th attempted step."""

    def __init__(self, n: int) -> None:
        self.n, self.seen = n, 0

    def step(self, z: float, dz: float, accepted: bool) -> None:
        del z, dz, accepted
        self.seen += 1
        if self.seen == self.n:
            os._exit(9)


def main(stage: str, path: Path) -> None:
    """Run one execution of the three."""
    warnings.simplefilter("ignore", BoundaryWarning)
    simulation = walker(0.65 * TRANGE, check=RELAXED, n_save=6)
    if stage == "first":
        simulation.run(path, progress=_ExitAtPlane(4))
    elif stage == "second":
        simulation.run(path, resume=True, progress=_ExitAtStep(3))
    else:
        simulation.run(path, resume=True, progress="none")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
