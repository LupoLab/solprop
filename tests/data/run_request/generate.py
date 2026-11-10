# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write the stopped runs of ``tests/driver/test_run_request_identity.py`` (SEP 47).

Both were written by commit 50d47d16, the last before the stepper settings and the
statistics left ``PropagationProblem`` for the run request, so they cannot be
regenerated from a later tree: a file written after the change would prove nothing.
Each is the soliton example on a narrower window, shortened to 0.3 m with five planes,
on ``numpy_ref``, stopped by its own stop flag at its first saved plane:

- ``soliton_stopped.h5``: the Runge-Kutta march;
- ``soliton_adams_stopped.h5``: ``Multistep()``, whose terminal checkpoint keeps the
  Adams history.

Run from outside the checkout, with the solprop of that commit, as
``python generate.py <directory>``.
"""

import sys
import threading
from pathlib import Path

from solprop.log import NullReporter
from solprop.recipe import Recipe

CHANGES = {
    "backend": "numpy_ref",
    "n_save": 5,
    "grid.trange": "1000 fs",
    "length": "0.3 m",
}
"""What both runs change in the soliton example."""


class StopAt(NullReporter):
    """Set the run's stop flag at its first saved plane."""

    def __init__(self, flag: threading.Event) -> None:
        self.flag = flag

    def plane(self, z, summary):
        self.flag.set()


def main(directory: Path) -> None:
    """Write both files into `directory`."""
    for name, extra in (
        ("soliton_stopped.h5", {}),
        ("soliton_adams_stopped.h5", {"integrator": {"type": "Multistep"}}),
    ):
        recipe = Recipe.example("soliton")
        for key, value in (CHANGES | extra).items():
            recipe.set(key, value)
        flag = threading.Event()
        recipe.run(directory / name, progress=StopAt(flag), stop=flag)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
