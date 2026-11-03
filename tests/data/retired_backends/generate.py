# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write the Numba files of ``tests/driver/test_retired_backend_runs.py`` (SEP 43).

``numba_cpu`` no longer exists, so these can't be regenerated from this tree. Each came
from a version that had it, run from outside the checkout:

- ``numba_cpu_0.1.0.h5``: the published 0.1.0, output schema 1, a finished run.
- ``numba_cpu_stopped.h5``: the last commit before the removal, f8fd5f0, output schema
  2, a run stopped after its second saved plane.
"""

import contextlib
import sys

from solprop.driver import prop_gnlse
from solprop.log import NullReporter


class StopError(RuntimeError):
    """Stands in for a machine going away."""


class StopAfter(NullReporter):
    """Stop the run once it has saved enough planes."""

    def __init__(self, planes):
        self.planes = planes
        self.seen = 0

    def plane(self, z, summary):
        self.seen += 1
        if self.seen >= self.planes:
            raise StopError


def main(path: str, stopped: bool) -> None:
    """Write one file."""
    with contextlib.suppress(StopError):
        prop_gnlse(
            0.11,
            2e-4,
            [0.0, 0.0, -11.83e-27],
            wavelength=835e-9,
            wavelength_limits=(500e-9, 2e-6),
            trange=1e-12,
            fwhm=50e-15,
            peak_power=1e3,
            shape="sech",
            backend="numba_cpu",
            noise=None,
            output=path,
            n_save=5,
            progress=StopAfter(2) if stopped else "none",
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] == "stopped")
