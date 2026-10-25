# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Save a two-channel mixture run's field and statistics at this Kokkos thread count.

Run by ``test_mixture_native_threads.py`` in a subprocess, because the Kokkos runtime
reads its thread count once per process (``backends/matched.py``). The first argument
is the ``.npz`` file to write; the others name what to save, in order: ``field``, a
periodic row ``rows:<statistic>``, a final statistic ``stats:<statistic>``, or the
counts ``threads``, ``steps`` and ``rows``. The run is eight Runge-Kutta steps on a
fixed schedule.
"""

import dataclasses
import importlib
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np

# The runtime loads before solprop's FFTs, as in conftest. Without the optional
# wheel the doctest collection still imports this module, and nothing runs.
if importlib.util.find_spec("solprop_native") is not None:
    importlib.import_module("solprop_native")

from mixture_cases import mixture_case
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule

NAMES = ("species_ionisation", "species_electron_density")
STEPS = 8


def problem_and_grid():
    """Return the eight-step fixed-schedule mixture problem and its grid."""
    _, grid, _, _, original = mixture_case()
    problem = dataclasses.replace(
        original,
        stepper=dataclasses.replace(
            original.stepper,
            schedule=FixedSchedule.uniform(0, grid.length, grid.length / STEPS),
        ),
        stats_spec=StatsSpec(names=NAMES, periodic=NAMES, period=1),
    )
    return problem, grid


# The doctest collection imports every module under tests/, and the arguments it
# was given are not array names: only a subprocess run saves the arrays.


def _runtime_threads() -> np.ndarray:
    """Return the Kokkos pool this probe ran with, for the caller to check."""
    return np.array(
        importlib.import_module("solprop_native").build_info().runtime_threads
    )


if __name__ == "__main__":
    solprop_native = importlib.import_module("solprop_native")
    problem, grid = problem_and_grid()
    backend = solprop_native.NativeBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        summary = backend.propagate_to(grid.length)
        observed = backend.observe()
        values = backend.stats(StatsSpec(names=NAMES))
        arrays = {
            "field": observed.field,
            "threads": np.array(solprop_native.build_info().thread_count),
            "steps": np.array(summary.n_steps),
            "rows": np.array(observed.step_stats.n_rows),
        }
        for name in NAMES:
            arrays[f"rows:{name}"] = observed.step_stats.values[name]
            arrays[f"stats:{name}"] = values[name]
        np.savez(
            sys.argv[1],
            runtime_threads=_runtime_threads(),
            **{f"case{index}": arrays[key] for index, key in enumerate(sys.argv[2:])},
        )
    finally:
        backend.close()
