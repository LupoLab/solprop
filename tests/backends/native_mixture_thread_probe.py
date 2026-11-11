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
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np

# The runtime loads before solprop's FFTs, as in conftest. Without the optional
# wheel the doctest collection still imports this module, and nothing runs.
if importlib.util.find_spec("solprop_native") is not None:
    importlib.import_module("solprop_native")

from mixture_cases import mixture_translation
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule

NAMES = ("species_ionisation", "species_electron_density")
STEPS = 8


def problem_and_grid():
    """Return the eight-step fixed-schedule mixture problem, its request and grid."""
    _, grid, _, _, (problem, original_request) = mixture_translation()
    run_request = original_request.with_stepper(
        dataclasses.replace(
            original_request.stepper,
            schedule=FixedSchedule.uniform(0, grid.length, grid.length / STEPS),
        )
    ).for_run(stats=StatsSpec(names=NAMES, periodic=NAMES, period=1))
    return problem, run_request, grid


# The doctest collection imports every module under tests/, and the arguments it
# was given are not array names: only a subprocess run saves the arrays.


def _runtime_threads() -> np.ndarray:
    """Return the Kokkos pool this probe ran with, for the caller to check."""
    return np.array(
        importlib.import_module("solprop_native").build_info().runtime_threads
    )


if __name__ == "__main__":
    from backends._session_probe_b1 import history, last_field, rows, run
    from backends.forced_threads import record_sizing

    record_sizing(sys.argv[1])
    solprop_native = importlib.import_module("solprop_native")
    problem, run_request, grid = problem_and_grid()
    backend = solprop_native.NativeBackend(problem, run_request, grid=grid)
    try:
        result = run(backend, problem)
        values = result.observations[-1].stats
        taken = rows(result)
        arrays: dict[str, Any] = {
            "field": last_field(result),
            "threads": np.array(solprop_native.build_info().thread_count),
            "steps": np.array(history(result).n_steps),
            "rows": np.array(taken.n_rows),
        }
        for name in NAMES:
            arrays[f"rows:{name}"] = taken.values[name]
            arrays[f"stats:{name}"] = values[name]
        np.savez(
            sys.argv[1],
            runtime_threads=_runtime_threads(),
            **{f"case{index}": arrays[key] for index, key in enumerate(sys.argv[2:])},
        )
    finally:
        backend.close()
