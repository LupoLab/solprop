# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Save a two-channel mixture run's every statistic at this Kokkos thread count.

Run by ``test_mixture_ladder.py`` in a subprocess, because the Kokkos runtime reads its
thread count once per process (``backends/matched.py``). The first argument is the
``.npz`` file to write; the others name what to save, in order: ``field``, a final
statistic ``stats:<statistic>``, a periodic row ``rows:<statistic>``, the rows'
positions ``z``, or the count ``steps``. The run is eight Runge-Kutta steps on a fixed
schedule, recording :data:`PERIODIC` at every step.
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

NAMES = (
    "species_ionisation",
    "species_electron_density",
    "species_peak_ionisation_rate",
    "species_density",
    "species_ionisation_energy",
    "electron_density",
    "peak_ionisation_rate",
)
PERIODIC = ("species_ionisation", "species_electron_density")


def fixed_schedule(run_request, grid, steps=8):
    """Return `run_request` replaying ``steps`` uniform Runge-Kutta steps to the end."""
    assert run_request.stepper.integrator is None
    return run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper,
            schedule=FixedSchedule.uniform(0.0, grid.length, grid.length / steps),
        )
    )


# The doctest collection imports every module under tests/, and the arguments it
# was given are not array names: only a subprocess run saves the arrays.

if __name__ == "__main__":
    from backends._session_probe_b1 import history, last_field, rows, run
    from backends.forced_threads import record_sizing

    record_sizing(sys.argv[1])
    solprop_native = importlib.import_module("solprop_native")
    _, grid, _, _, (problem, original_request) = mixture_translation()
    run_request = fixed_schedule(original_request, grid).for_run(
        stats=StatsSpec(names=PERIODIC, periodic=PERIODIC, period=1)
    )
    backend = solprop_native.NativeBackend(problem, run_request, grid=grid)
    try:
        # The final plane's statistics are every name, its rows the periodic ones.
        result = run(
            backend,
            problem,
            run_request.for_run(
                stats=StatsSpec(names=NAMES, periodic=PERIODIC, period=1)
            ),
        )
        values = result.observations[-1].stats
        taken = rows(result)
        arrays: dict[str, Any] = {
            "field": last_field(result),
            "steps": np.array(history(result).n_steps),
            "z": taken.z,
        }
        for name in NAMES:
            arrays[f"stats:{name}"] = values[name]
        for name in PERIODIC:
            arrays[f"rows:{name}"] = taken.values[name]
        np.savez(
            sys.argv[1],
            runtime_threads=np.array(solprop_native.build_info().runtime_threads),
            **{f"case{index}": arrays[key] for index, key in enumerate(sys.argv[2:])},
        )
    finally:
        backend.close()
