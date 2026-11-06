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

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np

# The runtime loads before solprop's FFTs, as in conftest. Without the optional
# wheel the doctest collection still imports this module, and nothing runs.
if importlib.util.find_spec("solprop_native") is not None:
    importlib.import_module("solprop_native")

from mixture_cases import mixture_case
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


def fixed_schedule(problem, grid, steps=8):
    """Return `problem` replaying ``steps`` uniform Runge-Kutta steps to its end."""
    assert problem.stepper.integrator is None
    return dataclasses.replace(
        problem,
        stepper=dataclasses.replace(
            problem.stepper,
            schedule=FixedSchedule.uniform(0.0, grid.length, grid.length / steps),
        ),
    )


# The doctest collection imports every module under tests/, and the arguments it
# was given are not array names: only a subprocess run saves the arrays.

if __name__ == "__main__":
    from backends.forced_threads import record_sizing

    record_sizing(sys.argv[1])
    solprop_native = importlib.import_module("solprop_native")
    _, grid, _, _, original = mixture_case()
    problem = dataclasses.replace(
        fixed_schedule(original, grid),
        stats_spec=StatsSpec(names=PERIODIC, periodic=PERIODIC, period=1),
    )
    backend = solprop_native.NativeBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        summary = backend.propagate_to(grid.length)
        observed = backend.observe()
        values = backend.stats(StatsSpec(names=NAMES))
        arrays = {
            "field": observed.require_field(),
            "steps": np.array(summary.n_steps),
            "z": observed.step_stats.z,
        }
        for name in NAMES:
            arrays[f"stats:{name}"] = values[name]
        for name in PERIODIC:
            arrays[f"rows:{name}"] = observed.step_stats.values[name]
        np.savez(
            sys.argv[1],
            runtime_threads=np.array(solprop_native.build_info().runtime_threads),
            **{f"case{index}": arrays[key] for index, key in enumerate(sys.argv[2:])},
        )
    finally:
        backend.close()
