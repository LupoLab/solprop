# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Save a two-channel launch's population statistics at this Kokkos thread count.

Run by ``test_native_mixture_fractions.py`` in a subprocess, because the Kokkos runtime
reads its thread count once per process (``backends/matched.py``). The first argument
is the ``.npz`` file to write; the others name a case and a statistic to save, each
``<kerr>:<n_pol>:<statistic>``.
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
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.grid import RadialGrid
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule
from solprop.translate.problem import assemble

NAMES = ("species_ionisation", "species_peak_ionisation_rate")
PEAK = 3.6e10
"""The largest node field, V/m: the original's ``3e10 (1 + row / 10)``, top row."""


def launch_case(kerr, n_pol):
    """Return ``(grid, problem)``: two channels on eight radial nodes, at :data:`PEAK`.

    With ``kerr`` the columns read the analytic signal (a Kerr response without its
    harmonic); without it they are the real field's.
    """
    mixture, grid, _, responses, (_, original_request) = mixture_translation(kerr=kerr)
    space = RadialGrid(6e-4, 8, collar_fraction=0.0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-9),
        80e-6,
        jones=(1,) if n_pol == 1 else (1, 0.5j),
    )
    medium = SpatialMedium(mixture)
    problem, run_request = assemble(
        grid,
        None,
        responses,
        pulse.build(grid, space, medium=medium),
        spatial_grid=space,
        medium=medium,
        controller=original_request.stepper.controller,
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
        schedule=FixedSchedule.uniform(0, grid.length, grid.length),
    )
    reference = NumpyRefBackend(problem, run_request, grid=grid)
    try:
        reference.set_initial(problem.initial.field0, None)
        reference._hoist(0.0)
        state = reference._require_state()
        peak = max(float(np.max(abs(one))) for one in reference._plasma_nodes(state[0]))
    finally:
        reference.close()
    assert peak > 0.0
    return (
        grid,
        dataclasses.replace(
            problem,
            initial=dataclasses.replace(
                problem.initial, field0=problem.initial.field0 * (PEAK / peak)
            ),
        ),
        run_request.for_run(stats=StatsSpec(names=NAMES)),
    )


def launch_statistics(backend_class, kerr, n_pol):
    """Return the launch's statistics on one backend."""
    grid, problem, run_request = launch_case(kerr, n_pol)
    backend = backend_class(problem, run_request, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        return backend.stats(run_request.observation.stats)
    finally:
        backend.close()


# The doctest collection imports every module under tests/, and the arguments it
# was given are not case names: only a subprocess run saves the arrays.
if __name__ == "__main__":
    from backends.forced_threads import record_sizing

    record_sizing(sys.argv[1])
    native = importlib.import_module("solprop_native")
    saved: dict[str, Any] = {
        "runtime_threads": np.array(native.build_info().runtime_threads),
    }
    for position, name in enumerate(sys.argv[2:]):
        kerr, n_pol, statistic = name.split(":")
        values = launch_statistics(native.NativeBackend, kerr == "kerr", int(n_pol))
        saved[f"case{position}"] = np.asarray(values[statistic])
    np.savez(sys.argv[1], **saved)
