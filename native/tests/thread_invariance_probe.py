# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write a multicolumn run's field, history and statistics for the thread-count test.

A two-mode argon capillary with plasma, replayed on a prescribed schedule with the
explicit pair, so one and many threads perform the same discrete computation and
differ by rounding alone (solprop SEP 25). Usage:
``thread_invariance_probe.py <out.npz>``.
"""

# ruff: noqa: I001 -- Kokkos must load before solprop imports pyFFTW on macOS.

from __future__ import annotations

import json
import sys
import warnings

import solprop_native

import numpy as np

from solprop.backends.registry import select
from solprop.driver import Simulation
from solprop.exceptions import MaterialRangeWarning
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, RungeKutta

LENGTH = 0.01
STEPS = 40
NAMES = (
    "omega0",
    "energy",
    "peak_power",
    "peak_power_total",
    "fwhm_t",
    "fwhm_t_outer",
    "electron_density",
    "peak_ionisation_rate",
    "peak_intensity",
    "quadrature_error",
)

spec = StatsSpec(names=NAMES, periodic=NAMES, period=1)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", MaterialRangeWarning)
    simulation = Simulation.from_capillary(
        125e-6,
        LENGTH,
        "Ar",
        1.0,
        wavelength=800e-9,
        wavelength_limits=(250e-9, 1.5e-6),
        trange=100e-15,
        fwhm=15e-15,
        energy=5e-6,
        modes=2,
        loss=False,
        n_r=8,
        n_save=2,
        rtol=1e-6,
        atol=1e-10,
        stats=spec,
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / STEPS),
        integrator=RungeKutta(),
        backend="native",
        noise=None,
    )
    problem = simulation.translate()

backend = select(
    "native",
    problem,
    grid=simulation.grid,
    modes=simulation.modes,
    profiles=simulation.profiles,
)
try:
    backend.set_initial(problem.initial.field0, None)
    summary = backend.propagate_to(LENGTH)
    observation = backend.observe()
    at_plane = backend.stats(spec)
    metadata = backend.metadata()
finally:
    backend.close()

history = observation.step_history
np.savez(
    sys.argv[1],
    allow_pickle=False,
    field=observation.require_field(),
    z=history.z,
    dz=history.dz,
    error_norm=history.error_norm,
    accepted=history.accepted,
    **{f"rows_{name}": observation.step_stats.values[name] for name in NAMES},
    **{f"plane_{name}": at_plane[name] for name in NAMES},
)

print(
    json.dumps(
        {
            "samples": simulation.grid.n_to,
            "steps": summary.n_steps,
            "rejected": summary.n_rejected,
            "rows": observation.step_stats.n_rows,
            "threads": solprop_native.build_info().thread_count,
            "synthesis_provider": metadata["synthesis_provider"],
            "projection_provider": metadata["projection_provider"],
        },
        sort_keys=True,
    )
)
