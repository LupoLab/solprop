# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write ten fixed steps of a soliton run, for the OpenMP thread-count test.

The run replays a prescribed schedule with the explicit pair, so one and many threads
perform the same discrete computation and differ by rounding alone (solprop SEP 25).
Usage: ``thread_probe.py <out.npy> [native|numpy_ref]``.
"""

# ruff: noqa: I001 -- Kokkos must load before solprop can import pyFFTW on macOS.

from __future__ import annotations

import json
import sys

import numpy as np

import solprop_native

from solprop.backends.registry import select
from solprop.grid import EnvelopeGrid
from solprop.nonlinear.gnlse import problem_from
from solprop.pulses import SechPulse
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import assemble

BETA2 = -11.83e-27
GAMMA = 0.11
TAU0 = 100e-15
WAVELENGTH = 835e-9
LENGTH = 1e-3
STEPS = 10

grid = EnvelopeGrid.from_limits(1.0, WAVELENGTH, (500e-9, 2e-6), 1e-12)
model = problem_from([0.0, 0.0, BETA2], omega_ref=grid.omega0, gamma=GAMMA, shock=False)
peak = 9 * abs(BETA2) / (GAMMA * TAU0**2)
initial = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=peak).build(grid)
problem = assemble(
    grid,
    model.mode,
    [model.kerr],
    initial,
    controller=PIController(rtol=1e-10, atol=1e-18),
    save_planes=[LENGTH],
    schedule=FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
    dz_max=LENGTH / STEPS,
    dz_init=LENGTH / STEPS,
    density=1.0,
    shock=False,
    prefactor_index=1.0,
)
backend = select(sys.argv[2] if len(sys.argv) > 2 else "native", problem, grid=grid)
try:
    backend.set_initial(problem.initial.field0, None)
    summary = backend.propagate_to(LENGTH)
    field = backend.observe().require_field()
    info = solprop_native.build_info()
    sizing = backend.metadata().get("execution_sizing", {})
finally:
    backend.close()
np.save(sys.argv[1], field)
print(
    json.dumps(
        {
            "samples": grid.n_to,
            "steps": summary.n_steps,
            "rejected": summary.n_rejected,
            "threads": info.thread_count,
            "explicit_thread_pools": info.explicit_thread_pools,
            "execution_sizing": sizing,
        },
        sort_keys=True,
    )
)
