# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write ten fixed steps of a 16 384-sample envelope run, for the plan-threshold test.

Every transform of this case has 16 384 samples per execution. The test explicitly
forces the FFT pool width, bypassing automatic sizing, so its plans may factorise
differently with the requested thread count.
Usage: ``fft_threshold_probe.py <out.npy>``.
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

from legacy_geometry import LEGACY_GRID

BETA2 = -11.83e-27
GAMMA = 0.11
TAU0 = 100e-15
WAVELENGTH = 835e-9
LENGTH = 1e-3
STEPS = 10

grid = EnvelopeGrid.from_limits(
    LENGTH, WAVELENGTH, (500e-9, 2e-6), 16e-12, **LEGACY_GRID
)
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
backend = select("native", problem, grid=grid)
try:
    backend.set_initial(problem.initial.field0, None)
    summary = backend.propagate_to(LENGTH)
    field = backend.observe().require_field()
    info = solprop_native.build_info()
finally:
    backend.close()
np.save(sys.argv[1], field)
print(
    json.dumps(
        {"samples": grid.n_to, "steps": summary.n_steps, "threads": info.thread_count},
        sort_keys=True,
    )
)
