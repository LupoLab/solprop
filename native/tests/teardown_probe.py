# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run one problem on the native and reference backends, then let the interpreter exit.

The exit is the test: an FFTW plan is still alive when the interpreter finalises, and it
shares FFTW's process-wide planner with the Kokkos-FFT plans, because it is made through
ctypes on the very library this build links. The last line printed lists the backends
that ran.
"""

# ruff: noqa: I001 -- Kokkos loads first, before any other OpenMP runtime on macOS.

from __future__ import annotations

import ctypes
import json

import solprop_native

from solprop.backends.registry import select
from solprop.grid import EnvelopeGrid
from solprop.nonlinear.gnlse import problem_from
from solprop.pulses import SechPulse
from solprop.stepper import PIController
from solprop.translate.assembly import assemble

from session_runs import run

BETA2 = -11.83e-27
GAMMA = 0.11
TAU0 = 100e-15
WAVELENGTH = 835e-9
FFTW_FORWARD = -1
FFTW_ESTIMATE = 1 << 6


class FftwPlan:
    """A 64-point transform planned on one FFTW library, destroyed with the object."""

    def __init__(self, path: str) -> None:
        library = self.library = ctypes.CDLL(path)
        library.fftw_malloc.restype = ctypes.c_void_p
        library.fftw_malloc.argtypes = [ctypes.c_size_t]
        library.fftw_free.argtypes = [ctypes.c_void_p]
        library.fftw_plan_dft_1d.restype = ctypes.c_void_p
        library.fftw_plan_dft_1d.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        library.fftw_destroy_plan.argtypes = [ctypes.c_void_p]
        self.buffer = library.fftw_malloc(64 * 16)
        self.plan = library.fftw_plan_dft_1d(
            64, self.buffer, self.buffer, FFTW_FORWARD, FFTW_ESTIMATE
        )

    def __del__(self) -> None:
        self.library.fftw_destroy_plan(self.plan)
        self.library.fftw_free(self.buffer)


grid = EnvelopeGrid.from_limits(1.0, WAVELENGTH, (500e-9, 2e-6), 1e-12)
model = problem_from([0.0, 0.0, BETA2], omega_ref=grid.omega0, gamma=GAMMA, shock=False)
peak = abs(BETA2) / (GAMMA * TAU0**2)
initial = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=peak).build(grid)
problem, run_request = assemble(
    grid,
    model.mode,
    [model.kerr],
    initial,
    controller=PIController(rtol=1e-8, atol=1e-16),
    save_planes=[1e-4],
    dz_max=1e-4,
    dz_init=1e-5,
    density=1.0,
    shock=False,
    prefactor_index=1.0,
)

names = ("native", "numpy_ref")
backends: list[object] = []
for name in names:
    # Each backend is kept to the end of the script, as a user's script keeps its
    # simulations, so what it holds is released by the interpreter's own finalisation.
    backend = select(name, problem, run_request, grid=grid)
    run(backend, problem, run_request)
    backends.append(backend)
# A plan on the very FFTW this build links.
library = solprop_native.build_info().fft_library
native = FftwPlan(library) if "fftw" in library.lower() else None

print(json.dumps(names))
