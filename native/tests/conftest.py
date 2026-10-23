# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared problem fixtures.

Import the extension before solprop's Numba backend can load pyFFTW's private OpenMP
runtime. The reverse order is refused on macOS; import ordering alone does not remove
bundled runtimes. Cross-backend timings run in separate processes, and mixed-runtime
qualification uses a pyFFTW build linked to the same Homebrew libraries.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from solprop.grid import EnvelopeGrid
from solprop.nonlinear.gnlse import problem_from
from solprop.pulses import SechPulse
from solprop.stepper import PIController
from solprop.translate.problem import KernelProblem, assemble

# Trigger the documented macOS runtime ordering without adding an unused binding.
__import__("solprop_native")

type PhaseACase = tuple[KernelProblem, EnvelopeGrid]
type RamanCase = tuple[KernelProblem, EnvelopeGrid]
type ComplexArray = npt.NDArray[np.complex128]


@pytest.fixture(scope="session")
def phase_a_case() -> PhaseACase:
    """Return a small, scalar, mode-averaged envelope problem and its grid."""
    beta2 = -11.83e-27
    gamma = 0.11
    tau0 = 100e-15
    wavelength = 835e-9
    grid = EnvelopeGrid.from_limits(1.0, wavelength, (500e-9, 2e-6), 1e-12)
    model = problem_from(
        [0.0, 0.0, beta2],
        omega_ref=grid.omega0,
        gamma=gamma,
        shock=False,
    )
    peak = abs(beta2) / (gamma * tau0**2)
    field = SechPulse(
        wavelength=wavelength,
        width=tau0,
        peak_power=peak,
    ).build(grid)
    problem = assemble(
        grid,
        model.mode,
        [model.kerr],
        field,
        save_planes=[1e-3],
        density=1.0,
        shock=False,
        prefactor_index=1.0,
        controller=PIController(rtol=1e-8, atol=1e-16),
        dz_max=1e-3,
        dz_init=1e-4,
    )
    return problem, grid


@pytest.fixture(scope="session")
def raman_case() -> RamanCase:
    """Return a compact envelope problem with one delayed silica Raman term."""
    beta2 = -11.83e-27
    gamma = 0.11
    wavelength = 835e-9
    grid = EnvelopeGrid.from_limits(0.01, wavelength, (500e-9, 2e-6), 1e-12)
    model = problem_from(
        [0.0, 0.0, beta2],
        omega_ref=grid.omega0,
        gamma=gamma,
        shock=False,
        raman=True,
        raman_fraction=0.18,
    )
    field = SechPulse(
        wavelength=wavelength,
        width=100e-15,
        peak_power=abs(beta2) / (gamma * (100e-15) ** 2),
    ).build(grid)
    problem = assemble(
        grid,
        model.mode,
        model.responses,
        field,
        save_planes=[1e-5],
        density=1.0,
        shock=False,
        prefactor_index=1.0,
        controller=PIController(rtol=1e-8, atol=1e-16),
        dz_max=1e-5,
        dz_init=1e-5,
    )
    return problem, grid


@pytest.fixture
def writable_initial(phase_a_case: PhaseACase) -> ComplexArray:
    """Return an owning, writable initial field for mutation/lifetime tests."""
    problem, _ = phase_a_case
    return np.array(problem.initial.field0, copy=True)
