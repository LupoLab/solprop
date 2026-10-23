# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Synthetic runs for the processing tests.

A Gaussian carried along z by the closed-form phase of pure group-velocity dispersion,
stored in a memory handler and, plane for plane, in an HDF5 file. The dispersion is
chosen so that the pulse doubles its duration over the length, so every temporal
quantity has an analytic value to compare with.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from solprop.grid import EnvelopeGrid, FieldGrid, TimeGrid
from solprop.output import HDF5Output, MemoryOutput, open_file
from solprop.output.schema import (
    GridRecord,
    ModeRecord,
    RunMetadata,
    SavedPlane,
    StepperRecord,
)
from solprop.pulses import GaussianPulse
from solprop.stats import collect, default_spec

WAVELENGTH = 800e-9
FWHM = 30e-15
ENERGY = 1e-6
LENGTH = 0.1
TRANGE = 400e-15
LIMITS = (400e-9, 2e-6)
N_Z = 5

# The 1/e half-width of the field of a Gaussian whose intensity full width at half
# maximum is FWHM; the broadening law is written in terms of it.
TAU0 = FWHM / (2.0 * np.sqrt(np.log(2.0)))

# tau(z) = tau0 sqrt(1 + (beta2 z / tau0^2)^2), so this doubles the duration at z = L.
BETA2 = np.sqrt(3.0) * TAU0**2 / LENGTH


def build_grid(kind: str) -> TimeGrid:
    if kind == "field":
        return FieldGrid.from_limits(LENGTH, WAVELENGTH, LIMITS, TRANGE)
    return EnvelopeGrid.from_limits(LENGTH, WAVELENGTH, LIMITS, TRANGE)


def build_run(
    kind: str = "envelope",
    *,
    n_modes: int = 1,
    n_pol: int = 1,
    t0: float = 0.0,
    beta2: float = BETA2,
    with_stats: bool = True,
) -> MemoryOutput:
    grid = build_grid(kind)
    pulse = GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY, t0=t0)
    weights = np.array(
        [[1.0 / ((m + 1) * (p + 1)) for p in range(n_pol)] for m in range(n_modes)]
    )
    seed = pulse.build(grid)[0, 0] * weights[:, :, np.newaxis]
    simulation_kind = "multimode" if n_modes > 1 else "mode_averaged"
    metadata = RunMetadata(
        grid=GridRecord.from_grid(grid),
        modes=tuple(
            ModeRecord(
                kind="solprop.modes.capillary.MarcatiliMode",
                effective_area=0.5e-8 / (index + 1),
                configuration={"index": index},
            )
            for index in range(n_modes)
        ),
        simulation_kind=simulation_kind,
        configuration={"beta2": beta2},
        stepper=StepperRecord(tableau="DOPRI5", controller={"rtol": 1e-8}),
    )
    spec = default_spec(simulation_kind)
    output = MemoryOutput()
    output.open(metadata)
    for z in np.linspace(0.0, LENGTH, N_Z):
        # The internal linear operator of a purely dispersive frame (conventions.md 2).
        phase = np.exp(-0.5j * beta2 * (grid.omega - grid.omega0) ** 2 * z)
        plane = SavedPlane(z=float(z), field=seed * phase)
        stats = collect(spec, grid, plane) if with_stats else {}
        output.save(SavedPlane(z=float(z), field=seed * phase, stats=stats))
    return output


def as_file(memory: MemoryOutput, path) -> HDF5Output:
    """Write the run to a file and open that file again, read-only."""
    memory.to_file(path)
    return open_file(path)


@dataclass(frozen=True)
class Synthetic:
    """The parameters of the test run, and the builders that make one."""

    wavelength: float = WAVELENGTH
    fwhm: float = FWHM
    energy: float = ENERGY
    length: float = LENGTH
    trange: float = TRANGE
    n_z: int = N_Z
    tau0: float = TAU0
    beta2: float = BETA2

    def build_grid(self, kind: str = "envelope") -> TimeGrid:
        return build_grid(kind)

    def build_run(self, kind: str = "envelope", **kwargs) -> MemoryOutput:
        return build_run(kind, **kwargs)

    def as_file(self, memory: MemoryOutput, path) -> HDF5Output:
        return as_file(memory, path)

    def broadened_fwhm(self, z: float) -> float:
        """Return the analytic intensity width of the test pulse at a position, s."""
        return self.fwhm * np.sqrt(1.0 + (self.beta2 * z / self.tau0**2) ** 2)


@pytest.fixture
def synthetic():
    return Synthetic()


@pytest.fixture(params=["envelope", "field"])
def kind(request):
    return request.param


@pytest.fixture
def grid(kind):
    return build_grid(kind)


@pytest.fixture
def memory(kind):
    return build_run(kind)


@pytest.fixture
def handlers(memory, tmp_path):
    """The same run twice: as a memory handler and as an HDF5 handler."""
    handler = as_file(memory, tmp_path / "run.h5")
    yield memory, handler
    handler.close()


@pytest.fixture(params=["memory", "file"])
def output(handlers, request):
    """The run through one handler or the other; every accessor test runs on both."""
    return handlers[0] if request.param == "memory" else handlers[1]
