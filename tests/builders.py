# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared builders for tests: construction code several test modules repeated.

A builder here replaces copies that were the same code, so a migrated test builds
exactly what it built before; a builder that assembles a problem or a simulation was
pinned, when it was migrated, by the configuration hash of what the old copies built.
"""

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Literal

import numpy as np

from legacy_geometry import LEGACY_GRID, bare_absorber
from march_bounds import FIXED_ORDER
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import Simulation
from solprop.driver.scan import ScanPoint
from solprop.grid import EnvelopeGrid, TimeGrid
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest

__all__ = [
    "HASH_PROBE",
    "fixed_fibre_point",
    "launch_derivative",
    "numpy_ref_launch_derivative",
    "run_python",
]

HASH_PROBE = textwrap.dedent(
    """\
    from solprop.driver.simulation import Simulation

    def run(self, *args, **kwargs):
        print(self.metadata().configuration_hash())
        raise SystemExit(0)

    Simulation.run = run
    """
)
"""A prefix for an exported script: its run prints the configuration hash and exits."""


def launch_derivative(backend: NumpyRefBackend, problem: PropagationProblem) -> Any:
    """Return an open ``numpy_ref`` backend's stage derivative ``N(E, z=0)`` at launch.

    The backend is loaded with the problem's launch field, hoisted to z = 0 and asked
    for its nonlinear stage there with unit stage weights, as the first stage of a
    trial is.
    """
    backend.set_initial(problem.initial.field0, None)
    backend._hoist(0.0)
    return backend._nonlinear(
        backend._require_state(),
        backend._stage_scalars(0.0, 0.0),
        np.ones(problem.shape, complex),
    )


def numpy_ref_launch_derivative(
    problem: PropagationProblem, run_request: RunRequest, grid: TimeGrid
) -> Any:
    """Open ``numpy_ref`` on the problem, return :func:`launch_derivative`, close it."""
    backend = NumpyRefBackend(problem, run_request, grid=grid)
    try:
        return launch_derivative(backend, problem)
    finally:
        backend.close()


def run_python(path: Path) -> "subprocess.CompletedProcess[str]":
    """Run a written script with this interpreter in its own folder, as a person would.

    A fresh process, so without the tests' type hook; asserts it exits cleanly, showing
    its standard error if not, and returns the completed process.
    """
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=path.parent,
    )
    assert result.returncode == 0, result.stderr
    return result


def fixed_fibre_point(
    point: ScanPoint,
    backend: str = "numpy_ref",
    noise: Literal["input", "modified"] = "modified",
) -> Simulation:
    """Build a scan point's millimetre of fibre on a fixed four-step schedule.

    The point's index scales the nonlinearity, the dispersion and the peak power; its
    seed draws the noise. The former grid without a collar: a short fixed schedule is
    refused beside the default collar, and the scan tests that build it are about
    scans, not boundaries.
    """
    grid = EnvelopeGrid.from_limits(0.001, 835e-9, (500e-9, 2e-6), 1e-12, **LEGACY_GRID)
    return Simulation.from_gnlse(
        0.11 * (point.index + 1),
        0.001,
        [0.0, 0.0, -1.183e-26 * (point.index + 1)],
        wavelength=835e-9,
        grid=grid,
        absorber=bare_absorber(grid),
        width=100e-15,
        peak_power=float(point.index + 1),
        shape="sech",
        save_planes=[0.0005, 0.001],
        schedule=FixedSchedule.uniform(
            0.0, 0.001, 0.00025, save_planes=[0.0005, 0.001]
        ),
        stats=StatsSpec(names=("energy", "peak_power"), periodic=("energy",), period=2),
        noise=noise,
        seed=point.seed,
        # The fixed order leaves no controller decision on the prescribed schedule.
        integrator=FIXED_ORDER,
        backend=backend,
    )
