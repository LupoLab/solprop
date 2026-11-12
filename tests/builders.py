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
from typing import Any

import numpy as np

from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.grid import TimeGrid
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest

__all__ = [
    "HASH_PROBE",
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
        np.ones((1, *problem.shape), complex),
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
