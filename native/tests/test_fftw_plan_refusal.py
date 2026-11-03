# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A mistyped FFTW planning effort must be refused (SEP 43 inventory, item 73).

``numba_cpu`` refused ``flag="fastest"``. Native takes the effort from
``SOLPROP_FFTW_PLAN`` and reads anything it doesn't know as ``measure``
(``fftw_seed_effort`` in ``include/solprop_native/fft_plan.hpp``), so a typo changes a
run's plans in silence. The neighbouring thread variables are refused in a fresh process
(``test_invalid_thread_variable_is_rejected_in_a_fresh_process``); this holds the
planning effort to the same.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

SCRIPT = """
import solprop_native
from solprop.grid import EnvelopeGrid
from solprop.nonlinear.gnlse import problem_from
from solprop.pulses import SechPulse
from solprop.stepper import PIController
from solprop.translate.problem import assemble

grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 1e-12)
model = problem_from(
    [0.0, 0.0, -11.83e-27], omega_ref=grid.omega0, gamma=0.11, shock=False
)
problem = assemble(
    grid,
    model.mode,
    [model.kerr],
    SechPulse(wavelength=835e-9, width=100e-15, peak_power=1.0).build(grid),
    save_planes=[1e-3],
    density=1.0,
    shock=False,
    prefactor_index=1.0,
    controller=PIController(rtol=1e-8, atol=1e-16),
    dz_max=1e-3,
    dz_init=1e-4,
)
solprop_native.NativeBackend(problem, grid=grid).close()
"""


@pytest.mark.xfail(
    strict=True,
    reason=(
        "SEP 43 inventory: native reads a mistyped SOLPROP_FFTW_PLAN as measure, "
        "silently (item 73)"
    ),
)
def test_an_unknown_planning_effort_is_refused() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "SOLPROP_FFTW_PLAN": "fastest"},
    )
    assert completed.returncode != 0
    assert "SOLPROP_FFTW_PLAN" in completed.stderr


def test_the_planning_efforts_it_names_are_admitted() -> None:
    """The script runs under the known efforts, so a refusal above would be real."""
    for effort in ("estimate", "measure", "patient"):
        completed = subprocess.run(
            [sys.executable, "-c", SCRIPT],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "SOLPROP_FFTW_PLAN": effort},
        )
        assert completed.returncode == 0, completed.stderr
