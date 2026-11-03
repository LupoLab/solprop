# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A compact burst on the band column, at the team sizes it can split among (SEP 43).

``numba_cpu``'s fraction-reuse thread test
(``test_fraction_reuse_always_evaluates_first_sample_and_is_thread_invariant``) held its
reusing column, on a field zero but for a burst and a preionisation of 1e-20, at 1 to 18
threads, two calls each. Native's column reuses too (``plasma.hpp``: "an unchanged
exposure its fraction"), and a team splits each row of its one node, so a reused sample
can start a chunk: a thread must not take a fraction its predecessor wrote.

**Weaker than the retired test in one way.** That test wrote the column's input
directly, so its zeros were exact. Native's column reads a synthesis, so here the
field's quiet samples are small rather than zero; the rate, a quadratic above a
threshold, is what is exactly zero there, and it is the exposure's being unchanged that
the reuse acts on. The retired test's other half, that a one-sample column keeps a
preionisation of 1e-20, has no input here: no time grid has one sample.

**The bounds, fixed from the operations.** Field and statistics against the first
width at :func:`test_band_plasma.march_bound` and :func:`assert_statistics_agree`, the
reference comparison's own, as ``test_the_band_column_agrees_across_thread_counts``
holds the others, and against the reference at the same bound.
"""

# ruff: noqa: I001 -- Kokkos loads first, before any other OpenMP runtime on macOS.

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, cast

import solprop_native

import numpy as np
import pytest
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import Simulation
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import FieldGrid
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, RungeKutta
from solprop.translate.problem import KernelProblem

from legacy_geometry import LEGACY_GRID
from test_band_plasma import (
    STATISTICS,
    QuadraticRate,
    assert_statistics_agree,
    march,
    march_bound,
)
from test_band_stress import LENGTH, THRESHOLD, TRANGE, node_fields

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)

INFO = solprop_native.build_info()
STEPS = 2
"""Fixed steps of the march, m: the column's rows and their exposure are what matter."""
PREIONISATION = 1e-20
"""The retired test's: small enough that a fraction's rounding would show."""
PROBE = Path(__file__).with_name("band_burst_thread_probe.py")
WIDTHS = (1, 4, 8, 12, 18)
"""The retired test's thread counts, oversubscribed where a machine has fewer."""


def _burst(steps: int, energy: float) -> tuple[FieldGrid, KernelProblem]:
    rate = QuadraticRate(2e10, THRESHOLD)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        simulation = Simulation.from_capillary(
            125e-6,
            LENGTH,
            "Ar",
            1.0,
            plasma=rate,
            ionisation_field_range=(1e3, 1e11),
            preionisation=PREIONISATION,
            wavelength=800e-9,
            grid=FieldGrid.from_limits(
                LENGTH, 800e-9, (200e-9, 2e-6), TRANGE, **LEGACY_GRID
            ),
            pulses=[GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=energy)],
            n_r=8,
            n_save=2,
            plasma_integration="band",
            integrator=RungeKutta(),
            schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / steps),
            absorber_scheme="split_step",
            noise=None,
        )
        # Mode-averaged: one node, so a host team of several threads splits each row.
        simulation.settings = dataclasses.replace(
            simulation.settings, quadrature=None, components=None
        )
        problem = simulation.translate()
    grid = simulation.grid
    assert isinstance(grid, FieldGrid)
    return grid, dataclasses.replace(problem, stats_spec=StatsSpec(names=STATISTICS))


def burst(steps: int) -> tuple[FieldGrid, KernelProblem]:
    """Return a one-node argon column whose rate is nonzero on a few cycle peaks alone.

    The energy is scaled, as ``near_threshold``'s is, to put the launch's peak field
    half as far again as the quadratic rate's threshold, so the rate is exactly zero
    outside a compact burst and the pulse is still over the threshold at the exit.
    """
    grid, problem = _burst(steps, 10e-6)
    peak = float(np.max(np.abs(node_fields(problem, grid))))
    return _burst(steps, 10e-6 * (1.5 * THRESHOLD / peak) ** 2)


def _run_at(threads: int, directory: Path) -> dict[str, np.ndarray]:
    """Run the probe with the kernels forced to `threads`, in a fresh process."""
    out = directory / f"burst-{threads}.npz"
    completed = subprocess.run(
        [sys.executable, str(PROBE), str(out)],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "OMP_PROC_BIND": "false",
            "SOLPROP_NATIVE_THREADS": str(threads),
            "SOLPROP_OVERSUBSCRIBE": "1",
            "SOLPROP_THREADS": "",
            "SOLPROP_NATIVE_KERNEL_THREADS": str(threads),
        },
    )
    assert completed.returncode == 0, completed.stderr
    record = cast(dict[str, Any], json.loads(completed.stdout.splitlines()[-1]))
    assert record["threads"] == threads
    # One node: the response's team is the pool, so the row is really split among them.
    assert (
        max(
            choice["threads"]
            for name, choice in record["sizing"].items()
            if name.startswith("kernels.response.")
        )
        == threads
    )
    with np.load(out) as stored:
        return {name: stored[name] for name in stored.files}


def _statistics(run: dict[str, np.ndarray], call: int) -> dict[str, np.ndarray]:
    return {name: run[f"{name}_{call}"] for name in STATISTICS}


@pytest.mark.skipif(
    INFO.device_backend != "none", reason="a device build's host space is serial"
)
def test_a_compact_burst_agrees_at_every_team_size_and_on_repeat(
    tmp_path: Path,
) -> None:
    """Five team sizes, two calls each, against one thread and the reference."""
    grid, problem = burst(STEPS)
    bound = march_bound(problem, grid, STEPS)
    runs = {threads: _run_at(threads, tmp_path) for threads in WIDTHS}
    expected = runs[1]["field_0"]
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    reference, reference_stats = march(NumpyRefBackend, problem, grid)
    assert float(np.max(np.abs(expected - reference))) <= bound * scale
    assert_statistics_agree(_statistics(runs[1], 0), reference_stats, bound)
    # The burst did ionise at the exit, where a preionisation of 1e-20 alone would round
    # to a fraction of exactly zero.
    density = np.asarray(reference_stats["electron_density"], dtype=float)
    assert np.max(density) > 0.0
    for threads, run in runs.items():
        for call in (0, 1):
            field = run[f"field_{call}"]
            assert float(np.max(np.abs(field - expected))) <= bound * scale, (
                threads,
                call,
            )
            assert_statistics_agree(
                _statistics(run, call), _statistics(runs[1], 0), bound
            )
