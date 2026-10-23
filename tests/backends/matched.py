# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Matched discrete computations for the backend tests (SEP 25 class R).

Two runs compared at a rounding bound must take the same discrete method: the same
steps, orders, startup and history. A prescribed :class:`FixedSchedule` fixes the
steps, but a variable-order Adams method still chooses each step's order from its
defects, and a rounding difference near the hysteresis can choose differently.
:func:`fixed_order` pins the order, so the two runs share every decision and differ
by rounding alone.

The Kokkos runtime reads its thread count once per process, so a one-against-many
thread comparison runs a probe script in fresh processes; :func:`native_thread_fields`
runs one and reads back the arrays it saved.

The rest are rounding bounds several modules share, each fixed from what the two
sides compute: :func:`stash_bound` for a stashed plasma scan against a fresh one,
:func:`dispersive_response_bound` for two evaluations of SEP 6's cubic, and
:func:`assert_columns_agree` for two specialisations of a compiled response column.
"""

import dataclasses
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from bounds import assert_within, fft_bound, rate_log_sensitivity, scan_bound
from solprop.stepper.multistep import Multistep
from solprop.translate.problem import KernelProblem


def fixed_order(problem: KernelProblem) -> KernelProblem:
    """Return `problem` with its Adams method held at its top order, if it has one.

    The Runge-Kutta march (``integrator=None``) has no order to choose and is returned
    unchanged.
    """
    method = problem.stepper.integrator
    if not isinstance(method, Multistep) or not method.variable_order:
        return problem
    return dataclasses.replace(
        problem,
        stepper=dataclasses.replace(
            problem.stepper,
            integrator=dataclasses.replace(method, variable_order=False),
        ),
    )


def native_thread_fields(
    probe: Path, threads: int, names: Sequence[str], directory: Path
) -> dict[str, np.ndarray]:
    """Run a thread probe at one Kokkos thread count and return its saved arrays.

    The probe takes the path of an ``.npz`` file to write and then its case names, and
    saves one array per case under the case's position (``case0``, ``case1``, ...).
    """
    environment = os.environ | {
        "SOLPROP_NATIVE_THREADS": str(threads),
        "OMP_PROC_BIND": "false",
    }
    output = directory / f"{probe.stem}-{threads}-{len(list(directory.iterdir()))}.npz"
    subprocess.run(
        [sys.executable, str(probe), str(output), *names],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        cwd=probe.parents[1],
    )
    with np.load(output) as saved:
        return {name: saved[f"case{index}"] for index, name in enumerate(names)}


def stash_bound(rate, peak_rate: float, field_range: tuple[float, float], n_to: int):
    """Bound a stashed plasma scan against the same state's fresh scan, relatively.

    The two read one state synthesised twice, so their node fields differ by one
    transform round trip, :func:`bounds.fft_bound`. The largest rate moves by the
    rate's log-sensitivity ``S`` at `peak_rate` times that
    (:func:`bounds.rate_log_sensitivity`), and the final fraction, ``1 - (1 - p)
    exp(-X)``, by no more than its exposure ``X``, which adds the scan chain's own
    rounding, :func:`bounds.scan_bound`. `rate` maps a field amplitude in V/m to a
    rate in 1/s, monotone over `field_range`.
    """
    sensitivity = max(1.0, rate_log_sensitivity(rate, peak_rate, field_range))
    return sensitivity * fft_bound(n_to) + scan_bound(n_to)


def dispersive_response_bound(n_to: int, evaluator: str, channels: int = 1) -> float:
    """Bound two evaluations of SEP 6's dispersive cubic on one field, relatively.

    The count of ``test_numba_dispersive_kerr.py``'s module docstring: per channel the
    symmetric evaluator takes three transforms (the filtered field, its product, the
    output) and the polynomial six, each ``8 eps ceil(log2 n_to)``; the two sides'
    budgets are added. Two evaluations that differ only in where their multipliers
    come from, or in the order a pass groups its nodes, are within it.
    """
    transforms = (3 if evaluator == "symmetric" else 6) * channels
    return 2 * transforms * 8 * float(np.finfo(float).eps) * math.ceil(math.log2(n_to))


def assert_columns_agree(actual, expected, samples: int) -> None:
    """Hold two variants of a compiled response column to the column chain's budget.

    Two specialisations of one column -- a reused fraction, an omitted gather, a
    thread count -- are different compiled code over the same arithmetic, which a
    compiler may contract and order differently (SEP 25). Each output array (the
    polarisation, the Raman drive, the stash) is held normwise to the complete plasma
    chain's relative cap, :func:`bounds.scan_bound`; an output the column leaves at
    zero stays exactly zero.
    """
    for values, reference in zip(actual, expected, strict=True):
        assert np.all(np.isfinite(values))
        if np.any(reference):
            assert_within(values, reference, scan_bound(max(2, samples)))
        else:
            np.testing.assert_array_equal(values, reference)
