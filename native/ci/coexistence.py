# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check that an installed solprop-native wheel coexists with the scientific stack.

The wheel carries its own OpenMP runtime, FFTW and (on Linux and Windows) OpenBLAS.
NumPy, SciPy, pyFFTW and Numba carry theirs. This script imports them in both
orders, each scenario in a fresh process, starts every package's thread pool (a
threaded pyFFTW transform, a NumPy and a SciPy BLAS call, a parallel Numba loop), then
runs one short propagation on the native backend, one on ``numba_cpu`` and the native
one again, interleaved in the same process, and compares them::

    python -I native/ci/coexistence.py [--bypass-guard]

Scenarios:

``native-first``
    ``solprop_native`` first, then NumPy, SciPy, pyFFTW and Numba.
``stack-first``
    NumPy, SciPy, pyFFTW and Numba first, then ``solprop_native``. On macOS the
    package refuses this order by design (pyFFTW's wheel carries another OpenMP
    runtime); the refusal is the expected outcome there.
``solprop``
    ``import solprop`` and run, as a user script does: the backends import their
    packages when they are first used.
``stack-first-unguarded`` (only with ``--bypass-guard``, an investigation)
    as ``stack-first``, with pyFFTW hidden from ``sys.modules`` while
    ``solprop_native`` imports, so that the macOS refusal does not trigger. Its
    outcome is reported and never fails the script.

The exit status is non-zero when a required scenario fails. Each scenario prints one
JSON line; the summary follows.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

GUARD_MESSAGE = "import solprop_native before pyfftw"
STACK = ("numpy", "scipy.linalg", "scipy.fft", "pyfftw", "numba")


def _import_stack() -> None:
    import importlib

    for name in STACK:
        importlib.import_module(name)


def _import_native(*, hide_pyfftw: bool) -> None:
    hidden = {}
    if hide_pyfftw:
        hidden = {k: v for k, v in sys.modules.items() if k.split(".")[0] == "pyfftw"}
        for name in hidden:
            del sys.modules[name]
    try:
        import solprop_native  # noqa: F401
    finally:
        sys.modules.update(hidden)


def _exercise_stack() -> dict[str, float]:
    """Start each package's thread pool and check one result from each."""
    import numba
    import numpy as np
    import pyfftw
    import pyfftw.builders
    import scipy.linalg

    rng = np.random.default_rng(29)
    signal = rng.standard_normal(1 << 16) + 1j * rng.standard_normal(1 << 16)
    transform = pyfftw.builders.fft(signal, threads=4, planner_effort="FFTW_ESTIMATE")
    fft_error = float(
        np.max(np.abs(transform() - np.fft.fft(signal))) / np.max(np.abs(signal))
    )
    matrix = rng.standard_normal((512, 512)) + 512 * np.eye(512)
    rhs = rng.standard_normal(512)
    solution = scipy.linalg.solve(matrix, rhs)
    blas_residual = float(np.max(np.abs(matrix @ solution - rhs)))

    @numba.njit(parallel=True, cache=False)
    def parallel_sum(values):  # pragma: no cover - compiled
        total = 0.0
        for index in numba.prange(values.size):
            total += values[index]
        return total

    values = rng.standard_normal(1 << 20)
    numba_error = float(
        abs(parallel_sum(values) - values.sum()) / np.sum(np.abs(values))
    )
    # Each bound is a few hundred ulps of the operation's scale: these check that the
    # packages compute at all beside the native runtime, not their accuracy.
    if not (fft_error < 1e-12 and blas_residual < 1e-10 and numba_error < 1e-12):
        raise AssertionError((fft_error, blas_residual, numba_error))
    return {
        "pyfftw_error": fft_error,
        "blas_residual": blas_residual,
        "numba_error": numba_error,
    }


def _propagate(backend: str):
    """Run one short capillary propagation and return the output field (optics)."""
    import numpy as np
    from solprop.driver import prop_capillary

    output = prop_capillary(
        50e-6,  # core radius, m
        0.01,  # length, m: a few adaptive steps
        "He",
        1.0,  # bar
        lambda0=800e-9,
        fwhm=30e-15,
        energy=5e-6,
        trange=400e-15,
        wavelength_limits=(200e-9, 4e-6),
        n_save=2,
        backend=backend,
        # Noiseless: SEP 37's default draws a fresh realisation every run, and the
        # repeat below must agree to rounding.
        noise=None,
    )
    field = np.asarray(output["field"])[-1]
    if not np.all(np.isfinite(field)):
        raise AssertionError(f"{backend}: non-finite field")
    return field


def _interleaved_runs() -> dict[str, float]:
    import numpy as np

    native = _propagate("native")
    numba_cpu = _propagate("numba_cpu")
    native_again = _propagate("native")
    scale = float(np.linalg.norm(native))

    def difference(a, b):
        return float(np.linalg.norm(a - b) / scale)

    result = {
        "native_vs_numba_cpu": difference(native, numba_cpu),
        "native_repeat": difference(native, native_again),
    }
    # Two adaptive runs of one problem on two backends agree to the controller's
    # accuracy envelope, not to rounding (SEP 25); 1e-5 is far above that envelope at
    # this tolerance and far below what a corrupted runtime produces. A repeat on the
    # same backend differs only by reduction order.
    if not (result["native_vs_numba_cpu"] < 1e-5 and result["native_repeat"] < 1e-9):
        raise AssertionError(result)
    return result


def child(scenario: str) -> None:
    """Run one scenario in this (fresh) process and print its JSON result."""
    started = time.perf_counter()
    result: dict[str, object] = {"scenario": scenario}
    try:
        if scenario == "native-first":
            _import_native(hide_pyfftw=False)
            _import_stack()
        elif scenario == "stack-first":
            _import_stack()
            _import_native(hide_pyfftw=False)
        elif scenario == "stack-first-unguarded":
            _import_stack()
            _import_native(hide_pyfftw=True)
        elif scenario != "solprop":
            raise ValueError(scenario)
        if scenario != "solprop":
            result["stack"] = _exercise_stack()
        result["runs"] = _interleaved_runs()
        if scenario == "solprop":
            result["stack"] = _exercise_stack()
        result["outcome"] = "passed"
    except RuntimeError as error:
        if GUARD_MESSAGE not in str(error):
            raise
        result["outcome"] = "refused"
        result["message"] = str(error)
    result["seconds"] = round(time.perf_counter() - started, 1)
    sys.stdout.write(json.dumps(result) + "\n")


def run_scenario(scenario: str) -> dict[str, object]:
    """Run a scenario in a fresh interpreter and collect its result."""
    completed = subprocess.run(
        # faulthandler: a crash in native code still leaves the Python stack.
        [
            sys.executable,
            "-I",
            "-X",
            "faulthandler",
            str(Path(__file__).resolve()),
            "--child",
            scenario,
        ],
        capture_output=True,
        text=True,
        env=os.environ,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if completed.returncode == 0 and lines:
        return json.loads(lines[-1])
    return {
        "scenario": scenario,
        "outcome": "failed",
        "returncode": completed.returncode,
        "stderr_tail": completed.stderr[-3000:],
    }


def main() -> None:
    """Run every scenario and report; exit non-zero if a required one fails."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument(
        "--bypass-guard",
        action="store_true",
        help="also run stack-first with the macOS pyFFTW refusal bypassed (a report)",
    )
    arguments = parser.parse_args()
    if arguments.child:
        child(arguments.child)
        return

    expected_refusal = sys.platform == "darwin"
    scenarios = ["native-first", "stack-first", "solprop"]
    if arguments.bypass_guard:
        scenarios.append("stack-first-unguarded")
    failed = []
    for scenario in scenarios:
        result = run_scenario(scenario)
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()
        outcome = result["outcome"]
        if scenario == "stack-first-unguarded":
            continue
        if scenario == "stack-first" and expected_refusal:
            ok = outcome == "refused"
        else:
            ok = outcome == "passed"
        if not ok:
            failed.append(scenario)
    if failed:
        sys.exit(f"coexistence failed: {', '.join(failed)}")
    sys.stdout.write("coexistence passed\n")


if __name__ == "__main__":
    main()
