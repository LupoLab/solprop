# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run a case on ``native`` with every thread pool forced, in a fresh process.

Native sizes each operation by the work it holds (``operation_threads`` in
``native/src/core.cpp``), so a small case at ``SOLPROP_NATIVE_THREADS=4`` runs most of
its kernels on one thread and compares one thread with one thread. An explicit pool
bypasses that rule, which is what ``numba_cpu``'s ``regime="large"`` was for. So
:func:`forced_run` names every pool, and a test reads the widths actually taken from the
run's ``execution_sizing`` through :func:`widest` and asserts them. A comparison whose
threads never ran would otherwise pass.

The Kokkos runtime reads its pools once per process (``matched.py``), so each width is a
fresh process running ``native_forced_thread_probe.py``; :func:`forced_run` returns the
arrays that probe saved, its sizing and the Kokkos pool it ran with.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PROBE = Path(__file__).with_name("native_forced_thread_probe.py")
"""The subprocess that runs one case of ``forced_thread_cases.py``."""

POOLS = (
    "SOLPROP_NATIVE_KERNEL_THREADS",
    "SOLPROP_NATIVE_FFT_THREADS",
    "SOLPROP_NATIVE_RAMAN_FFT_THREADS",
    "SOLPROP_NATIVE_RAMAN_TAIL_FFT_THREADS",
)
"""Every pool an explicit request takes out of the size rule, ``_pool_request``."""


def thread_environment(threads: int, pools: dict[str, int | str] | None) -> dict:
    """Return the environment of a probe at a width, every pool forced by default.

    `pools` replaces the forced pools; ``{}`` leaves the size rule in charge, and
    ``SOLPROP_NATIVE_THREADS`` still sets the capacity. The lanes' one-thread budget is
    lifted and oversubscription allowed (ADR-052).
    """
    requested = dict.fromkeys(POOLS, threads) if pools is None else pools
    inherited = {
        name: value
        for name, value in os.environ.items()
        if name not in POOLS and name != "SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES"
    }
    return inherited | {
        "SOLPROP_NATIVE_THREADS": str(threads),
        "SOLPROP_THREADS": "",
        "SOLPROP_OVERSUBSCRIBE": "1",
        "OMP_PROC_BIND": "false",
        **{name: str(value) for name, value in requested.items()},
    }


def forced_run(
    case: str,
    threads: int,
    directory: Path,
    *,
    pools: dict[str, int | str] | None = None,
) -> dict:
    """Run `case` at `threads` in a fresh process and return what it saved.

    The result holds the case's arrays, ``"sizing"`` (the backend's
    ``execution_sizing``, one dict per backend the case built, merged by key) and
    ``"pool"`` (the Kokkos runtime's width, asserted to be `threads` so that a silent
    cap fails here rather than comparing one thread with one thread).
    """
    output = directory / f"forced-{len(list(directory.iterdir()))}.npz"
    completed = subprocess.run(
        [sys.executable, str(PROBE), str(output), case],
        check=False,
        capture_output=True,
        text=True,
        env=thread_environment(threads, pools),
        cwd=PROBE.parents[1],
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    with np.load(output) as saved:
        result = {name: saved[name] for name in saved.files}
    result["sizing"] = json.loads(str(result["sizing"]))
    assert int(result["pool"]) == threads
    return result


def widest(sizing: dict[str, dict[str, int]], operation: str) -> int:
    """Return the most threads any execution of `operation` took, ``"pool.op"``.

    An operation that never ran gives 0, so ``widest(...) == 4`` fails on a route that
    stopped reaching it.
    """
    return max(
        (
            choice["threads"]
            for name, choice in sizing.items()
            if name.startswith(f"{operation}.")
        ),
        default=0,
    )
