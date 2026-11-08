# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A native call on any Python thread runs OpenMP at the runtime's width (issue #130).

Kokkos sets the OpenMP default width only on the thread that initialised it. On any
other thread it is every CPU, which an OpenMP OpenBLAS and FFTW's unnamed loops read,
so a run on a worker thread switched GNU OpenMP between two widths at every pair. A
native call's scope sets the calling thread's default for its duration.
"""

from __future__ import annotations

import os
import threading

import pytest

import solprop_native
from solprop_native import _core


def _on_a_thread() -> int:
    seen: list[int] = []
    worker = threading.Thread(target=lambda: seen.append(_core.scoped_openmp_default()))
    worker.start()
    worker.join()
    return seen[0]


def test_a_native_call_sees_the_runtime_width_on_every_thread():
    width = solprop_native.build_info().runtime_threads
    if solprop_native.build_info().openmp_library is None:
        assert _core.scoped_openmp_default() == 1
        return
    assert _core.scoped_openmp_default() == width
    # A Python thread starts from the runtime's process-wide default (every CPU unless
    # OMP_NUM_THREADS says otherwise), which the scope replaces and then restores.
    assert _on_a_thread() == width
    assert (os.process_cpu_count() or 1) >= width


def test_llvm_threads_spin_briefly_inside_native_calls_only():
    """On LLVM's runtime the core spins idle threads inside a call (ADR-071)."""
    inside = _core.scoped_block_time()
    if inside < 0:
        pytest.skip("GNU's runtime reads its spin count once, at start")
    expected = solprop_native._WAIT.spin_milliseconds
    if expected == 0:
        pytest.skip("this process's wait policy keeps the runtime's block time")
    assert inside == expected
