# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Focused CUDA/HIP bring-up checks for the optional native backend."""

import numpy as np
import pytest

from bounds import assert_within, ladder_bound
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.stepper import FixedSchedule

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def accelerator_backend():
    """Return a coherent installed accelerator backend or skip the GPU lane."""
    native = pytest.importorskip("solprop_native")
    info = native.build_info()
    if info.device_backend == "none":
        pytest.skip("the installed solprop-native build has no accelerator backend")
    native.check_build()
    assert info.device_backend in {"cuda", "hip"}
    assert info.execution_space in {"Cuda", "HIP"}
    assert info.device_count == 1
    assert info.fftw_threads == 0
    assert info.fft_backend in {"cuFFT", "rocFFT"}
    return native.NativeBackend


def _field(backend_class, problem, grid):
    backend = backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        summary = backend.propagate_to(1e-3)
        return backend.observe().require_field().copy(), summary
    finally:
        backend.close()


def test_one_device_step_meets_the_predeclared_fft_bound(
    accelerator_backend, soliton_problem, soliton_grid
):
    """Exercise one complete seven-stage device trial against the NumPy oracle."""
    problem = soliton_problem(
        soliton_grid,
        3,
        save_planes=[1e-3],
        schedule=FixedSchedule.uniform(0.0, 1e-3, 1e-3),
    )
    actual, summary = _field(accelerator_backend, problem, soliton_grid)
    expected, reference = _field(NumpyRefBackend, problem, soliton_grid)
    deviation = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    bound = ladder_bound("native", 16 * np.finfo(float).eps, soliton_grid.n_to, 7)
    assert summary.n_steps == reference.n_steps == 1
    assert deviation < bound


def test_same_card_launch_geometry_is_repeatable(
    accelerator_backend, soliton_problem, soliton_grid
):
    """Two fresh states on one unchanged build agree within the step's device bound.

    The same one-step computation twice; a device's reductions and transforms may
    order their sums differently from launch to launch, so the two are held to the
    one-step rung's predeclared device envelope (:func:`bounds.ladder_bound`).
    """
    problem = soliton_problem(
        soliton_grid,
        3,
        save_planes=[1e-3],
        schedule=FixedSchedule.uniform(0.0, 1e-3, 1e-3),
    )
    first, first_summary = _field(accelerator_backend, problem, soliton_grid)
    second, second_summary = _field(accelerator_backend, problem, soliton_grid)
    # A prescribed one-step schedule: its work is accounting, exact.
    assert first_summary.n_steps == second_summary.n_steps
    assert first_summary.n_rejected == second_summary.n_rejected
    bound = ladder_bound("native", 16 * np.finfo(float).eps, soliton_grid.n_to, 7)
    assert_within(second, first, bound, what="repeat launch")
