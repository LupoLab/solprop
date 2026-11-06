# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Focused CUDA/HIP bring-up checks for the optional native backend."""

from typing import Any, cast

import numpy as np
import pytest
from benchmarks.device_split import STEP, _simulation

from bounds import assert_within, ladder_bound
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import select
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


# The probe's band reaches below the glass's tabulated range: not the point here.
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_a_large_state_crosses_both_ways_through_the_ring_bit_for_bit(
    accelerator_backend,
):
    # The ring needs 256 MiB; 192^2 x 600 is 354 MiB. A lossless copy is exact (SEP 25).
    del accelerator_backend
    simulation = _simulation(192, 2, 1, None)
    problem = simulation.translate()
    engine = cast(
        "Any",
        select(
            "native",
            problem,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        ),
    )
    try:
        rng = np.random.default_rng(53)
        shape = problem.shape
        field = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        # A load masks outside the active band, which isn't what this test is about.
        field[..., ~np.asarray(problem.grids.active_mask)] = 0.0
        assert field.nbytes >= 2**28
        assert engine.memory_ledger()["host_crossing_ring"] == 0
        engine.set_initial(field, None, z=problem.initial.z)
        assert engine.memory_ledger()["host_crossing_ring"] > 0
        back = engine._native_state.field_to_host().reshape(shape)
        assert back.tobytes() == field.tobytes()
        observed = engine.observe(field=True).require_field()
        assert observed.tobytes() == field.tobytes()
    finally:
        engine.close()


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_the_plans_of_the_scratch_hold_one_work_area(accelerator_backend):
    # A forward and an inverse rank-3 plan take a state's work area each at this size,
    # on their own (solprop #529). Run a step on it, so that a shared area can't be
    # one no transform has used, and read the field the transforms left.
    del accelerator_backend
    simulation = _simulation(512, 2, 1, None)
    problem = simulation.translate()
    engine = cast(
        "Any",
        select(
            "native",
            problem,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        ),
    )
    try:
        state = int(np.prod(problem.shape)) * 16
        engine.set_initial(problem.initial.field0, None, z=problem.initial.z)
        engine.propagate_to(2 * STEP)
        field = engine.observe(field=True).require_field()
        assert np.isfinite(field).all() and np.any(field != 0.0)
        assert engine.memory_ledger()["fft_plan_device_growth"] < 1.5 * state
    finally:
        engine.close()


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_a_small_state_holds_no_pinned_ring(accelerator_backend):
    del accelerator_backend
    simulation = _simulation(32, 2, 1, None)
    problem = simulation.translate()
    engine = cast(
        "Any",
        select(
            "native",
            problem,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        ),
    )
    try:
        engine.set_initial(problem.initial.field0, None, z=problem.initial.z)
        engine.observe(field=True)
        assert engine.memory_ledger()["host_crossing_ring"] == 0
    finally:
        engine.close()
