# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Mixture propagation, channel statistics, repeatability and accepted-stage stashes."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from backends.forced_threads import assert_threaded
from backends.host_cases_mixture_threads import NAMES, PERIODIC
from backends.matched import native_thread_fields
from bounds import (
    AccuracyEnvelope,
    assert_within,
    assert_work_within,
    departure,
    device_backend,
)
from mixture_cases import mixture_case, mixture_translation
from sessions.runs import launch_statistics
from sessions.runs import run as session_run
from solprop.backends.events import ObservationKind
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.session import Start
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule

PROBE = Path(__file__).with_name("host_cases_mixture_threads.py")


def run(backend_type, problem, run_request, grid, *, steps=None):
    """Run the session to the end, or to its first `steps` accepted steps.

    Return the field there, the summary since the launch, and every statistic of
    :data:`NAMES` at the launch and there. A short run observes the end of every
    `steps`-th accepted step (``step_planes``), which changes no step, and closes the
    session at the first.
    """
    stepper = run_request.stepper
    if steps is not None:
        stepper = dataclasses.replace(stepper, step_planes=steps)
    backend = backend_type(problem, run_request.with_stepper(stepper), grid=grid)
    try:
        result = session_run(
            backend,
            problem,
            backend.request.for_run(stats=StatsSpec(names=NAMES)),
            observations=None if steps is None else 2,
        )
        launch, there = result.observations[0], result.observations[1]
        if steps is not None:
            assert there.kind is ObservationKind.STEP
        else:
            assert len(result.observations) == 2
        return result.fields[1], there.summary, launch.stats, there.stats
    finally:
        backend.close()


@pytest.mark.parametrize("steps", [1, 8, pytest.param(None, marks=pytest.mark.medium)])
def test_mixture_propagation_ladder_and_every_species_statistic(
    thread_budget_lifted, steps, equivalence_backend_class
):
    _, grid, _, _, (problem, run_request) = mixture_translation()
    if steps is not None:
        # A one/eight-step ladder uses the declared initial step. Forcing the
        # entire 1 mm into one step amplifies FFT plan roundoff beyond the fixed
        # bound; the adaptive full run below still reaches the fibre end with the
        # unchanged controller.
        run_request = run_request.with_stepper(
            dataclasses.replace(
                run_request.stepper,
                schedule=FixedSchedule.uniform(
                    0.0, grid.length, run_request.stepper.dz_init
                ),
            )
        )
    else:
        # Start this adaptive qualification at its declared maximum step. A 1e-5 m
        # start seeds the controller with an embedded norm of 7.45e-7; libm rounding
        # then selects 59 versus 60 steps and the final fields differ by 1.98e-6.
        # The maximum-step start resolves the first error estimate; neither the
        # controller, its tolerances, nor the bounds below are changed.
        run_request = run_request.with_stepper(
            dataclasses.replace(run_request.stepper, dz_init=run_request.stepper.dz_max)
        )
    # Seven stages per step, each with two FFTs, a cubic evaluation and two
    # accumulated current integrals per channel. Budget 64 eps per fine sample,
    # per channel, per stage.
    if steps is None:
        assert_adaptive_run_is_accurate(
            equivalence_backend_class, problem, run_request, grid
        )
        return
    bound = 64 * grid.n_to * 2 * 7 * steps * np.finfo(float).eps
    assert bound > 0
    ours = run(equivalence_backend_class, problem, run_request, grid, steps=steps)
    oracle = run(NumpyRefBackend, problem, run_request, grid, steps=steps)
    assert ours[1].n_steps == oracle[1].n_steps == steps
    difference = float(np.linalg.norm(ours[0] - oracle[0]) / np.linalg.norm(oracle[0]))
    assert difference < bound
    for measured, expected in zip(ours[2:], oracle[2:], strict=True):
        for name in NAMES:
            assert_allclose(
                measured[name],
                expected[name],
                rtol=bound,
                atol=bound if name == "species_ionisation" else 0.0,
            )


def assert_adaptive_run_is_accurate(backend_class, problem, run_request, grid):
    """Hold an adaptive run to the oracle's accuracy, not to its trajectory.

    Two adaptive integrations may take different steps once rounding tips one
    decision (SEP 25 section 4.4), so the compiled run is compared with a run at
    ``rtol = 1e-10`` and may depart from it by at most four times the oracle's own
    departure at the same settings (:class:`bounds.AccuracyEnvelope`), the field and
    each statistic alike, with an absolute floor of the 100-step rounding budget for
    a statistic the oracle happens to meet exactly; its step count is a work band of
    a factor 1.25 about the oracle's.
    """
    tight = run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper,
            controller=dataclasses.replace(run_request.stepper.controller, rtol=1e-10),
        )
    )
    converged = run(NumpyRefBackend, problem, tight, grid)
    oracle = run(NumpyRefBackend, problem, run_request, grid)
    ours = run(backend_class, problem, run_request, grid)
    assert oracle[1].n_steps <= 100
    assert_work_within(ours[1].n_steps, oracle[1].n_steps, factor=1.25, what="steps")
    floor = 64 * grid.n_to * 2 * 7 * 100 * np.finfo(float).eps
    field = AccuracyEnvelope(
        "numpy_ref at rtol 1e-10, mixture case, 1 mm, relative L2",
        max(departure(oracle[0], converged[0]), floor),
    )
    assert departure(ours[0], converged[0]) <= field.bound
    for name in NAMES:
        best = converged[3][name]
        scale = float(np.max(np.abs(best))) or 1.0
        statistic = AccuracyEnvelope(
            f"numpy_ref at rtol 1e-10, {name}, largest-value relative",
            max(float(np.max(np.abs(oracle[3][name] - best))) / scale, floor),
        )
        error = float(np.max(np.abs(ours[3][name] - best))) / scale
        assert error <= statistic.bound, name


def ladder_bound(grid, steps):
    """The ladder's budget above: 64 eps a fine sample, channel and stage, 7 a step."""
    return 64 * grid.n_to * 2 * 7 * steps * np.finfo(float).eps


def test_mixture_run_is_repeatable_and_thread_invariant(tmp_path):
    """A repeat and a four-thread run of eight fixed steps, within the ladder's budget.

    On one schedule the three runs are one computation, its sums ordered by the
    threads, so the field and every statistic agree to the rounding the ladder
    allows a whole backend (the ionised fractions absolutely as well). Each run is its
    own process, whose Kokkos pool has that thread count.
    """
    pytest.importorskip("solprop_native")
    if device_backend("native") != "none":
        pytest.skip("a device build's host space is serial")
    _, grid, _, _, _ = mixture_case()
    bound = ladder_bound(grid, 8)
    names = ["steps", "field", *(f"stats:{name}" for name in NAMES)]
    first, repeat, parallel = (
        native_thread_fields(PROBE, threads, names, tmp_path) for threads in (1, 1, 4)
    )
    assert_threaded(
        parallel["sizing"], 4, "kernels.stage", "kernels.response", "fft.fft"
    )
    for actual in (repeat, parallel):
        assert int(actual["steps"]) == int(first["steps"]) == 8
        assert_within(actual["field"], first["field"], bound, what="field")
        for name in NAMES:
            absolute = bound if name == "species_ionisation" else 0.0
            assert_allclose(
                actual[f"stats:{name}"],
                first[f"stats:{name}"],
                rtol=bound,
                atol=absolute,
                err_msg=name,
            )


def test_periodic_species_rows_keep_their_positions_across_threads(tmp_path):
    """One thread against four: the rows' positions exact, their values in budget.

    The two runs replay eight fixed steps, so their rows stand at the same positions,
    exactly, with values within the ladder's budget.
    """
    pytest.importorskip("solprop_native")
    if device_backend("native") != "none":
        pytest.skip("a device build's host space is serial")
    _, grid, _, _, _ = mixture_case()
    bound = ladder_bound(grid, 8)
    names = ["z", *(f"rows:{name}" for name in PERIODIC)]
    outputs = [
        native_thread_fields(PROBE, threads, names, tmp_path) for threads in (1, 4)
    ]
    assert_threaded(
        outputs[1]["sizing"], 4, "kernels.stage", "kernels.response", "fft.fft"
    )
    assert outputs[0]["z"].size > 0
    assert_array_equal(outputs[0]["z"], outputs[1]["z"])
    for name in PERIODIC:
        absolute = bound if name == "species_ionisation" else 0.0
        assert_allclose(
            outputs[1][f"rows:{name}"],
            outputs[0][f"rows:{name}"],
            rtol=bound,
            atol=absolute,
            err_msg=name,
        )


def test_periodic_channel_statistics_match_a_fresh_backend(backend_class):
    _, grid, _, _, (problem, original) = mixture_translation()
    periodic = ("species_ionisation", "species_electron_density")
    run_request = original.with_stepper(
        dataclasses.replace(
            original.stepper,
            schedule=FixedSchedule.uniform(0, grid.length, grid.length / 8),
        )
    ).for_run(stats=StatsSpec(names=NAMES, periodic=periodic, period=1))
    bound = 64 * grid.n_to * np.finfo(float).eps
    assert bound > 0.0
    backend = backend_class(problem, run_request, grid=grid)
    try:
        result = session_run(backend, problem)
        taken = result.joined[1]
        assert taken.n_rows == 8
        assert result.observations[-1].z == grid.length
        cached = result.observations[-1].stats
        if backend_class.name == "native":
            assert backend.memory_ledger()["plasma_stash"] == (
                2 * problem.transform.n_nodes * (1 + 2 * problem.flags.n_channels) * 8
            )
        fresh = backend_class(problem, run_request, grid=grid)
        try:
            # This fixture has constant densities and geometry. Starting from the
            # final field forces a fresh exposure scan with identical material data.
            index = len(run_request.stepper.save_planes)
            values = launch_statistics(
                fresh,
                problem,
                fresh.request.for_run(stats=StatsSpec(names=NAMES), start_plane=index),
                Start(field=result.fields[-1], z=grid.length, plane_index=index),
            )
        finally:
            fresh.close()
        for name in NAMES:
            assert_allclose(values[name], cached[name], rtol=bound, atol=0.0)
        for name in periodic:
            assert_allclose(
                taken.values[name][-1],
                cached[name],
                rtol=bound,
                atol=0.0,
            )
    finally:
        backend.close()
