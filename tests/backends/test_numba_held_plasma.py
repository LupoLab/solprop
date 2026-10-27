# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify the compiled current streams, complete rows and memory admission."""

import dataclasses

import numpy as np
import pytest

from backends.test_band_plasma_parity import CASES, bound, case
from backends.test_radial_nonlinear import QuadraticRate, derivative
from bounds import assert_within, gamma
from solprop.backends.memory import _native_held_current_memory, held_current_memory
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.exceptions import TranslationError
from solprop.translate.ionisation import build_rate_table
from solprop.translate.plasma_integration import design_running_integral
from solprop.translate.problem import _balanced_chunk, _held_current_chunk

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
]


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("rule", ["band", "trapezoid"])
def test_source_matches_numpy_with_complete_masked_rows(name, rule):
    grid, problem = case(name)
    problem = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses, plasma_integral=design_running_integral(grid, rule)
        ),
    )
    relative = bound(problem, grid, "numba_cpu", 0)
    assert relative > 0
    expected = derivative(NumpyRefBackend, problem, grid)
    actual = derivative(NumbaCpuBackend, problem, grid)
    scale = np.max(abs(expected))
    assert scale > 0
    np.testing.assert_allclose(actual, expected, rtol=0, atol=relative * scale)
    assert np.count_nonzero(actual[..., ~grid.active_mask]) == 0
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        ledger = backend.memory_ledger()
        for key, value in held_current_memory(problem).items():
            assert ledger[key] == value, key
        backend.set_initial(problem.initial.field0, None)
        backend._trial_step(0.0, 0.0)
        backend.set_initial(np.zeros_like(problem.initial.field0), None)
        backend._trial_step(0.0, 0.0)
        assert np.count_nonzero(backend._stages) == 0
    finally:
        backend.close()


def test_node_admission_counts_all_held_histories_together():
    grid, problem = case("mode_averaged")
    transform, responses = problem.transform, problem.responses
    with pytest.raises(TranslationError, match="held plasma current history"):
        _held_current_chunk(transform, responses, 1, grid.n_to, 1)
    absent = dataclasses.replace(
        responses,
        plasma_channels=(),
        plasma_integral=design_running_integral(grid, "trapezoid"),
    )
    assert _held_current_chunk(transform, absent, 1, grid.n_to, 1) is transform
    no_plasma = dataclasses.replace(problem, responses=absent)
    assert sum(held_current_memory(no_plasma).values()) == 0
    assert sum(_native_held_current_memory(no_plasma).values()) == 0


def test_planar_admission_reserves_a_whole_transverse_row():
    grid, problem = case("planar")
    transform = problem.transform
    assert transform.geometry is not None
    n_x = transform.geometry.shape[0]
    # This cannot even hold the row's twenty baseline complex buffers, before
    # adding the held-current streams and integration histories.
    budget = 20 * 16 * grid.n_to * n_x - 1
    with pytest.raises(TranslationError, match="full transverse row"):
        _held_current_chunk(transform, problem.responses, 1, grid.n_to, budget)
    transform = dataclasses.replace(transform, node_chunk=n_x + 1)
    admitted = _held_current_chunk(transform, problem.responses, 1, grid.n_to, 1 << 30)
    assert admitted.node_chunk == n_x


def test_budget_chunks_spread_the_nodes_evenly_over_their_passes():
    # A budget that admits all but one node must not leave a one-node tail pass:
    # nine nodes under a limit of eight take two passes of five and four.
    _, problem = case("multimode")
    transform = problem.transform
    assert transform.geometry is None
    n_nodes = transform.n_nodes
    assert _balanced_chunk(transform) is transform
    tight = dataclasses.replace(transform, node_chunk=n_nodes - 1)
    assert _balanced_chunk(tight).node_chunk == -(-n_nodes // 2)
    single = dataclasses.replace(transform, node_chunk=1)
    assert _balanced_chunk(single) is single


def test_planar_chunks_stay_whole_rows_when_balanced():
    _, problem = case("planar")
    transform = problem.transform
    assert transform.geometry is not None
    n_x = transform.geometry.shape[0]
    rows = transform.n_nodes // n_x
    assert rows == 4
    # Three of four rows a pass would leave a one-row tail: two rows a pass.
    three = dataclasses.replace(transform, node_chunk=3 * n_x)
    assert _balanced_chunk(three).node_chunk == 2 * n_x
    # A chunk shorter than a row is not a whole number of rows and is kept.
    partial = dataclasses.replace(transform, node_chunk=5)
    assert _balanced_chunk(partial) is partial


@pytest.mark.parametrize("rule", ["band", "trapezoid"])
def test_embedded_current_projection_accumulates_partial_chunks(rule, monkeypatch):
    grid, problem = case("multimode")
    assert problem.transform.projection_coarse is not None
    # The small He fixture has exactly zero tabulated ionisation. Use the
    # existing analytic rate fixture so both projected current streams are
    # populated and accumulation cannot pass by comparing two empty arrays.
    rate = build_rate_table(QuadraticRate(), (1e3, 1e10))
    rate.log_rate.setflags(write=False)
    rate.coefficients.setflags(write=False)
    problem = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses,
            plasma_integral=design_running_integral(grid, rule),
            ionisation_table=(rate,) * len(problem.responses.plasma_channels),
            plasma_channels=tuple(
                dataclasses.replace(channel, rate=rate)
                for channel in problem.responses.plasma_channels
            ),
        ),
        transform=dataclasses.replace(problem.transform, node_chunk=3),
    )
    relative = bound(problem, grid, "numba_cpu", 0)
    assert relative > 0
    reference = NumpyRefBackend(problem, grid=grid)
    compiled = NumbaCpuBackend(problem, grid=grid)
    try:
        reference._hoist(0.0)
        state = problem.initial.field0
        bundle = reference._stage_scalars(0.0, 0.0)
        transform, grids = problem.transform, problem.grids
        nodes = passes.synthesise(
            passes.to_time(state[0], grids.crop_index, grids.n_to, kind="field"),
            transform.synthesis,
            bundle.taper_scale,
            transform.n_pol,
        )
        current = reference._column_sources(nodes, bundle, None).current
        assert current is not None
        # SEP 27 section 4.6: apply the existing column allowance to J before
        # multiplying by either collar, then project with absolute weights.
        # The tiny residual on the trailing edge cannot scale the error in
        # the full oscillating current whose cancellation produced that tail.
        current_limit = relative * np.max(abs(current), axis=-1, keepdims=True)
        rounding = gamma(2 * transform.n_nodes * transform.n_pol + 4)
        collars = (grids.t_window_fine, grids.t_window_fine_derivative)
        limits = [
            passes.project(
                abs(collar) * current_limit + rounding * abs(collar * current),
                abs(transform.projection_coarse),
                bundle.taper_scale,
                transform.n_pol,
            )
            for collar in collars
        ]
        assert all(np.max(limit) > 0 for limit in limits)
        reference._nonlinear(state, bundle, np.ones_like(state))
        assert reference._last_plasma is not None
        _, _, coarse_gate, coarse_edge = reference._last_plasma
        assert coarse_gate is not None and coarse_edge is not None
        expected = np.stack((coarse_gate, coarse_edge))
        # Observe the column before its products are projected. This separately
        # checks the current allowance rather than only its propagated bound.
        compiled_current = np.zeros_like(current)
        respond = compiled._respond_chunk

        def collect(bundle, first, last, raman):
            respond(bundle, first, last, raman)
            workspace = compiled._field_plasma
            assert workspace is not None
            count = (last - first) * compiled._n_pol_node
            compiled_current[first:last] = workspace.current[:count].reshape(
                compiled_current[first:last].shape
            )

        monkeypatch.setattr(compiled, "_respond_chunk", collect)
        compiled.set_initial(state, None)
        compiled._trial_step(0.0, 0.0)
        workspace = compiled._field_plasma
        assert workspace is not None and workspace.coarse_projected is not None
        actual = workspace.coarse_projected.reshape(expected.shape)
        assert_within(
            compiled_current, current, current_limit, elementwise=True, what="current"
        )
        assert_within(
            actual, expected, np.stack(limits), elementwise=True, what="projection"
        )
    finally:
        reference.close()
        compiled.close()
