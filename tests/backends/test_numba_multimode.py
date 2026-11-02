# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""``numba_cpu``'s multimode execution: its node regimes and stacked projection."""

import math
from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest

from backends import test_numpy_ref_multimode as multimode
from backends.test_numpy_ref_multimode import build, modes_of, peak_gap, pulse
from bounds import EPS, assert_within, higham_gamma
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numba_cpu import backend as numba_backend
from solprop.backends.numba_cpu import fft as numba_fft
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.modes import ModeCollection, transverse_quadrature

grid = multimode.grid
capillary = multimode.capillary


@pytest.mark.parametrize("forced", [None, "small", "large"])
@pytest.mark.parametrize("modal", [False, True])
def test_modal_kernel_regime_keeps_overrides_and_the_trial(
    grid, capillary, monkeypatch, modal, forced, record_property
):
    """Changing node dispatch preserves the trial and allocates no extra buffers.

    The two regimes are one trial step by two dispatches, which may order the
    reductions differently: seven stages, each a transform's 32 eps a level and the
    projection's and synthesis's ``gamma(2 n_nodes + 2)`` (the vector-route bound
    below).
    """
    monkeypatch.setattr(numba_fft, "MODAL_REGIME_ELEMENTS", 1)
    mode_input = [capillary] if modal else capillary
    options = (
        {
            "quadrature": transverse_quadrature(
                [capillary], n_r=8, radial_rule="gauss_kronrod"
            )
        }
        if modal
        else {}
    )
    problem = build(grid, mode_input, pulse(grid), **options)
    fields = []
    ledgers = []
    for regime in ("small", forced):
        backend = NumbaCpuBackend(
            problem,
            grid=grid,
            threads=1 if regime == "small" else 2,
            fftw_threads=1,
            blas_threads=1,
            plan_flag="estimate",
            regime=regime,
        )
        try:
            backend.set_initial(problem.initial.field0, None)
            backend._trial_step(0.0, problem.stepper.dz_init)
            fields.append(backend._solution.copy())
            ledgers.append(backend.memory_ledger())
            metadata = backend.metadata()
            assert metadata["kernel_regime_elements"] == (
                1 if modal else numba_fft.REGIME_ELEMENTS
            )
            expected = regime or ("large" if modal else "small")
            assert metadata["regime"] == expected
            assert metadata["parallel_nodes"] == (expected == "large")
        finally:
            backend.close()
    n_nodes = problem.transform.n_nodes
    bound = 7 * (
        32 * EPS * math.ceil(math.log2(grid.n_to)) + higham_gamma(2 * n_nodes + 2)
    )
    assert_within(fields[1], fields[0], bound, what="trial step")
    # Allocator growth during planning is a process measurement, not an owned buffer.
    # Retain it separately: even two serial constructions need not grow the same heap.
    for index, ledger in enumerate(ledgers):
        growth = ledger.pop("fft_plan_heap_growth")
        record_property(f"plan_heap_growth_{index}", growth)
        assert isinstance(growth, int) and growth >= 0
    assert ledgers[0] == ledgers[1]


def projection_comparison(output, expected, bound):
    """Assert the entrywise envelope and retain the most strained comparison."""
    gap = np.abs(output - expected)
    assert np.all(gap <= bound)
    assert np.all(gap[bound == 0] == 0)
    ratios = np.divide(gap, bound, out=np.zeros_like(gap), where=bound > 0)
    worst = np.unravel_index(int(np.argmax(ratios)), ratios.shape)
    return {"error": float(gap[worst]), "bound": float(bound[worst])}


@pytest.mark.parametrize("layout", ["scalar", "state", "rows"])
@pytest.mark.parametrize("zero", [False, True])
def test_stacked_projection_preserves_rules_chunks_and_owned_storage(
    grid, capillary, layout, zero, record_property, monkeypatch
):
    """Compare both rules against separate products with a prior error envelope."""
    monkeypatch.setattr(numba_backend, "_STACKED_PROJECTION_MIN_NODES", 1)
    modes = modes_of(capillary, 2)
    if layout == "rows":
        modes = ModeCollection.with_polarisations(modes)
    start = pulse(grid, len(modes))
    if layout == "state":
        start = np.repeat(start, 2, axis=1) / math.sqrt(2)
    rule = transverse_quadrature(modes, n_r=4, radial_rule="gauss_kronrod")
    problem = build(grid, modes, start, quadrature=rule)
    chunk_nodes = 4 if rule.n_nodes % 4 else 5
    problem = replace(
        problem, transform=replace(problem.transform, node_chunk=chunk_nodes)
    )
    rng = np.random.default_rng(33)
    comparisons = []
    with closing(
        NumbaCpuBackend(problem, grid=grid, threads=1, plan_flag="estimate")
    ) as backend:
        assert backend.metadata()["modal_projection"] == "stacked_packed"
        m, r = problem.transform.projection.shape
        p = problem.shape[2]
        ledger = backend.memory_ledger()
        assert ledger["packed_projection"] == 4 * m * r * 8
        assert ledger["stacked_projection"] == 2 * m * p * grid.n_to * 8
        assert ledger["embedded_projection"] == 2 * m * r * 8
        assert backend._plans.coarse_projected is not None
        # A non-unit taper tests the stage's matrix scaling as well as packing.
        scale = 1.25
        for packed, scaled in zip(
            backend._projection_chunks, backend._scaled_projection_chunks, strict=True
        ):
            np.multiply(packed, scale, out=scaled)
        backend._plans.projected.fill(np.nan)
        backend._plans.coarse_projected.fill(np.nan)
        transform = problem.transform
        for first in range(0, transform.n_nodes, chunk_nodes):
            last = min(first + chunk_nodes, transform.n_nodes)
            rows = slice(first * transform.n_pol, last * transform.n_pol)
            count = (last - first) * backend._n_pol_node
            source = rng.normal(size=(rows.stop - rows.start, p * grid.n_to))
            if zero:
                source.fill(0)
            backend._polarisation[:count] = source.reshape(count, grid.n_to)
            fine = scale * transform.projection[:, rows]
            coarse = scale * transform.projection_coarse[:, rows]
            previous = [
                np.zeros_like(source[:m])
                if first == 0
                else output.reshape(m, -1).copy()
                for output in (
                    backend._plans.projected,
                    backend._plans.coarse_projected,
                )
            ]
            # Two k-term real products and their final accumulation.
            # Derive the entrywise envelope before either output is compared.
            q = 2 * source.shape[0] + 8
            gamma = q * np.finfo(float).eps / (1 - q * np.finfo(float).eps)
            assert gamma > 0
            bounds = [
                2 * gamma * (np.abs(matrix) @ np.abs(source) + np.abs(prior))
                for matrix, prior in zip((fine, coarse), previous, strict=True)
            ]
            expected = [
                matrix @ source + prior
                for matrix, prior in zip((fine, coarse), previous, strict=True)
            ]
            backend._project(rows, count, first == 0)
            for output, want, bound in zip(
                (backend._plans.projected, backend._plans.coarse_projected),
                expected,
                bounds,
                strict=True,
            ):
                comparisons.append(
                    projection_comparison(output.reshape(m, -1), want, bound)
                )
        last_nodes = transform.n_nodes % chunk_nodes
        assert 0 < last_nodes < chunk_nodes
        assert backend._projection_chunks[-1].shape[1] == last_nodes * transform.n_pol
    record_property("projection_comparisons", comparisons)


def test_stacked_projection_is_absent_without_an_embedded_rule(grid, capillary):
    """A mode-averaged problem retains the separate path and adds no storage."""
    problem = build(grid, capillary, pulse(grid))
    with closing(
        NumbaCpuBackend(problem, grid=grid, threads=1, plan_flag="estimate")
    ) as backend:
        assert backend.metadata()["modal_projection"] == "separate"
        ledger = backend.memory_ledger()
        assert ledger["packed_projection"] == ledger["stacked_projection"] == 0
        assert ledger["embedded_projection"] == 0


@pytest.mark.parametrize(
    ("n_modes", "n_nodes", "chunk", "expected"),
    [
        (1, 384, 85, False),
        (2, 384, 170, True),
        (3, 1024, 42, True),
        (4, 384, 85, True),
        (5, 384, 85, False),
        (6, 1024, 42, False),
        (2, 255, 85, False),
        (2, 256, 85, True),
        (2, 384, 384, False),
        (4, 33, 33, False),
    ],
)
def test_stacked_projection_dispatch_keeps_unqualified_shapes_separate(
    n_modes, n_nodes, chunk, expected
):
    """Exercise measured shapes and both sides of the predeclared brackets."""
    assert numba_backend._stacked_projection_shape(n_modes, n_nodes, chunk) is expected


def test_stacked_projection_trial_keeps_the_existing_bound(
    grid, capillary, monkeypatch, record_property
):
    """Exercise packed stage scaling and the full initial trial at a tiny shape."""
    monkeypatch.setattr(numba_backend, "_STACKED_PROJECTION_MIN_NODES", 1)
    modes = modes_of(capillary, 2)
    rule = transverse_quadrature(modes, n_r=4, radial_rule="gauss_kronrod")
    problem = build(grid, modes, pulse(grid, 2), quadrature=rule)
    problem = replace(problem, transform=replace(problem.transform, node_chunk=4))
    eps = np.finfo(float).eps
    reduction = rule.n_nodes * problem.transform.n_pol * problem.shape[2]
    # A seven-stage initial-trial budget.
    gamma = reduction * eps / (1 - reduction * eps)
    bound = 7 * (
        16 * eps * math.ceil(math.log2(grid.n_to))
        + 2 * gamma
        + 64 * eps / (1 - 64 * eps)
    )
    assert bound > 0
    fields = []
    for backend_class in (NumpyRefBackend, NumbaCpuBackend):
        with closing(backend_class(problem, grid=grid)) as backend:
            backend.set_initial(problem.initial.field0, None)
            fields.append(backend._trial_step(0.0, problem.stepper.dz_init)[1].copy())
    deviation = peak_gap(fields[1], fields[0])
    assert deviation <= bound
    record_property(
        "initial_trial_comparison", {"deviation": deviation, "bound": bound}
    )
