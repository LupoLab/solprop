# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""``numba_cpu``'s Raman transforms: their FFTW pools and the tail chunk's plans."""

import numpy as np
import pytest

from backends import test_numpy_ref_raman as raman
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numba_cpu import fft as numba_fft
from solprop.backends.numpy_ref import NumpyRefBackend

raman_field_grid = raman.raman_field_grid
raman_capillary = raman.raman_capillary


@pytest.mark.parametrize("override", [None, 1, 2])
def test_raman_pool_uses_its_doubled_operand_and_shared_override(
    thread_budget_lifted, raman_capillary, override, monkeypatch, record_property
):
    """Raman owns its size choice while explicit FFTW requests still govern both."""
    problem, _, _ = raman_capillary()
    monkeypatch.setattr(
        numba_fft, "_thread_budget", lambda: numba_fft.ThreadBudget(4, "affinity")
    )
    monkeypatch.setattr(numba_fft, "FFTW_THREAD_ELEMENTS", 2 * problem.grids.n_to)
    monkeypatch.delenv("SOLPROP_FFTW_THREADS", raising=False)
    if override == 1:
        monkeypatch.setenv("SOLPROP_FFTW_THREADS", "1")
    eps = np.finfo(float).eps
    reduction = problem.transform.n_nodes * problem.transform.n_pol * problem.shape[2]
    gamma = reduction * eps / (1 - reduction * eps)
    # The seven-stage initial-trial envelope, derived before either run.
    bound = 7 * (
        16 * eps * np.ceil(np.log2(problem.grids.n_to))
        + 2 * gamma
        + 64 * eps / (1 - 64 * eps)
    )
    reference = NumpyRefBackend(problem)
    try:
        reference.set_initial(problem.initial.field0, None)
        expected = reference._trial_step(0.0, problem.stepper.dz_init)[1].copy()
    finally:
        reference.close()
    backend = NumbaCpuBackend(
        problem,
        threads=1,
        fftw_threads=2 if override == 2 else None,
        plan_flag="estimate",
    )
    try:
        metadata = backend.metadata()
        assert metadata["fftw_threads"] == (override or 1)
        assert metadata["raman_fftw_threads"] == (override or 4)
        assert backend._plans.drive.shape == (1, 2 * problem.grids.n_to)
        backend.set_initial(problem.initial.field0, None)
        actual = backend._trial_step(0.0, problem.stepper.dz_init)[1].copy()
    finally:
        backend.close()
    deviation = float(np.max(np.abs(actual - expected)) / np.max(np.abs(expected)))
    assert deviation <= bound
    record_property("raman_initial_trial", {"deviation": deviation, "bound": bound})


@pytest.mark.parametrize("chunk", [3, 4, 5])
@pytest.mark.parametrize("override", [None, 1, 2])
def test_raman_tail_trial_matches_reference_without_replanning(
    thread_budget_lifted, raman_capillary, chunk, override, monkeypatch, record_property
):
    """Actual remainder plans preserve the trial, padding and explicit pool requests."""
    problem, _, _ = raman_capillary(modal=True, chunk=chunk)
    assert problem.transform.n_nodes == 5
    tail = problem.transform.n_nodes % chunk
    monkeypatch.setattr(
        numba_fft, "_thread_budget", lambda: numba_fft.ThreadBudget(4, "affinity")
    )
    monkeypatch.setattr(numba_fft, "FFTW_THREAD_ELEMENTS", 4 * problem.grids.n_to)
    monkeypatch.delenv("SOLPROP_FFTW_THREADS", raising=False)
    if override == 1:
        monkeypatch.setenv("SOLPROP_FFTW_THREADS", "1")
    eps = np.finfo(float).eps
    reduction = problem.transform.n_nodes * problem.transform.n_pol * problem.shape[2]
    gamma = reduction * eps / (1 - reduction * eps)
    # The same seven-stage envelope at this test's dimensions.
    bound = 7 * (
        16 * eps * np.ceil(np.log2(problem.grids.n_to))
        + 2 * gamma
        + 64 * eps / (1 - 64 * eps)
    )
    reference = NumpyRefBackend(problem)
    try:
        reference.set_initial(problem.initial.field0, None)
        expected = reference._trial_step(0.0, problem.stepper.dz_init)[1].copy()
    finally:
        reference.close()
    backend = NumbaCpuBackend(
        problem,
        threads=1,
        fftw_threads=2 if override == 2 else None,
        plan_flag="estimate",
    )
    try:
        metadata = backend.metadata()
        assert metadata["raman_tail_nodes"] == tail
        assert metadata["raman_fftw_threads"] == (override or 4)
        expected_tail_threads = (override or (4 if tail >= 2 else 1)) if tail else None
        assert metadata["raman_tail_fftw_threads"] == expected_tail_threads
        assert (backend._plans.raman_tail_to_time is not None) == bool(tail)

        def refuse_replanning(*args, **kwargs):
            pytest.fail("a live trial attempted to construct an FFTW plan")

        monkeypatch.setattr(numba_fft, "_plan", refuse_replanning)
        fields = []
        for _ in range(2):
            backend.set_initial(problem.initial.field0, None)
            fields.append(backend._trial_step(0.0, problem.stepper.dz_init)[1].copy())
            assert np.array_equal(
                backend._plans.drive[:, problem.grids.n_to :],
                np.zeros((chunk, problem.grids.n_to)),
            )
        # The same trial twice: within the trial's own bound against the reference.
        repeat = float(
            np.max(np.abs(fields[1] - fields[0])) / np.max(np.abs(fields[0]))
        )
        assert repeat <= bound
        deviation = float(
            np.max(np.abs(fields[0] - expected)) / np.max(np.abs(expected))
        )
        assert deviation <= bound
        ledger = backend.memory_ledger()
        assert ledger["node_raman_drive"] == chunk * 2 * problem.grids.n_to * 8
        assert ledger["node_raman_convolved"] == chunk * 2 * problem.grids.n_to * 8
    finally:
        backend.close()
    record_property(
        "raman_tail_initial_trial", {"deviation": deviation, "bound": bound}
    )
