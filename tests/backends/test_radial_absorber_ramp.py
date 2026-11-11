# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The radial collar's in-stage term on ``native``, a ramp read sample by sample (SEP
43).

``numba_cpu``'s ``test_the_numba_term_is_the_references_on_its_own_samples...`` held the
term against the reference's formula, on the fine samples the stage left. The surviving
Bessel-operator test takes a uniform temporal rate, which cannot see a rate read at the
wrong sample. Native keeps neither its term nor its fine samples apart, so the whole
first stage is compared with the reference's, with a ramp beside the composed matrix.
"""

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("solprop_native")


from solprop_native import NativeBackend  # pyright: ignore[reportMissingImports]

from backends.test_radial_absorber import LENGTH, absorber_case, problem_for
from solprop.backends.numpy_ref import NumpyRefBackend, passes

__all__ = ["absorber_case"]

EPS = np.finfo(float).eps


def _trial(backend, run_request, field, dz):
    """Load `field` into native's state and take one trial from z = 0 on its staged
    rows, the state's own (SEP 47 phase 5 retired the backend's driver calls)."""
    state = backend._native_state
    flat = np.ascontiguousarray(field, dtype=np.complex128).reshape(-1)
    state.set_initial(flat, np.empty(0, dtype=np.complex128))
    state.clear_fsal()
    backend._staged_rows()
    nodes = [float(node) for node in run_request.stepper.tableau.nodes]
    state.rk_trial(0.0, dz, nodes, prepare_only=False)


def test_a_ramped_rate_is_read_sample_by_sample(absorber_case):
    """Each bin within the retired bound of the two summation orders.

    Each time sample is an ``n_r``-term dot product plus the rate's term, so it differs
    by ``gamma(n_r + 1)`` of its absolute summands; a bin sums ``n_to`` of those, and an
    FFT adds about ``log2 n_to`` roundings of the same absolute scale. Twice that, for
    the two transforms. Weaker in that the rest of the stage is in the comparison; the
    ramp has to move the answer by more than the bound, so its term is exercised.
    """
    grid = absorber_case[0]
    base, run_request = problem_for(absorber_case, 8)
    # A ramp rather than a constant, so that the rate is read sample by sample.
    rate = np.linspace(0.0, 2000.0, grid.n_to)
    rate.setflags(write=False)
    # Beside zeros on the coarse axis: the case's zero-rate collar translates to no
    # temporal absorber at all.
    coarse = np.zeros(grid.n_t)
    coarse.setflags(write=False)
    problem = dataclasses.replace(
        base,
        responses=dataclasses.replace(
            base.responses, temporal_absorber=coarse, temporal_absorber_fine=rate
        ),
    )
    reference = NumpyRefBackend(problem, run_request, grid=grid)
    try:
        reference.set_initial(problem.initial.field0, None)
        reference._hoist(0.0)
        expected = reference._nonlinear(
            problem.initial.field0[None],
            reference._stage_scalars(0.0, 0.0),
            np.ones((1, *problem.shape), complex),
        )[0]
        matrix = reference._spatial_absorber.copy()
    finally:
        reference.close()
    backend = NativeBackend(problem, run_request, grid=grid)
    try:
        _trial(backend, run_request, problem.initial.field0, LENGTH / 8)
        measured = backend._native_state.stage_to_host(0).reshape(expected.shape)
    finally:
        backend.close()
    samples = passes.to_time(
        problem.initial.field0,
        problem.grids.crop_index,
        grid.n_to,
        kind=problem.grid_kind,
    )
    n_r = problem.shape[0]
    rows = samples.reshape(n_r, -1)
    absolute = np.abs(matrix) @ np.abs(rows) + (0.5 * rate * np.abs(samples)).reshape(
        n_r, -1
    )
    n_to = grid.n_to
    operations = n_r + 1 + 4 * int(np.log2(n_to))
    bound = 2 * operations * EPS * n_to * float(np.max(absolute))
    assert bound > 0
    assert_allclose(measured, expected, rtol=0, atol=bound)
