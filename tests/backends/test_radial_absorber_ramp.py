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


from solprop_native import NativeBackend

from backends.test_radial_absorber import LENGTH, absorber_case, problem_for
from solprop.backends.numpy_ref import NumpyRefBackend, passes

__all__ = ["absorber_case"]

EPS = np.finfo(float).eps


def test_a_ramped_rate_is_read_sample_by_sample(absorber_case):
    """Each bin within the retired bound of the two summation orders.

    Each time sample is an ``n_r``-term dot product plus the rate's term, so it differs
    by ``gamma(n_r + 1)`` of its absolute summands; a bin sums ``n_to`` of those, and an
    FFT adds about ``log2 n_to`` roundings of the same absolute scale. Twice that, for
    the two transforms. Weaker in that the rest of the stage is in the comparison; the
    ramp has to move the answer by more than the bound, so its term is exercised.
    """
    grid = absorber_case[0]
    base = problem_for(absorber_case, 8)
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
    reference = NumpyRefBackend(problem, grid=grid)
    try:
        reference.set_initial(problem.initial.field0, None)
        reference._hoist(0.0)
        expected = reference._nonlinear(
            problem.initial.field0,
            reference._stage_scalars(0.0, 0.0),
            np.ones(problem.shape, complex),
        )[0]
        matrix = reference._spatial_absorber.copy()
    finally:
        reference.close()
    backend = NativeBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend._trial_step(0.0, LENGTH / 8)
        measured = backend._native_state.stage_to_host(0).reshape(expected.shape)
    finally:
        backend.close()
    samples = passes.to_time(
        problem.initial.field0[0],
        problem.grids.crop_index,
        grid.n_to,
        kind=problem.grid_kind,
    )
    n_r = problem.shape[1]
    rows = samples.reshape(n_r, -1)
    absolute = np.abs(matrix) @ np.abs(rows) + (0.5 * rate * np.abs(samples)).reshape(
        n_r, -1
    )
    n_to = grid.n_to
    operations = n_r + 1 + 4 * int(np.log2(n_to))
    bound = 2 * operations * EPS * n_to * float(np.max(absolute))
    assert bound > 0
    assert_allclose(measured, expected, rtol=0, atol=bound)
