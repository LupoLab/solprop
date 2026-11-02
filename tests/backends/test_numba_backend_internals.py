# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Checks of ``numba_cpu``'s own machinery, moved out of the shared modules.

Each test here is about Numba's stage kernels, its host adapters or what its constructor
holds, and has a counterpart on the backends that stay in the module it came from where
the behaviour is shared. They go with ``numba_cpu`` (SEP 43 section 6.4).
"""

import copy
import dataclasses
import tracemalloc

import numpy as np
import pytest
from numpy.testing import assert_allclose

from backends.test_batch_validation import assert_family_refused
from backends.test_compiled_multistep import (
    ROUTES,
    assert_marches_without_responses,
    assert_sources_the_reference,
)
from backends.test_dispersive_kerr import capillary
from backends.test_free_boundaries import free_backend_problem  # noqa: F401 - a fixture
from backends.test_low_storage import MARCH
from backends.test_numpy_ref_backend import z_table_departure_and_bound
from backends.test_radial_absorber import (
    EPS,
    LENGTH,
    absorber_case,  # noqa: F401 - a fixture
    problem_for,
)
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numba_cpu.batch import NumbaCpuBatchBackend
from solprop.backends.numba_cpu.low_storage import NumbaLowStorage
from solprop.backends.numpy_ref import passes
from solprop.backends.numpy_ref.batch import NumpyBatchBackend
from solprop.exceptions import BackendError
from solprop.stepper import FixedSchedule

# As the low-storage module's: its fixtures launch into windows kept tight for speed.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

# The Runge-Kutta march's own field-sized rows, empty under the low-storage march.
RUNGE_KUTTA_ROWS = ("stages", "accumulator_solution", "accumulator_error")


@pytest.mark.parametrize(
    ("factory", "wrong_backend"),
    [(NumpyBatchBackend, "numba_cpu"), (NumbaCpuBatchBackend, "numpy_ref")],
)
def test_concrete_batch_checks_family_without_the_session_type_hook(
    factory, wrong_backend
):
    assert_family_refused(factory, wrong_backend)


@pytest.mark.parametrize("collar", [False, True], ids=["no collar", "in-stage collar"])
def test_numba_marches_a_problem_without_responses(narrow_grid, linear_problem, collar):
    assert_marches_without_responses(
        NumbaCpuBackend, narrow_grid, linear_problem, collar
    )


@pytest.mark.parametrize("route", ROUTES)
def test_numba_sources_the_analytic_field_and_the_envelope_plasma(route):
    assert_sources_the_reference(NumbaCpuBackend, ROUTES[route])


def test_numba_cpu_admits_the_leaf_on_the_fused_path():
    simulation = capillary("symmetric-kerr")
    problem = simulation.translate()
    NumbaCpuBackend(problem, grid=simulation.grid).close()


@pytest.mark.parametrize("missing", ["plans", "coefficients", "geometry"])
def test_cartesian_numba_stage_names_missing_resources(
    free_backend_problem,  # noqa: F811 - the fixture imported above
    missing,
    monkeypatch,
):
    grid, problem = free_backend_problem
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend._hoist(0.0)
        bundle = backend._stage_scalars(0.0, 0.0)
        if missing == "geometry":
            broken = copy.copy(problem)
            transform = copy.copy(problem.transform)
            object.__setattr__(transform, "geometry", None)
            object.__setattr__(broken, "transform", transform)
            monkeypatch.setattr(backend, "_problem", broken)
        else:
            monkeypatch.setattr(
                backend, "_transverse" if missing == "plans" else "_separable", None
            )
        with pytest.raises(BackendError, match="Cartesian stage requires"):
            backend._stage_free(backend._require_state(), 0, 0.0, bundle)
    finally:
        backend.close()


def test_numba_cpu_runs_the_march_with_no_runge_kutta_buffer(
    soliton_problem, narrow_grid
):
    """A widened integrator never reads as none and runs DOPRI5 (section 6.4).

    ``numba_cpu`` builds the march's resident engine (phase 2) and none of the
    Runge-Kutta march's buffers, and its first-same-as-last switch stays off (Q13).
    """
    length = 0.1
    problem = soliton_problem(
        narrow_grid,
        1,
        save_planes=[length],
        schedule=FixedSchedule.uniform(0.0, length, length / 10),
        integrator=MARCH,
    )
    backend = NumbaCpuBackend(problem, grid=narrow_grid)
    assert backend._low_storage is not None
    assert isinstance(backend._low_storage.engine, NumbaLowStorage)
    assert not backend._fsal
    ledger = backend.memory_ledger()
    for row in (*RUNGE_KUTTA_ROWS, "advanced", "exponent", "recycled_stage"):
        assert ledger[row] == 0, row
    backend.close()


def test_a_z_table_runs_on_the_numba_host_adapter_and_the_oracle(
    tabulated_problem, field_grid
):
    """Numba's host adapter qualifies the general CPU baseline beside the oracle."""
    departure, bound, compiled = z_table_departure_and_bound(
        NumbaCpuBackend, tabulated_problem, field_grid
    )
    assert departure < bound
    assert compiled._host_phi is not None and compiled._host_prefactor is not None
    compiled.close()


@pytest.mark.parametrize("temporal", [False, True])
def test_the_numba_term_is_the_references_on_its_own_samples_and_allocates_nothing(
    absorber_case,  # noqa: F811 - the fixture imported above
    temporal,
):
    """The planned transform and real GEMM against the reference's formula.

    The reference forms ``-rate/2 x + A x`` and transforms it through ``scipy.fft``;
    ``numba_cpu`` takes the composed matrix through a real GEMM (an analytic or
    envelope row by its real view) and transforms through its own FFTW plan. Both are
    read off the same fine samples the last stage left in the plan's buffer, so what
    differs is only the two summation orders.
    """
    grid = absorber_case[0]
    problem = problem_for(absorber_case, 8)
    if temporal:
        # A ramp rather than a constant, so that the rate is read sample by sample.
        rate = np.linspace(0.0, 2000.0, grid.n_to)
        rate.setflags(write=False)
        # Beside zeros on the coarse axis: the case's zero-rate collar translates to
        # no temporal absorber at all.
        coarse = np.zeros(grid.n_t)
        coarse.setflags(write=False)
        problem = dataclasses.replace(
            problem,
            responses=dataclasses.replace(
                problem.responses, temporal_absorber=coarse, temporal_absorber_fine=rate
            ),
        )
    else:
        # No temporal collar at all, so the radial term stands alone.
        problem = dataclasses.replace(
            problem,
            responses=dataclasses.replace(
                problem.responses, temporal_absorber=None, temporal_absorber_fine=None
            ),
        )
        rate = np.zeros(grid.n_to)
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend._trial_step(0.0, LENGTH / 8)
        measured = backend._absorber_term().copy()
        samples = backend._plans.modal_time.copy()
        matrix = backend._spatial_absorber.copy()
        tracemalloc.start()
        backend._absorber_term()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    if problem.grid_kind == "field":
        samples = samples.real
    n_r, n_pol = problem.shape[1:3]
    rows = samples.reshape(n_r, -1)
    term = (-0.5 * rate * samples + (matrix @ rows).reshape(samples.shape)).reshape(
        n_r, n_pol, -1
    )
    expected = (
        passes.to_frequency(term, grid.crop_index, kind=problem.grid_kind)
        * grid.active_mask
    ).reshape(measured.shape)
    # Each time sample is an n_r-term dot product plus the rate's term, so differs by
    # gamma_(n_r+1) of its absolute summands; a bin sums n_to of those, and each FFT
    # adds about log2(n_to) roundings of the same absolute scale. Twice that, for the
    # two transforms.
    n_to = grid.n_to
    absolute = np.abs(matrix) @ np.abs(rows) + (0.5 * rate * np.abs(samples)).reshape(
        n_r, -1
    )
    operations = n_r + 1 + 4 * int(np.log2(n_to))
    bound = 2 * operations * EPS * n_to * float(np.max(absolute))
    assert bound > 0
    assert_allclose(measured, expected, rtol=0, atol=bound)
    # Nothing field-sized: one column of the fine spectrum is the smallest buffer
    # the term would need if it allocated one; what is left is Python's own views.
    assert peak < n_to * 16
    assert ledger["in_stage_absorber_term"] == measured.nbytes
    assert ledger["in_stage_absorber_spectrum"] > 0
