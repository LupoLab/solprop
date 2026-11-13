# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A moving guide's source and ten steps on native, at the bounds derived for them.

SEP 43's inventory items 23 and 24. The retired Numba tests held one stage's source at
five positions of the real kinked taper, ``source_bound`` below 1e-11, and ten steps of
the step-index pair and of the complex basis at the flow's bound, below 1e-9, at 1e-8 J
over 1e-4 m, with ``imag(connection) > 1`` asserted. What survived was the C++ synthetic
source and a looser ten-step run (1e-9 J over 2e-4 m, ``moving_parity_bound`` with no
ceiling). The bounds and cases here are the originals', from ``general_parity`` and
``flow_parity``, fixed before the compiled run. Native's source at a position is its
resident Adams engine's first history sample there, the same stage kernel at that
position's weights; the reference's is its ``_adams_lab_source``.

The step-index cases' ten steps are rows of the case catalogue
(``tests/backends/flow_rows.py``, ``moving guide bound ...``), guarded by the maps
frozen at the launch; the sources stay here, since the catalogue's source rung takes a
zero-length trial, which a coupled flow refuses.
"""

import dataclasses

import numpy as np
import pytest

import flow_parity
from general_parity import conditioning, moving_count, source_bound
from moving_guide_cases import (
    BOUND_STEP_LENGTH,
    bound_kinked,
    bound_step_problem,
)
from sessions.runs import load_native
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.backend import _general_maps
from solprop.stepper import Multistep
from solprop.translate.stage import scalars

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.GeneralModalBasisWarning"
)
"""The step-index guides warn on the active band's basis, which is expected here."""

solprop_native = pytest.importorskip("solprop_native")
NativeBackend = solprop_native.NativeBackend

LENGTH = 1e-3
"""Length of the graded-index run, m."""

STEPS = 10
"""Fixed steps of every run."""


POSITIONS = (0.0, 2.3e-4, 4e-4, 5.5e-4, LENGTH)
"""Stage positions the one-stage checks take, m: the entrance, inside the first
section, the kink, inside the second and the exit."""


STEP_POSITIONS = (0.0, 3.7e-5, BOUND_STEP_LENGTH)
"""Stage positions of the step-index checks, m: the ends of the run and one inside."""


def source_at(implementation, problem, run_request, grid, positions):
    """Return each position's lab-frame source on the launch."""
    state = problem.initial.field0
    if implementation is NumpyRefBackend:
        backend = implementation(problem, run_request, grid=grid)
        try:
            backend.set_initial(state, None)
            # The backend's private hook keeps the state's leading axis of one.
            held = state[np.newaxis]
            return [np.array(backend._adams_lab_source(held, z)) for z in positions]
        finally:
            backend.close()
    # Native's resident Adams engine, on the same problem under the Adams march: its
    # history's first sample, which the engine evaluates at the native state loaded
    # with the launch, is the complete source there in the lab frame, the stage
    # kernel reading that position's weights from its tables.
    assert problem.initial.noise0 is None
    stepper = dataclasses.replace(run_request.stepper, integrator=Multistep())
    backend = implementation(problem, run_request.with_stepper(stepper), grid=grid)
    try:
        engine = backend._resident
        assert type(engine).__name__ == "_ResidentAdams"
        sources = []
        for z in positions:
            load_native(backend, state, clear_fsal=False)
            engine.start(float(z))
            sources.append(np.array(engine.snapshot_rows()["store"][0]))
        return sources
    finally:
        backend.close()


def assert_source_parity(problem, run_request, grid, positions):
    """Assert one stage's source within its bound at each position.

    The bound is ``source_bound``'s with the projection's conditioning of the maps at
    the stage and the combination's ``moving_count`` at its weights, formed before the
    compiled source is evaluated, on the launch state.
    """
    maps = _general_maps(problem)
    assert maps is not None
    assert maps.moving
    state = problem.initial.field0
    expected = source_at(NumpyRefBackend, problem, run_request, grid, positions)
    got = source_at(NativeBackend, problem, run_request, grid, positions)
    for position, mine, theirs in zip(positions, got, expected, strict=True):
        weights = scalars(problem, 0.0, position).transform_weights
        at = maps.at(weights)
        bound = source_bound(
            at,
            grid.n_to,
            conditioning(at, state, grid),
            moving_count(maps, [weights]),
        )
        assert 0.0 < bound < 1e-11
        assert np.max(np.abs(theirs)) > 0.0
        assert flow_parity.departure(mine, theirs) <= bound
        # Vacuous if native's maps stayed at the entrance: the position must show.
        if position != positions[0]:
            assert flow_parity.departure(mine, got[0]) > 1e3 * bound


def test_one_stage_of_a_moving_source_is_the_reference_within_the_bound():
    """Native's source at five stage positions of the kinked taper."""
    problem, run_request, grid = bound_kinked()
    assert_source_parity(problem, run_request, grid, POSITIONS)


@pytest.mark.slow
def test_the_step_index_pair_s_source_is_the_reference_s():
    """``HE_11`` and ``HE_12`` on the 10 % step-index taper, with a Kerr core."""
    problem, run_request, grid = bound_step_problem(
        (("HE", 1, 1, "even"), ("HE", 1, 2, "even"))
    )
    assert_source_parity(problem, run_request, grid, STEP_POSITIONS)


@pytest.mark.slow
def test_the_complex_basis_s_source_is_the_reference_s():
    """The circular ``HE_11`` pair beside the parity pair of ``HE_12``.

    The rotated connection carries imaginary couplings between the groups and the maps
    are complex in the declared basis, so the combination runs on complex coefficients
    with real weights.
    """
    problem, run_request, grid = bound_step_problem(
        (
            ("HE", 1, 1, "even"),
            ("HE", 1, 1, "odd"),
            ("HE", 1, 2, "even"),
            ("HE", 1, 2, "odd"),
        ),
        blocks=((0, 1), (2, 3)),
        basis_block=(0, 1),
    )
    flow = problem.linear.coupled_flow
    assert flow is not None
    assert flow.connection is not None
    assert float(np.max(abs(np.imag(flow.connection)))) > 1.0
    assert_source_parity(problem, run_request, grid, STEP_POSITIONS)
