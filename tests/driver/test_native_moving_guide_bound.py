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
"""

import dataclasses
import warnings

import numpy as np
import pytest

import flow_parity
from general_parity import conditioning, moving_count, source_bound
from modal_provider_cases import GENERAL_CHI3, grin_grid, make_provider
from moving_guide_cases import (
    STEP_RADIUS,
    grin_taper,
    kinked_radius,
    moving_simulation,
    step_grid,
)
from sessions.runs import load_native
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.backend import _general_maps
from solprop.driver import Simulation
from solprop.modes.guide import ModalGuide, circular_pair, provider_channels
from solprop.modes.material import MaterialKerr
from solprop.stepper import FixedSchedule, Multistep
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

ENERGY = 1e-8
"""The graded-index launch's energy, J: a launch ``Phi`` of about 0.14.

Ten thousand times the picojoule of the reference's own moving-guide tests, so that the
Kerr half of the step, and the map combination it reads, is exercised rather than a
rounding of it.
"""

POSITIONS = (0.0, 2.3e-4, 4e-4, 5.5e-4, LENGTH)
"""Stage positions the one-stage checks take, m: the entrance, inside the first
section, the kink, inside the second and the exit."""

STEP_LENGTH = 1e-4
"""Length of the step-index runs, m: ten steps below the taper's advisory step."""

STEP_POSITIONS = (0.0, 3.7e-5, STEP_LENGTH)
"""Stage positions of the step-index checks, m: the ends of the run and one inside."""


def kinked():
    """The kinked graded-index taper with a Kerr core: the problem and grid."""
    grid = grin_grid()
    guide = grin_taper(kinked_radius(), kerr=True)
    simulation = moving_simulation(guide, grid, energy=ENERGY)
    simulation.settings = dataclasses.replace(simulation.settings, absorber=None)
    return simulation.translate(), simulation._run_request(), grid


def step_problem(descriptors, *, blocks=(), basis_block=None):
    """Return the tapered step-index guide with a Kerr core, ten steps.

    ``Linear(4 -> 3.6 um, 1 mm)``, silica in 1.40, on the 835 nm grid, run over its
    first :data:`STEP_LENGTH` in ten steps: below the moving flow's advisory step of
    8e-5 m on this taper, where the flow's parity bound is sharp, with a launch of
    ``Phi`` about 0.3.
    """
    grid = step_grid()
    active = grid.omega[grid.active_mask]
    provider = make_provider(
        radius=STEP_RADIUS,
        indices=(1.45, 1.40),
        descriptors=descriptors,
        kerr=MaterialKerr(chi3=GENERAL_CHI3),
    )
    frequencies = np.linspace(0.99 * float(active[0]), 1.01 * float(active[-1]), 17)
    guide = ModalGuide.from_provider(
        provider,
        frequencies,
        nodes=provider.node_rule(float(frequencies[0]), 0.0, n_r=24),
        degenerate_blocks=blocks,
        basis=None
        if basis_block is None
        else circular_pair(provider_channels(provider), basis_block),
    )
    simulation = Simulation.from_waveguide(
        (),
        STEP_LENGTH,
        provider=guide,
        grid=grid,
        wavelength=835e-9,
        fwhm=20e-15,
        energy=ENERGY,
        shape="gauss",
        save_planes=[STEP_LENGTH],
        dz_max=STEP_LENGTH / STEPS,
        dz_init=STEP_LENGTH / STEPS,
        schedule=FixedSchedule(np.full(STEPS, STEP_LENGTH / STEPS)),
        backend="numpy_ref",
        noise=None,
    )
    simulation.settings = dataclasses.replace(simulation.settings, absorber=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return simulation.translate(), simulation._run_request(), grid


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


def assert_ten_step_parity(problem, run_request, grid):
    """Assert ten fixed steps on native within the flow bound of ``numpy_ref``."""
    expected = flow_parity.run(NumpyRefBackend, problem, run_request, grid)
    scales = flow_parity.flow_scales(problem, run_request, grid, expected)
    bound = flow_parity.ten_step_bound(scales)
    got = flow_parity.run(NativeBackend, problem, run_request, grid)
    assert 0.0 < bound < 1e-9
    assert flow_parity.departure(got, expected) <= bound


def test_one_stage_of_a_moving_source_is_the_reference_within_the_bound():
    """Native's source at five stage positions of the kinked taper."""
    problem, run_request, grid = kinked()
    assert_source_parity(problem, run_request, grid, POSITIONS)


@pytest.mark.slow
def test_ten_steps_of_the_step_index_pair_are_the_reference_s():
    """``HE_11`` and ``HE_12`` on the 10 % step-index taper, with a Kerr core."""
    problem, run_request, grid = step_problem(
        (("HE", 1, 1, "even"), ("HE", 1, 2, "even"))
    )
    assert_source_parity(problem, run_request, grid, STEP_POSITIONS)
    assert_ten_step_parity(problem, run_request, grid)


@pytest.mark.slow
def test_ten_steps_of_the_complex_basis_are_the_reference_s():
    """The circular ``HE_11`` pair beside the parity pair of ``HE_12``.

    The rotated connection carries imaginary couplings between the groups and the maps
    are complex in the declared basis, so the combination runs on complex coefficients
    with real weights.
    """
    problem, run_request, grid = step_problem(
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
    assert_ten_step_parity(problem, run_request, grid)
