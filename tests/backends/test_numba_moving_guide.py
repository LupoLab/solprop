# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""``numba_cpu`` executes a moving solid guide: parity, buffers and admission.

A moving guide's synthesis and projection maps are Chebyshev coefficients along z,
combined per stage over the bundle's ``transform_weights`` (kernel schema 14), and its
connection rides the ``"frozen_residual"`` coupled flow that this backend already
compiles. What is new on this backend is the per-stage combination, one compiled kernel
into two stage buffers allocated once, so every assertion here is against ``numpy_ref``
on the same problem:

* **parity, bound first**: the combined maps within the rounding of a sum of
  ``n_deg`` terms, element by element; one stage's source within the general route's
  source bound plus the combination's term (``tests/general_parity.py``); ten fixed
  steps within the coupled flow's bound carrying that source bound
  (``tests/flow_parity.py``) -- on the kinked graded-index taper here, and on the
  step-index ``HE_11``/``HE_12`` taper and the complex (circular) basis case in the slow
  tier, each with a Kerr core so that the source, and with it the combination, runs;
* **a constant guide is what it was**: its stage reads the reference's arrays and
  never combines;
* **buffers**: the ledger's moving rows, the coefficients held as views and nothing
  allocated in a stage; the ``combine`` kernel itself, parallel against serial, is
  ``tests/backends/test_numba_general_kernels.py``'s (run interpreted by the coverage
  lane), and the memory estimate ``tests/backends/test_memory.py``'s.

No run here is longer than a millimetre, in ten steps.
"""

import dataclasses
import tracemalloc
from typing import cast

import numpy as np
import pytest

import flow_parity
from driver.test_moving_guide import (
    STEP_RADIUS,
    build,
    grin_taper,
    kinked_radius,
    step_grid,
)
from general_parity import (
    UNIT,
    conditioning,
    moving_count,
    source_bound,
)
from modal_provider_cases import GENERAL_CHI3, grin_grid, make_provider
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.backend import _general_maps
from solprop.backends.numpy_ref.general_modal import general_source
from solprop.driver import Simulation
from solprop.exceptions import BackendError
from solprop.modes.guide import ModalGuide, circular_pair, provider_channels
from solprop.modes.material import MaterialKerr
from solprop.output.memory import MemoryOutput
from solprop.stepper import FixedSchedule
from solprop.translate.stage import scalars

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.GeneralModalBasisWarning"
)
"""The step-index guides warn on the active band's basis, which is expected here."""

LENGTH = 1e-3
"""Length of every run here, m: a millimetre keeps each run short."""

STEPS = 10
"""Fixed steps of every run: ten, to keep each run short."""

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


@pytest.fixture(scope="module")
def kinked():
    """The kinked graded-index taper with a Kerr core: the guide, problem and grid."""
    grid = grin_grid()
    guide = grin_taper(kinked_radius(), kerr=True)
    simulation = build(guide, grid, energy=ENERGY)
    simulation.settings = dataclasses.replace(simulation.settings, absorber=None)
    return guide, simulation.translate(), grid


def ten_step_parity(problem, grid, record_property, name):
    """Assert ten fixed steps on ``numba_cpu`` within the flow bound of ``numpy_ref``.

    The bound is ``flow_parity.ten_step_bound`` on the case's own scales, read before
    the compiled run, whose per-stage source bound carries the map combination's term.
    Slice E6's formation bound of the flow's actions grows as ``exp(2 ||h G||)`` and
    is loose where ``h K`` exceeds one -- ten 100 um steps on the step-index taper
    bound the departure by 3e-2 -- so the step-index cases step below the advisory
    step (:func:`step_problem`). Returns the reference's final state.
    """
    expected = flow_parity.run(NumpyRefBackend, problem, grid)
    scales = flow_parity.flow_scales(problem, grid, expected)
    bound = flow_parity.ten_step_bound(scales)
    got = flow_parity.run(NumbaCpuBackend, problem, grid)
    measured = flow_parity.departure(got, expected)
    record_property(f"{name}_ten_step_bound", bound)
    record_property(f"{name}_ten_step_departure", measured)
    record_property(f"{name}_phi", scales.phi)
    assert 0.0 < bound < 1e-9
    assert measured <= bound
    return expected


def source_parity(problem, grid, record_property, name, positions):
    """Assert one stage's compiled source within its bound at each position.

    The bound is ``source_bound``'s with the projection's conditioning of the maps at
    the stage and the combination's ``moving_count`` at its weights, formed before
    the compiled source is evaluated, on the launch state.
    """
    maps = _general_maps(problem)
    assert maps is not None and maps.moving
    state = problem.initial.field0[0]
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        workspace = backend._general
        assert workspace is not None
        for position in positions:
            weights = scalars(problem, 0.0, position).transform_weights
            at = maps.at(weights)
            bound = source_bound(
                at,
                grid.n_to,
                conditioning(at, state, grid),
                moving_count(maps, [weights]),
            )
            expected = general_source(
                state,
                maps,
                grid.crop_index,
                n_to=grid.n_to,
                pad_scale=grid.pad_scale,
                transform_weights=weights,
            )[:, 0]
            workspace.lab[:] = state[:, 0]
            got = workspace.source(weights)
            measured = flow_parity.departure(got, expected)
            record_property(f"{name}_source_bound_{position:.2e}", bound)
            record_property(f"{name}_source_departure_{position:.2e}", measured)
            assert 0.0 < bound < 1e-11
            assert measured <= bound
    finally:
        backend.close()


# ------------------------------------------------------------------------ parity


def test_the_stage_maps_are_the_reference_s_within_the_rounding_of_their_sum(kinked):
    """Each combined element within ``2 gamma_(n_deg)`` of ``sum_t |w_t| |c_t|``.

    Both backends sum the ``n_deg`` terms of the stage's section (a zero weight adds
    an exact zero), each product rounded once and each addition once, so each is
    within ``gamma_(n_deg) = n_deg u / (1 - n_deg u)`` of the exact sum relative to
    the magnitudes' sum, element by element, and the two within twice it.
    """
    _, problem, grid = kinked
    maps = _general_maps(problem)
    assert maps is not None and maps.moving
    assert maps.synthesis_terms is not None and maps.projection_terms is not None
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        workspace = backend._general
        assert workspace is not None
        for position in POSITIONS:
            weights = scalars(problem, 0.0, position).transform_weights
            assert weights is not None
            n_deg = int(np.count_nonzero(weights))
            gamma = n_deg * UNIT / (1.0 - n_deg * UNIT)
            workspace.move(weights)
            expected = maps.at(weights)
            for got, want, terms in (
                (workspace.synthesis, expected.synthesis, maps.synthesis_terms),
                (workspace.projection, expected.projection, maps.projection_terms),
            ):
                majorant = np.tensordot(abs(weights), abs(terms), axes=1)
                assert np.all(abs(got - want) <= 2.0 * gamma * majorant)
    finally:
        backend.close()


def test_one_stage_of_a_moving_source_is_the_reference_within_the_bound(
    kinked, record_property
):
    """The compiled source at five stage positions, against the reference's."""
    _, problem, grid = kinked
    source_parity(problem, grid, record_property, "kinked", POSITIONS)


@pytest.mark.medium
def test_ten_steps_of_the_kinked_graded_index_taper_are_the_reference_s(
    kinked, record_property
):
    """Ten fixed steps across the kink, each backend from the same launch.

    The nonlinear half is exercised: the same launch through the same taper with no
    Kerr core departs from the answer by far more than the bound.
    """
    _, problem, grid = kinked
    expected = ten_step_parity(problem, grid, record_property, "kinked")
    linear = build(grin_taper(kinked_radius(), kerr=False), grid, energy=ENERGY)
    linear.settings = dataclasses.replace(linear.settings, absorber=None)
    alone = flow_parity.run(NumpyRefBackend, linear.translate(), grid)
    assert flow_parity.departure(alone, expected) > 1e-6


def step_problem(descriptors, *, blocks=(), basis_block=None):
    """Return the tapered step-index guide with a Kerr core, ten steps.

    ``Linear(4 -> 3.6 um, 1 mm)``, silica in 1.40, on the 835 nm grid, run over its
    first :data:`STEP_LENGTH` in ten steps: below the moving flow's advisory step of
    8e-5 m on this taper, where the flow's parity
    bound is sharp, with a launch of ``Phi`` about 0.3.
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
        lambda0=835e-9,
        fwhm=20e-15,
        energy=1e-8,
        pulseshape="gauss",
        save_planes=[STEP_LENGTH],
        backend="numpy_ref",
        dz_max=STEP_LENGTH / STEPS,
        dz_init=STEP_LENGTH / STEPS,
        schedule=FixedSchedule(np.full(STEPS, STEP_LENGTH / STEPS)),
        noise=None,
    )
    simulation.settings = dataclasses.replace(simulation.settings, absorber=None)
    return simulation.translate(), grid


@pytest.mark.slow
def test_ten_steps_of_the_step_index_pair_are_the_reference_s(record_property):
    """``HE_11`` and ``HE_12`` on the 10 % step-index taper, with a Kerr core."""
    problem, grid = step_problem((("HE", 1, 1, "even"), ("HE", 1, 2, "even")))
    source_parity(problem, grid, record_property, "step_index", STEP_POSITIONS)
    ten_step_parity(problem, grid, record_property, "step_index")


@pytest.mark.slow
def test_ten_steps_of_the_complex_basis_are_the_reference_s(record_property):
    """The circular ``HE_11`` pair beside the parity pair of ``HE_12``.

    The rotated connection carries imaginary couplings between the groups and the
    maps are complex in the declared basis, so the combination runs on complex
    coefficients with real weights.
    """
    problem, grid = step_problem(
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
    assert flow is not None and flow.connection is not None
    assert float(np.max(abs(np.imag(flow.connection)))) > 1.0
    source_parity(problem, grid, record_property, "complex_basis", STEP_POSITIONS)
    ten_step_parity(problem, grid, record_property, "complex_basis")


@pytest.mark.medium
def test_a_moving_guide_runs_end_to_end_on_numba_as_on_the_reference(kinked):
    """``Simulation.from_waveguide(provider=..., backend="numba_cpu")``, ten steps.

    The entry point's physical energy at the exit against the reference's run, within
    the ten-step bound of the state (an energy is a quadratic form of it, so twice).
    """
    guide, problem, grid = kinked
    expected = flow_parity.run(NumpyRefBackend, problem, grid)
    bound = flow_parity.ten_step_bound(flow_parity.flow_scales(problem, grid, expected))
    energies = {}
    for backend in ("numpy_ref", "numba_cpu"):
        simulation = build(guide, grid, energy=ENERGY)
        simulation.settings = dataclasses.replace(
            simulation.settings, absorber=None, backend=backend
        )
        planes = cast("MemoryOutput", simulation.run(progress="none")).planes()
        energies[backend] = float(np.sum(planes[-1].stats["physical_energy"]))
    assert energies["numba_cpu"] == pytest.approx(
        energies["numpy_ref"], rel=2.0 * bound
    )


# ------------------------------------------------------- constant guide, doors


def test_a_constant_guide_reads_the_reference_s_maps_and_never_combines(
    monkeypatch,
):
    """The constant route is untouched: no stage buffers, no combination, no weights.

    Its stage reads the reference's own arrays, shared rather than copied, so its
    arithmetic is the constant route's; a bundle carrying weights for it is refused in
    the reference's words.
    """
    grid = grin_grid()
    simulation = build(grin_taper(25e-6, kerr=True), grid)
    simulation.settings = dataclasses.replace(simulation.settings, absorber=None)
    problem = simulation.translate()
    maps = _general_maps(problem)
    assert maps is not None and not maps.moving
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        workspace = backend._general
        tables = problem.transform.modal_tables
        assert workspace is not None and tables is not None
        assert np.shares_memory(workspace.synthesis, tables.synthesis)
        assert np.shares_memory(workspace.projection, tables.projection)

        def refuse(*_):
            raise AssertionError

        monkeypatch.setattr(
            workspace, "kernels", workspace.kernels._replace(combine=refuse)
        )
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(LENGTH / STEPS)
        with pytest.raises(BackendError, match="maps that do not move"):
            workspace.move(np.ones(1))
        assert workspace.memory_ledger()["general_stage_maps"] == 0
    finally:
        backend.close()


def test_a_moving_stage_without_its_weights_is_refused(kinked):
    """A moving guide's maps come from the bundle's weights, or not at all."""
    _, problem, grid = kinked
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        workspace = backend._general
        assert workspace is not None
        with pytest.raises(BackendError, match="this stage carries none"):
            workspace.source()
    finally:
        backend.close()


def test_numba_refuses_moving_maps_without_their_flow_in_the_reference_s_words(kinked):
    """The shared admission list: moving maps with no flow, and CF4 beside them."""
    _, problem, grid = kinked
    linear = dataclasses.replace(problem.linear)
    object.__setattr__(linear, "coupled_flow", None)
    spoiled = dataclasses.replace(problem)
    object.__setattr__(spoiled, "linear", linear)
    for backend_class in (NumpyRefBackend, NumbaCpuBackend):
        name = "numpy_ref" if backend_class is NumpyRefBackend else "numba_cpu"
        with pytest.raises(BackendError, match=f"{name} executes a moving guide's"):
            backend_class(spoiled, grid=grid)


# ----------------------------------------------------------------- buffers


def test_the_ledger_counts_the_moving_maps_as_the_reference_does(kinked):
    """The coefficients are the record's, viewed; the stage maps are two buffers.

    ``general_synthesis`` and ``general_projection`` are every coefficient of each
    map, the reference's rows; ``general_stage_maps`` is the two maps a stage writes.
    ``estimate_memory`` against this ledger, to the byte, is
    ``tests/backends/test_memory.py``'s.
    """
    _, problem, grid = kinked
    tables = problem.transform.modal_tables
    assert tables is not None and tables.synthesis is not None
    one = tables.synthesis[0, 0].nbytes
    backend = NumbaCpuBackend(problem, grid=grid)
    reference = NumpyRefBackend(problem, grid=grid)
    try:
        ledger = backend.memory_ledger()
        shared = reference.memory_ledger()
        workspace = backend._general
        assert workspace is not None
        for held, record in zip(
            workspace._terms, (tables.synthesis, tables.projection), strict=True
        ):
            assert np.shares_memory(held, record)
    finally:
        backend.close()
        reference.close()
    assert ledger["general_synthesis"] == 10 * one
    assert ledger["general_projection"] == 10 * one
    assert ledger["general_stage_maps"] == 2 * one
    for row in ("general_synthesis", "general_projection", "general_stage_maps"):
        assert ledger[row] == shared[row], row


def test_a_moving_stage_allocates_no_buffer(kinked):
    """The combination writes into the stage buffers: nothing sizeable is allocated.

    Measured with ``tracemalloc`` after a warm-up call, against one node's three
    components on the fine grid, the smallest buffer the pass would need if it
    allocated one; one stage map is far larger.
    """
    _, problem, grid = kinked
    weights = scalars(problem, 0.0, 5.5e-4).transform_weights
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        workspace = backend._general
        assert workspace is not None
        workspace.lab[:] = problem.initial.field0[0, :, 0]
        workspace.source(weights)
        tracemalloc.start()
        workspace.source(weights)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert workspace.synthesis.nbytes > 100 * peak
    finally:
        backend.close()
    assert peak < 3 * grid.n_to * 16
