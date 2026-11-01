# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The hybrid planar transform on ``numba_cpu`` against the oracle.

Kernel schema 15's ``"slab"`` kind, slab modes in x times an FFT in y, on the compiled
backend: every case of ``tests/backends/planar_cases.py`` (the hybrid families launched
as a Gaussian in y, plus the analytic-signal field family and the collar alone) against
``numpy_ref`` on the validation ladder -- one linear application, one nonlinear
evaluation, one step, the family's run of at most ten steps, each statistic, a
``prop_planar`` run -- and the backend's own promises: repeatability and agreement
across thread counts within the ladder's bound, a ledger that names every buffer, and a
stage that allocates nothing.

**The bounds, stated before measuring.** The two backends perform the same
operations in the same order everywhere but three places, each of which is bounded:

- **The operator.** Both form the column's square ``q`` in the same order and differ
  chiefly in the principal square root: numpy's and numba's complex
  ``sqrt`` are each within one ulp of the exact root, so ``Re k_z`` differs by at most
  ``2 eps k_max`` and the phase over a length ``L`` by ``2 eps k_max L``, the frame's
  subtraction and the product with the step being the same rounded operations on the
  same operands. ``k_max`` is the largest ``sqrt |k^2 eps_co|`` on the active band,
  which bounds ``|k_z|`` since the wall term and ``k_y^2`` only reduce it.
- **One evaluation.** pocketfft and FFTW differ by ``32 eps`` per level of each
  transform's depth, as the Cartesian ladder counts them (``ceil log2 n_to`` in time,
  ``ceil log2 n_y`` in y); the x synthesis and projection are real GEMMs of lengths
  ``n_p`` and ``n_x`` whose dot products each round within ``gamma(n)`` of
  ``|S||a|``, and ``|P||S|`` is at most ``n_p`` times ``PS``, the pad scale times
  the identity (Cauchy-Schwarz on the modes' magnitudes), so each adds
  ``2 n_p gamma(n_x)``; a plasma scan adds ``gamma(12 n_to)`` (the Cartesian
  ladder's).
- **A run** of ``s`` steps is the operator term plus ``7 s`` evaluations: a replayed
  schedule keeps the two trajectories on the same steps, so the rounding does not
  compound as an independent integration's would.
"""

import dataclasses
import math
import tracemalloc

import numpy as np
import pytest

from backends.matched import fixed_order, stash_bound
from backends.planar_cases import CASES, planar_case
from backends.test_planar_hybrid import HALF_GAP, LENGTH
from backends.test_radial_nonlinear import derivative
from bounds import assert_within, gamma, step_evaluations
from legacy_geometry import LEGACY_GRID, bare_absorber
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numba_cpu import fft as numba_fft
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import select
from solprop.driver import Simulation, SimulationSettings, prop_planar
from solprop.grid import EnvelopeGrid
from solprop.modes import PlanarGrid, SlabMode
from solprop.noise import NoiseRealisation
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, PlanarPulse
from solprop.stats import StatsSpec, default_spec
from solprop.stepper import PIController
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.problem import Initial
from solprop.translate.spatial import separable_values

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
]

EPS = np.finfo(float).eps
UNIT = EPS / 2
FRONT_GRID = EnvelopeGrid.from_limits(
    LENGTH, 800e-9, (650e-9, 1000e-9), 600e-15, **LEGACY_GRID
)
"""Slice D1's front-end grid (``tests/driver/test_prop_planar.py``)."""
PLASMA_CASES = ("field-plasma", "envelope-plasma")
FIELD_STATISTICS = (
    "energy",
    "peak_intensity",
    "on_axis_intensity",
    "on_axis_spectrum",
    "beam_radius",
    "fwhm_r",
)
PLASMA_STATISTICS = (
    "electron_density",
    "peak_ionisation_rate",
    "species_ionisation",
    "species_electron_density",
    "species_peak_ionisation_rate",
)


def operator_bound(problem, length):
    """Return ``2 eps k_max L``, the principal roots' one ulp each over ``L``."""
    separable = problem.linear.separable
    assert separable is not None
    square = separable.k_squared[separable.active]
    return 2 * EPS * math.sqrt(float(np.max(np.abs(square)))) * length


def evaluation_bound(name, grid, space):
    """Return one evaluation's relative bound: the transforms, the x GEMMs, a scan."""
    depth = math.ceil(math.log2(grid.n_to)) + math.ceil(math.log2(space.n_y))
    bound = 32 * EPS * depth + 2 * space.n_p * gamma(space.x.size)
    if name in PLASMA_CASES:
        bound += gamma(12 * grid.n_to)
    return bound


def ladder_bound(name, grid, space, problem, steps, length):
    """Return the bound of ``steps`` replayed steps over ``length``."""
    evaluations = max(1.0, step_evaluations(problem) * steps)
    operator = operator_bound(problem, length)
    method = problem.stepper.integrator
    if method is not None:
        # The exponential formulation forms each step's propagator and its phi
        # functions afresh from h D, each to an ulp of its phase: up to order + 1 of
        # them in the predictor and again in the corrector. On the field-kerr run the
        # backends part by 9.0e-10 to 2.3e-9 with every order choice the same, the
        # rounding following the reductions' thread counts.
        operator *= 2 * (method.order + 1)
    return operator + evaluations * evaluation_bound(name, grid, space)


def with_steps(problem, steps, length):
    """Return the problem on ``steps`` equal replayed steps, saved at ``length``."""
    return dataclasses.replace(
        problem,
        stepper=dataclasses.replace(
            problem.stepper,
            schedule=FixedSchedule.uniform(0.0, length, length / steps),
            save_planes=np.array([length]),
        ),
    )


def run(backend_class, problem, grid, length, **options):
    """Propagate the problem's launch and return the state and the summary."""
    backend = backend_class(problem, grid=grid, **options)
    try:
        backend.set_initial(problem.initial.field0, None)
        summary = backend.propagate_to(length)
        return backend.observe().require_field().copy(), summary
    finally:
        backend.close()


def relative(actual, expected):
    """Return the largest departure relative to the largest reference magnitude."""
    scale = float(np.max(np.abs(expected)))
    assert scale > 0
    return float(np.max(np.abs(actual - expected))) / scale


@pytest.fixture(scope="module")
def cases():
    """Build each case once per module; construction only."""
    built = {}

    def build(name, **options):
        key = (name, *sorted(options.items()))
        if key not in built:
            built[key] = planar_case(name, **options)
        return built[key]

    return build


# ----------------------------------------------------------------- the ladder


def test_the_admitted_problem_is_the_hybrid_planar_kind(cases):
    """Slice D1's refusal is gone: the kind constructs, and its metadata says so."""
    grid, space, problem = cases("kerr")
    geometry = problem.transform.geometry
    assert geometry is not None
    assert geometry.kind == "slab"
    backend = select("numba_cpu", problem, grid=grid)
    try:
        metadata = backend.metadata()
        # Five whole y rows fit the node chunk of 204 = 5 * 37 + 19 nodes.
        assert metadata["slab_y_block"] == 5
        assert metadata["node_chunk"] == 5 * space.x.size
        assert metadata["transverse_layout"] == "columns"
    finally:
        backend.close()


@pytest.mark.parametrize("name", ["kerr", "field-kerr"])
def test_one_linear_application_is_the_reference_within_the_operator_bound(cases, name):
    """``exp(dz D)`` of every column against the reference's ``separable_values``.

    Bound: the phase term ``2 eps k_max dz`` and ``8 eps`` for the exponential itself,
    which the reference takes as one complex ``exp`` and the kernel as a real ``exp``
    and one ``sincos``.
    """
    grid, _, problem = cases(name)
    dz = grid.length / 10
    operator, _ = separable_values(problem.linear.separable)
    expected = np.exp(dz * operator[0, :, 0, :])
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        forward = np.empty_like(backend._forward[0])
        inverse = np.empty_like(forward)
        backend._planar.write_propagator(backend._slab_data, dz, forward, inverse)
    finally:
        backend.close()
    bound = operator_bound(problem, dz) + 8 * EPS
    assert np.max(np.abs(forward - expected)) <= bound
    assert np.max(np.abs(inverse * expected - 1)) <= 2 * bound
    # The wall term is in the operator: without it every column would propagate as
    # free space, and the decay of each mode would be absent.
    assert np.max(np.abs(expected[:, grid.active_mask])) < 1


@pytest.mark.parametrize("name", list(CASES))
def test_one_nonlinear_evaluation_is_the_reference_s(cases, name, record_property):
    """The first stage's derivative at the launch, on every case."""
    grid, space, problem = cases(name)
    bound = evaluation_bound(name, grid, space)
    reference = derivative(NumpyRefBackend, problem, grid)
    actual = derivative(NumbaCpuBackend, problem, grid)
    # The collar's term is -alpha E / 2 where the field is small, so its rounding is
    # relative to the field it multiplies: alpha_max max|a| / 2 joins the scale.
    geometry = problem.transform.geometry
    assert geometry is not None
    temporal = problem.responses.temporal_absorber_fine
    alpha = float(np.max(geometry.absorber_rate)) + (
        float(np.max(temporal))
        if temporal is not None and problem.responses.absorber_scheme == "nonlinear"
        else 0.0
    )
    scale = float(np.max(np.abs(reference))) + 0.5 * alpha * float(
        np.max(np.abs(problem.initial.field0))
    )
    measured = float(np.max(np.abs(actual - reference))) / scale
    record_property("measured", measured)
    record_property("bound", bound)
    assert measured <= bound


@pytest.mark.medium
@pytest.mark.parametrize("steps", [1, None])
@pytest.mark.parametrize("name", list(CASES))
def test_a_step_and_a_run_are_the_reference_s(cases, name, steps, record_property):
    """One step, and the family's run of ten steps (two for the envelope plasma).

    The run also has to move the state by far more than the bound, so the parity is
    not of two copies of the launch; and a planar run with the nonlinear terms
    removed departs from it by far more, so they are exercised.
    """
    grid, space, problem = cases(name)
    schedule = problem.stepper.schedule
    assert schedule is not None
    count = schedule.steps.size if steps is None else steps
    # One step is the family's own step; the run is the family's whole schedule.
    length = grid.length * count / schedule.steps.size
    if steps is not None:
        problem = with_steps(problem, count, length)
    bound = ladder_bound(name, grid, space, problem, count, length)
    reference, summary = run(NumpyRefBackend, problem, grid, length)
    actual, ours = run(NumbaCpuBackend, problem, grid, length)
    assert ours.n_steps == summary.n_steps == count
    measured = relative(actual, reference)
    record_property("measured", measured)
    record_property("bound", bound)
    assert measured <= bound
    assert relative(problem.initial.field0, reference) > 1e3 * bound


@pytest.mark.medium
@pytest.mark.parametrize("name", ["kerr", "raman", "field-plasma", "field-kerr"])
def test_the_responses_are_exercised(cases, name):
    """The runs above are nonlinear: removing the responses moves the state."""
    grid, space, problem = cases(name)
    linear = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses, response_order=problem.responses.response_order[:1]
        ),
    )
    schedule = problem.stepper.schedule
    assert schedule is not None
    bound = ladder_bound(name, grid, space, problem, schedule.steps.size, grid.length)
    nonlinear, _ = run(NumbaCpuBackend, problem, grid, grid.length)
    plain, _ = run(NumbaCpuBackend, linear, grid, grid.length)
    assert relative(plain, nonlinear) > 1e3 * bound


# ------------------------------------------------------------- the statistics


@pytest.mark.medium
@pytest.mark.parametrize("name", ["kerr", "field-plasma", "envelope-plasma"])
def test_each_statistic_is_the_reference_s_on_the_same_state(cases, name):
    """Every spatial statistic, and each plasma one, at the reference's final state.

    The field statistics are the one shared host function on both backends
    (:func:`solprop.backends.spatial.collect_spatial` synthesising the node volume
    itself), so they agree to that function's own reductions, the Cartesian
    statistics' ``gamma`` (a width three y samples cannot resolve is nan on both).
    The plasma statistics are the compiled
    columns against the reference's passes on nodes that differ by one evaluation's
    rounding; each is bounded by its own measured response to a one-ulp launch
    perturbation (the largest of three random-sign draws) times that rounding in
    ulps, plus ``gamma(n_to)`` for either backend's own scan.
    """
    grid, space, problem = cases(name)
    plasma = name in PLASMA_CASES
    names = FIELD_STATISTICS + (PLASMA_STATISTICS if plasma else ())
    spec = StatsSpec(names=names)
    state, _ = run(NumpyRefBackend, problem, grid, grid.length)

    def statistics(backend_class, field):
        backend = backend_class(problem, grid=grid)
        try:
            backend.set_initial(field, None)
            return backend.stats(spec)
        finally:
            backend.close()

    theirs = statistics(NumpyRefBackend, state)
    ours = statistics(NumbaCpuBackend, state)
    assert set(ours) == set(theirs) == set(names)
    reductions = gamma(128 * (space.n_columns + space.n_nodes) + 32 * grid.n_to)
    for field_name in FIELD_STATISTICS:
        np.testing.assert_allclose(
            ours[field_name], theirs[field_name], rtol=reductions, err_msg=field_name
        )
    assert float(np.ravel(ours["energy"])[0]) > 0
    if not plasma:
        return
    generator = np.random.default_rng(49)
    response = dict.fromkeys(PLASMA_STATISTICS, 0.0)
    for _ in range(3):
        signs = generator.choice([-1.0, 1.0], size=state.shape)
        moved = statistics(NumpyRefBackend, state * (1 + UNIT * signs))
        for stat in PLASMA_STATISTICS:
            scale = np.maximum(np.abs(theirs[stat]), np.max(np.abs(theirs[stat])))
            change = np.max(np.abs(moved[stat] - theirs[stat]) / scale)
            response[stat] = max(response[stat], float(change))
    seeds = evaluation_bound(name, grid, space) / UNIT
    scan = gamma(2 * grid.n_to)
    for stat in PLASMA_STATISTICS:
        scale = np.max(np.abs(theirs[stat]))
        assert scale > 0, stat
        bound = seeds * response[stat] + scan
        assert np.max(np.abs(ours[stat] - theirs[stat])) <= bound * scale, stat


@pytest.mark.medium
def test_the_stash_is_the_fresh_scan(cases):
    """The plasma columns a step leaves behind are what a fresh scan of its state reads.

    With the first-same-as-last tableau the last stage's input is the accepted state,
    so the between-planes channel reads the stash and a save plane scans afresh; the
    two routes must give the same numbers to one synthesis's rounding carried through
    the rate (:func:`matched.stash_bound`), and the channel must be the stash.
    """
    grid, _, problem = cases("field-plasma")
    names = ("electron_density", "peak_ionisation_rate")
    # The stash is the first-same-as-last tableau's: the Runge-Kutta march's.
    problem = dataclasses.replace(
        problem,
        stats_spec=StatsSpec(names=names, periodic=names, period=1),
        stepper=dataclasses.replace(problem.stepper, integrator=None),
    )
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(grid.length)
        assert backend._stash_current
        held = backend._require_state()[0]
        stashed = backend._ionisation(held)
        backend._stash_current = False
        fresh = backend._ionisation(held)
        rows = backend.observe().step_stats
    finally:
        backend.close()
    rate = problem.responses.plasma_coefficients.rate
    bound = stash_bound(
        rate, float(np.max(fresh[1])), (rate.field_min, rate.field_max), grid.n_to
    )
    for one, other in zip(stashed, fresh, strict=True):
        assert_within(one, other, bound)
    assert rows.n_rows == problem.stepper.schedule.steps.size
    assert float(stashed[0]) > 0


@pytest.mark.medium
def test_a_modified_noise_draw_rides_the_slab_phase(cases):
    """The noise is carried by each column's own unitary phase, wall term included.

    A realisation the translation would refuse beside a planar grid, set on the
    contract directly: both backends add ``N0 exp(i z Im D)`` at every stage, and
    one step with it agrees within the one-step bound plus the phase's
    ``2 eps k_max z``.
    """
    grid, space, problem = cases("kerr")
    length = grid.length / 10
    problem = with_steps(problem, 1, length)
    generator = np.random.default_rng(7)
    shape = problem.initial.field0.shape
    noise = (
        generator.standard_normal(shape) + 1j * generator.standard_normal(shape)
    ) * (1e-3 * np.max(np.abs(problem.initial.field0)))
    noise[..., ~grid.active_mask] = 0
    problem = dataclasses.replace(
        problem,
        initial=Initial(problem.initial.field0, noise, "modified", {}),
    )
    realisation = NoiseRealisation(noise[0], None, "modified")
    states = []
    for backend_class in (NumpyRefBackend, NumbaCpuBackend):
        backend = backend_class(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, realisation)
            phase = backend._unitary_phase(backend._stage_scalars(length, length))
            backend.propagate_to(length)
            states.append((backend.observe().require_field().copy(), np.array(phase)))
        finally:
            backend.close()
    (reference, expected), (actual, phase) = states
    bound = ladder_bound("kerr", grid, space, problem, 1, length)
    assert relative(actual, reference) <= bound
    assert np.max(np.abs(phase - expected)) <= operator_bound(problem, length)
    assert np.max(np.abs(expected)) > 0


# ------------------------------------------------------------- the entry point


@pytest.mark.medium
def test_prop_planar_runs_on_numba_as_on_the_reference():
    """``prop_planar(..., backend="numba_cpu")`` end to end: fields and statistics.

    Ten Kerr steps of the hybrid families' front-end launch through the user's door,
    with no room outside the aperture (``collar_fraction = 0``, the default) and the
    free-space statistics; the saved plane agrees within the run's bound and the
    statistics within twice it (squares of the field) plus their own reductions'
    ``gamma``, the Cartesian ladder's.
    """
    space = PlanarGrid(
        [SlabMode(HALF_GAP, "Ar", 1.0, order=p) for p in (1, 3)], 1.5e-3, 16
    )
    launch = PlanarPulse(
        GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3),
        400e-6,
        waist_x=0.735 * HALF_GAP,
    )
    responses = [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)]
    outputs = {}
    for backend in ("numpy_ref", "numba_cpu"):
        settings = SimulationSettings(
            controller=PIController(rtol=1e-8, atol=1e-12),
            save_planes=[LENGTH],
            dz_max=LENGTH / 10,
            dz_init=LENGTH / 10,
            schedule=FixedSchedule(np.full(10, LENGTH / 10)),
            backend=backend,
            stats=default_spec("free"),
            absorber=bare_absorber(FRONT_GRID),
        )
        outputs[backend] = prop_planar(
            FRONT_GRID, space, responses, launch, settings=settings, progress="none"
        )
    problem = Simulation.from_planar(
        FRONT_GRID, space, responses, launch, settings=settings
    ).translate()
    reference, actual = outputs["numpy_ref"], outputs["numba_cpu"]
    assert actual.metadata is not None
    assert actual.metadata.simulation_kind == "slab"
    fields, expected = actual.fields("fft"), reference.fields("fft")
    bound = ladder_bound("kerr", FRONT_GRID, space, problem, 10, LENGTH)
    assert relative(fields[-1], expected[-1]) <= bound
    assert relative(expected[0], expected[-1]) > 1e3 * bound
    statistics = 2 * bound + gamma(128 * space.n_columns + 32 * FRONT_GRID.n_to)
    for name in default_spec("free").names:
        ours = np.asarray(actual[f"stats/{name}"])
        theirs = np.asarray(reference[f"stats/{name}"])
        scale = float(np.max(np.abs(theirs)))
        assert scale > 0, name
        assert np.max(np.abs(ours - theirs)) <= statistics * scale, name


# ------------------------------------------------------- this backend's own promises


@pytest.mark.medium
def test_two_runs_agree(cases):
    """Repeatability: two runs of one fixed-order method on one schedule.

    The same computation twice, held to the family's ladder bound
    (:func:`ladder_bound`), not to its bits: a threaded reduction may order its sums
    differently from one run to the next.
    """
    grid, space, problem = cases("kerr-collar")
    problem = fixed_order(problem)
    first, _ = run(NumbaCpuBackend, problem, grid, grid.length)
    second, _ = run(NumbaCpuBackend, problem, grid, grid.length)
    schedule = problem.stepper.schedule
    assert schedule is not None
    steps = schedule.steps.size
    bound = ladder_bound("kerr-collar", grid, space, problem, steps, grid.length)
    assert relative(second, first) <= bound


@pytest.mark.medium
@pytest.mark.parametrize("name", ["kerr-collar", "field-plasma"])
def test_one_thread_and_four_agree(thread_budget_lifted, cases, name, monkeypatch):
    """The large regime's parallel kernels, forced on a small case, are the serial ones.

    The cutoff of the parallel spatial passes is lowered so that the planar finish and
    propagator, the stage combine and the accumulation all compile with ``prange``
    beside the response column's parallel node loop; every loop is over independent
    columns or nodes, so the state depends on the thread count by the order of its
    sums at most: one fixed-order method on one schedule, within the ladder bound.
    """
    grid, space, problem = cases(name)
    problem = fixed_order(problem)
    serial, _ = run(NumbaCpuBackend, problem, grid, grid.length, threads=1)
    monkeypatch.setattr(numba_fft, "REGIME_ELEMENTS", 1)
    backend = NumbaCpuBackend(problem, grid=grid, threads=4, regime="large")
    try:
        assert backend._parallel_spatial
        assert backend._spec.parallel
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(grid.length)
        parallel = backend.observe().require_field().copy()
    finally:
        backend.close()
    schedule = problem.stepper.schedule
    assert schedule is not None
    steps = schedule.steps.size
    bound = ladder_bound(name, grid, space, problem, steps, grid.length)
    assert relative(parallel, serial) <= bound


def test_the_ledger_names_the_planar_buffers(cases):
    """The column volume (both halves beside an in-stage term), the row rate, plans."""
    grid, space, problem = cases("kerr-split")
    geometry = problem.transform.geometry
    assert geometry is not None
    bare = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses, temporal_absorber=None, temporal_absorber_fine=None
        ),
        transform=dataclasses.replace(
            problem.transform,
            geometry=dataclasses.replace(
                geometry, absorber_rate=np.zeros_like(geometry.absorber_rate)
            ),
        ),
    )
    field_bytes = space.n_columns * grid.n_omega * 16
    for case, halves in ((problem, 2), (bare, 1)):
        backend = NumbaCpuBackend(case, grid=grid)
        try:
            assert backend._in_stage_absorber == (halves == 2)
            ledger = backend.memory_ledger()
            assert ledger["transverse_scratch"] == halves * field_bytes
            assert ledger["slab_row_rate"] == space.n_columns * 8
            # The matrices and their taper-scaled copies, real.
            assert ledger["synthesis"] == ledger["projection"]
            assert ledger["synthesis"] == 2 * space.x.size * space.n_p * 8
            # The temporal plans hold a block's (p, y) rows; the node buffers its
            # (x, y) nodes.
            rows = backend._slab_block * space.n_p
            assert ledger["modal_time"] == rows * grid.n_to * 16
            assert ledger["node_field"] == (
                backend._slab_block * space.x.size * grid.n_to * 16
            )
            assert ledger["transverse_fft_plan_heap_growth"] >= 0
        finally:
            backend.close()


@pytest.mark.medium
def test_a_stage_allocates_no_buffer(cases):
    """Every buffer is built once: a warm stage allocates nothing field-sized.

    Measured with ``tracemalloc`` after a warm-up stage, against the smallest buffer a
    stage would need if it allocated one -- a single row on the fine grid -- so what
    is left is Python's own small objects.
    """
    grid, _, problem = cases("kerr-collar")
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(grid.length / 10)
        state = backend._require_state().reshape(backend._n_columns, grid.n_omega)
        dz = grid.length / 10
        bundle = backend._stage_scalars(dz, 1.5 * dz)
        backend._stage(state, 1, dz, bundle)
        tracemalloc.start()
        backend._stage(state, 1, dz, bundle)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        backend.close()
    assert peak < grid.n_to * 16


@pytest.mark.medium
def test_a_linear_run_with_the_collar_split_off_is_the_reference_s(cases):
    """No response and no in-stage absorber: every stage derivative is zero.

    The Kerr case with only its accumulation compiled and ``absorber_scheme=
    "split_step"``, so no term enters the stages and the linear flow alone carries the
    state (the case has no y collar); the bound is the ladder's.
    """
    grid, space, problem = cases("kerr")
    linear = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses,
            response_order=problem.responses.response_order[-1:],
            absorber_scheme="split_step",
        ),
    )
    schedule = linear.stepper.schedule
    assert schedule is not None
    bound = ladder_bound("kerr", grid, space, linear, schedule.steps.size, grid.length)
    expected, _ = run(NumpyRefBackend, linear, grid, grid.length)
    actual, _ = run(NumbaCpuBackend, linear, grid, grid.length)
    assert relative(actual, expected) < bound
