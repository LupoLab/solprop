# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""``numba_cpu`` executes the general modal route: parity, admission and buffers.

The compiled route is a second execution path in ``numba_cpu`` beside the fused legacy
pipeline, with the reference backend as its oracle, so every assertion here is against
``numpy_ref`` on the same problem:

* **parity**, to the rounding bound of ``tests/general_parity.py``, written before
  the measurement from the operation counts and the projection's conditioning --
  one stage's source on each parity case, and ten fixed steps of each;
* **admission**, which is the reference's own list run under this backend's name,
  so the two refuse the same problems with the same words;
* **buffers**: the ledger rows, the plans built once, and no allocation inside a
  stage beyond Python's own small objects.

The kernels are compared with the reference arithmetic one by one, interpreted, in
``tests/backends/test_numba_general_kernels.py``.
"""

import dataclasses
import tracemalloc
import warnings
from contextlib import ExitStack

import numpy as np
import pytest

from bounds import higham_gamma
from general_parity import (
    CASES,
    LENGTH,
    build_case,
    conditioning,
    launch,
    nonlinear_scale,
    problem_for,
    reference_source,
    scaled_case,
    source_bound,
    ten_step_bound,
)
from modal_provider_cases import GENERAL_CHI3, general_grid, split_guide
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numba_cpu import general_modal as compiled_module
from solprop.backends.numba_cpu.batch import NumbaCpuBatchBackend
from solprop.backends.numba_cpu.fft import build_general_plans
from solprop.backends.numba_cpu.general_modal import GENERAL_LEDGER_ROWS
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.backend import _general_maps
from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.exceptions import BackendError, GeneralModalBasisWarning
from solprop.output.memory import MemoryOutput
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule
from solprop.translate.coupled import ModalMetricTables
from solprop.translate.general import general_modal_route
from solprop.translate.responses import regional_responses

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.GeneralModalBasisWarning"
)
"""The graded-index guide warns on its long-wavelength end, which is expected here."""

HEAVY = {"grin", "raman"}
"""Cases whose guide or window makes a run seconds rather than milliseconds."""


def marks(name):
    """Mark the heavier parity cases medium."""
    return pytest.param(name, marks=pytest.mark.medium) if name in HEAVY else name


@pytest.fixture(scope="module")
def cases():
    """Build each parity case once for the module, on demand."""
    built = {}

    def get(name):
        if name not in built:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", GeneralModalBasisWarning)
                built[name] = scaled_case(name)
        return built[name]

    return get


def run(backend_class, problem, grid, length=LENGTH):
    """Return the final state of ten fixed steps on one backend."""
    backend = backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(length)
        return backend.observe().require_field().copy()
    finally:
        backend.close()


def departure(got, expected):
    """Return the relative L2 departure of two states."""
    return float(np.linalg.norm(got - expected) / np.linalg.norm(expected))


# ------------------------------------------------------------------------ parity


@pytest.mark.parametrize("name", [marks(one) for one in CASES])
def test_one_stage_of_the_source_is_the_reference_within_the_bound(
    cases, name, record_property
):
    """The compiled source on the launch state, against the reference's.

    The bound is ``source_bound``'s, formed from the case's own channel and node
    counts, transform length and the projection's measured conditioning before the
    compiled source is evaluated.
    """
    grid, guide, route, state = cases(name)
    problem = problem_for(grid, guide, route, state)
    maps = _general_maps(problem)
    assert maps is not None
    bound = source_bound(maps, grid.n_to, conditioning(maps, state, grid))
    expected = reference_source(maps, state, grid)
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        workspace = backend._general
        assert workspace is not None
        workspace.lab[:] = state[:, 0]
        got = workspace.source()
        measured = departure(got, expected)
    finally:
        backend.close()
    record_property(f"{name}_source_bound", bound)
    record_property(f"{name}_source_departure", measured)
    assert measured <= bound


@pytest.mark.parametrize("name", [marks(one) for one in CASES])
def test_ten_steps_are_the_reference_within_the_bound(cases, name, record_property):
    """Ten fixed steps on each backend, from the same launch.

    ``ten_step_bound`` carries the source's bound through the step at the launch's
    nonlinear scale and adds both backends' rounding of the linear half; the
    schedule is fixed, so the two step through the same positions.
    """
    grid, guide, route, state = cases(name)
    problem = problem_for(grid, guide, route, state)
    maps = _general_maps(problem)
    assert maps is not None
    bound = ten_step_bound(
        nonlinear_scale(maps, state, grid),
        source_bound(maps, grid.n_to, conditioning(maps, state, grid)),
    )
    expected = run(NumpyRefBackend, problem, grid)
    got = run(NumbaCpuBackend, problem, grid)
    measured = departure(got, expected)
    record_property(f"{name}_ten_step_bound", bound)
    record_property(f"{name}_ten_step_departure", measured)
    assert measured <= bound
    # The nonlinear half is really exercised: the launch's own linear flow alone is
    # far from the answer.
    linear = dataclasses.replace(route, responses=None)
    alone = run(NumpyRefBackend, problem_for(grid, guide, linear, state), grid)
    assert departure(alone, expected) > 1e3 * bound


def test_a_partial_last_chunk_runs_its_own_plans(cases, monkeypatch):
    """A chunk that does not divide the nodes: the tail plans, delayed pass too.

    The reference's chunk divides this guide's 288 nodes on its own grid, so the
    maps are replaced with a 100-node chunk -- three chunks, the last of 88 -- in
    both backends; every chunk mixes the core's owned nodes with the cladding's
    unowned ones, which the delayed pass must leave alone.
    """
    grid, guide, route, state = cases("raman")
    problem = problem_for(grid, guide, route, state)
    maps = _general_maps(problem)
    assert maps is not None
    assert maps.n_nodes % 100
    chunked = dataclasses.replace(maps, node_chunk=100)
    monkeypatch.setattr(compiled_module, "_general_maps", lambda _: chunked)
    bound = source_bound(chunked, grid.n_to, conditioning(chunked, state, grid))
    expected = reference_source(chunked, state, grid)
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        workspace = backend._general
        assert workspace is not None
        assert workspace.plans.tail_to_time is not None
        assert workspace.plans.tail_history_to_time is not None
        workspace.lab[:] = state[:, 0]
        assert departure(workspace.source(), expected) <= bound
    finally:
        backend.close()


def test_the_two_forms_and_the_delay_are_the_records_own(cases):
    """The workspace reads its form and its delay off the record, as the reference."""
    for name, form, delayed in (
        ("pair", "real", False),
        ("pair_analytic", "analytic", False),
        ("raman", "real", True),
    ):
        grid, guide, route, state = cases(name)
        backend = NumbaCpuBackend(problem_for(grid, guide, route, state), grid=grid)
        try:
            workspace = backend._general
            assert workspace is not None
            assert workspace.maps.form == form
            assert (workspace.plans.drive is not None) == delayed
            assert workspace._samples.dtype == (
                np.complex128 if form == "analytic" else np.float64
            )
        finally:
            backend.close()


@pytest.mark.medium
def test_a_linear_guide_skips_the_source_and_steps_as_the_reference(monkeypatch):
    """No responding region: no source pass, and the linear flow is the reference's.

    The split block in the provider's own coordinates, which declares
    no response at all; the bound is the ten-step one at ``Phi = 0``.
    """
    grid = general_grid()
    guide = split_guide(grid, rotated=False)
    route = general_modal_route(
        guide, grid, z_max=LENGTH, omega_reference=float(grid.omega0)
    )
    problem = problem_for(grid, guide, route, launch(grid, 2))

    def refuse(self):
        raise AssertionError("the general source ran for a linear guide")

    monkeypatch.setattr(compiled_module.GeneralWorkspace, "source", refuse)
    got = run(NumbaCpuBackend, problem, grid)
    expected = run(NumpyRefBackend, problem, grid)
    assert departure(got, expected) <= ten_step_bound(0.0, 0.0)


@pytest.mark.medium
def test_the_in_stage_absorber_rides_the_compiled_route(cases):
    """The in-stage collar on the general route: the legacy transform, then the source.

    Ten steps of 2 um: the collar's rate is stable in the explicit scheme only
    below 26 um, so the case's own millimetre schedule is refused, and the bound is
    the ten-step one at the full launch scale, which covers the shorter run.
    """
    grid, guide, route, state = cases("pair")
    length = 2e-5
    short = {
        "save_planes": [length],
        "schedule": FixedSchedule(np.full(10, length / 10)),
        "dz_max": length / 10,
        "dz_init": length / 10,
    }
    problem = problem_for(
        grid,
        guide,
        route,
        state,
        absorber=absorber(grid),
        absorber_scheme="nonlinear",
        **short,
    )
    assert problem.responses.temporal_absorber_fine is not None
    maps = _general_maps(problem)
    assert maps is not None
    bound = ten_step_bound(
        nonlinear_scale(maps, state, grid),
        source_bound(maps, grid.n_to, conditioning(maps, state, grid)),
    )
    expected = run(NumpyRefBackend, problem, grid, length)
    got = run(NumbaCpuBackend, problem, grid, length)
    assert departure(got, expected) <= bound
    plain = problem_for(grid, guide, route, state, **short)
    assert departure(run(NumpyRefBackend, plain, grid, length), expected) > 1e3 * bound


def test_the_parallel_kernels_are_the_serial_ones(cases):
    """The node loops are independent and the projection reduces per bin.

    So the result depends on how many threads ran by the order of its sums at most:
    the ``prange`` variant is compared with the serial one, on the delayed case,
    within one stage's source bound (``general_parity.source_bound``).
    """
    grid, guide, route, state = cases("raman")
    problem = problem_for(grid, guide, route, state)
    outputs = []
    for parallel in (False, True):
        workspace = compiled_module.GeneralWorkspace(
            problem, flag="estimate", fftw_threads=1, parallel=parallel
        )
        workspace.lab[:] = state[:, 0]
        outputs.append(workspace.source().copy())
    maps = _general_maps(problem)
    assert maps is not None
    bound = source_bound(maps, grid.n_to, conditioning(maps, state, grid))
    assert departure(outputs[1], outputs[0]) <= bound


# --------------------------------------------------------------------- admission


def spoil(problem, group_name, name, value):
    """Return the problem with one field of one group replaced past the hook."""
    group = dataclasses.replace(getattr(problem, group_name))
    object.__setattr__(group, name, value)
    return dataclasses.replace(problem, **{group_name: group})


def spoilers():
    """Every combination the reference refuses, as one spoiler per case.

    The same constructions as ``tests/backends/test_general_modal.py``'s reference
    refusals, so that each is asserted for both backends in one place.
    """
    n_omega = general_grid().n_omega

    def metric_of(kind, error, dense=None):
        grid = general_grid()
        return ModalMetricTables(
            kind,
            grid.omega,
            np.array([0.0, LENGTH]),
            np.full((1, 1, 1, n_omega), 0.5) if kind == "diagonal" else None,
            dense,
            np.full((1, n_omega), error),
            np.full((1, n_omega), 0.5),
            np.full((1, n_omega), 1.5),
            "lab",
            "sampled",
        )

    def tables_with(problem, **values):
        tables = dataclasses.replace(problem.transform.modal_tables)
        for name, value in values.items():
            object.__setattr__(tables, name, value)
        return spoil(problem, "transform", "modal_tables", tables)

    def response(grid, name, value):
        record = regional_responses(
            [GENERAL_CHI3], [0], n_to=grid.n_to, dt_fine=grid.dt_fine
        )
        object.__setattr__(record, name, value)
        return record

    def dense(problem, factor, gauge=None):
        metric = problem.linear.modal_metric
        changed = dataclasses.replace(metric, gauge_id=gauge or metric.gauge_id)
        object.__setattr__(changed, "dense", metric.dense * factor)
        return spoil(problem, "linear", "modal_metric", changed)

    return {
        "diagonal metric": ("pair", lambda p, g: spoil(
            p, "linear", "modal_metric", metric_of("diagonal", 0.0)
        )),
        "absent metric": ("pair", lambda p, g: spoil(
            p, "linear", "modal_metric", None
        )),
        "identity that is not": ("pair", lambda p, g: spoil(
            p, "linear", "modal_metric", metric_of("identity", 0.5)
        )),
        "dense of another basis": ("circular", lambda p, g: dense(p, 1.0, "elsewhere")),
        "dense too far": ("circular", lambda p, g: dense(p, 1.5)),
        "no execution map": ("pair", lambda p, g: tables_with(
            p, synthesis=None, projection=None
        )),
        "power weight": ("pair", lambda p, g: spoil(
            p, "linear", "power_weight_constant", np.full(p.shape[1:], 1.5)
        )),
        "envelope grid": ("pair", lambda p, g: dataclasses.replace(
            p, grid_kind="envelope"
        )),
        "polarisation axis": ("pair", lambda p, g: dataclasses.replace(
            p, shape=(1, p.shape[1], 2, p.shape[3])
        )),
        "batch axis": ("pair", lambda p, g: dataclasses.replace(
            p, shape=(2, *p.shape[1:])
        )),
        "varying susceptibility": ("pair", lambda p, g: spoil(
            p, "responses", "regional_responses",
            response(g, "chi3", np.full((1, 2, 1), GENERAL_CHI3)),
        )),
        "one component": ("pair", lambda p, g: spoil(
            p, "responses", "regional_responses",
            response(g, "instantaneous_tensor", np.ones((1, 1, 1, 1, 1))),
        )),
        "unknown model": ("pair", lambda p, g: spoil(
            p, "responses", "regional_responses",
            response(g, "model_id", "isotropic-quartic"),
        )),
        "delay without its tag": ("pair", lambda p, g: spoil(
            p, "responses", "regional_responses",
            response(g, "delayed_tensor", np.zeros((1, 1, 3, 3, 3, 3))),
        )),
    }  # fmt: skip


SPOILERS = spoilers()


@pytest.mark.parametrize("which", sorted(SPOILERS))
def test_numba_cpu_refuses_what_the_reference_refuses_in_the_same_words(cases, which):
    """One admission list, run under each backend's name.

    ``numba_cpu`` calls the reference's own route refusal with its registry name,
    so the two cannot drift apart: each spoiled problem is refused by both, and the
    messages differ in the backend's name alone. Two of them name no backend at
    all -- a metric of another basis, a record whose tags and term axis disagree --
    because they are statements about the record, not about what executes it.
    """
    name, spoiler = SPOILERS[which]
    grid, guide, route, state = cases(name)
    problem = spoiler(problem_for(grid, guide, route, state), grid)
    with pytest.raises(BackendError) as reference:
        NumpyRefBackend(problem, grid=grid)
    with pytest.raises(BackendError) as compiled:
        NumbaCpuBackend(problem, grid=grid)
    assert ("numba_cpu" in str(compiled.value)) == ("numpy_ref" in str(reference.value))
    assert str(compiled.value) == str(reference.value).replace("numpy_ref", "numba_cpu")


def test_a_metric_or_a_response_without_tables_is_refused_by_name(
    soliton_grid, soliton_problem
):
    """A leaf of the general route without the reference it belongs to."""
    problem = soliton_problem(soliton_grid, 1, save_planes=[0.001])
    record = regional_responses(
        [GENERAL_CHI3], [0], n_to=problem.grids.n_to, dt_fine=1e-15
    )
    spoiled = spoil(problem, "responses", "regional_responses", record)
    with pytest.raises(BackendError, match=r"numba_cpu executes responses\.regional"):
        NumbaCpuBackend(spoiled, grid=soliton_grid)


# ------------------------------------------------------------ statistics, batches


def test_the_general_statistics_are_the_references_on_the_same_state(cases):
    """The accessor whitelist and the metric-weighted statistics, both backends.

    The statistics are :func:`solprop.stats.collect`'s on both, so on one state
    they agree to that function's own reductions, including the dense metric of a
    declared basis (``gamma`` of the samples); the legacy names are refused with the
    reference's message.
    """
    spec = StatsSpec(names=("physical_energy", "analytic_flux"))
    for name in ("pair", "circular"):
        grid, guide, route, state = cases(name)
        problem = problem_for(grid, guide, route, state)
        values = []
        for backend_class in (NumpyRefBackend, NumbaCpuBackend):
            backend = backend_class(problem, grid=grid)
            try:
                backend.set_initial(problem.initial.field0, None)
                values.append(backend.stats(spec))
                with pytest.raises(BackendError, match="energy"):
                    backend.stats(StatsSpec(names=("energy",)))
            finally:
                backend.close()
        bound = 2 * higham_gamma(state.size)
        for statistic in spec.names:
            np.testing.assert_allclose(
                values[1][statistic], values[0][statistic], rtol=bound, atol=0
            )


@pytest.mark.medium
def test_a_compiled_general_batch_is_what_its_members_run_alone(cases):
    """The lockstep batch of two launches of one guide, on ``numba_cpu``.

    A member and its solo run take ten fixed steps of one method, so they are held to
    the ten-step parity bound (``general_parity.ten_step_bound``), not their bits.
    """
    grid, guide, route, state = cases("pair")
    launches = (state, 0.5 * launch(grid, 2, seed=7) * np.max(np.abs(state)))
    problems = [problem_for(grid, guide, route, one) for one in launches]
    with ExitStack() as cleanup:
        members, alone = [], []
        for problem in problems:
            for bucket in (members, alone):
                backend = NumbaCpuBackend(problem, grid=grid)
                cleanup.callback(backend.close)
                backend.set_initial(problem.initial.field0, None)
                bucket.append(backend)
        runner = NumbaCpuBatchBackend(members)
        cleanup.callback(runner.close)
        runner.propagate_to(LENGTH)
        for owner in alone:
            owner.propagate_to(LENGTH)
        observed = runner.observe_members()
        maps = _general_maps(problems[0])
        assert maps is not None
        for one, owner, start in zip(observed, alone, launches, strict=True):
            bound = ten_step_bound(
                nonlinear_scale(maps, start, grid),
                source_bound(maps, grid.n_to, conditioning(maps, start, grid)),
            )
            assert departure(one.field, owner.observe().require_field()) <= bound
        assert not np.array_equal(
            observed[0].require_field(), observed[1].require_field()
        )


# ------------------------------------------------------------------------ buffers


def test_every_general_ledger_row_is_zero_on_another_route(
    soliton_grid, soliton_problem
):
    problem = soliton_problem(soliton_grid, 1, save_planes=[0.001])
    backend = NumbaCpuBackend(problem, grid=soliton_grid)
    try:
        ledger = backend.memory_ledger()
        assert backend.metadata()["general_node_chunk"] is None
    finally:
        backend.close()
    for row in GENERAL_LEDGER_ROWS:
        assert ledger[row] == 0, row


def test_the_ledger_counts_the_routes_own_buffers(cases):
    """Every persistent buffer is a named row, and the shared rows are the reference's.

    The maps and the record's resident reference are the same arrays on both
    backends and are counted identically; every other row is this backend's own
    buffer, and the rows add up to what the workspace holds.
    """
    grid, guide, route, state = cases("raman")
    problem = problem_for(grid, guide, route, state)
    reference = NumpyRefBackend(problem, grid=grid)
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        ledger = backend.memory_ledger()
        shared = reference.memory_ledger()
        workspace = backend._general
        assert workspace is not None
        metadata = backend.metadata()
    finally:
        backend.close()
        reference.close()
    for row in (
        "general_synthesis",
        "general_projection",
        "general_tensor",
        "general_electric",
        "general_magnetic",
        "general_coordinate_dual",
        "general_delayed_tensor",
        "general_delayed_spectrum",
    ):
        assert ledger[row] == shared[row] > 0, row
    plans = workspace.plans
    owned = {
        "general_node_spectrum": plans.spectrum,
        "general_node_field": plans.signal,
        "general_node_polarisation": plans.polarisation,
        "general_node_source": plans.polarisation_spectrum,
        "general_node_drive": plans.drive,
        "general_node_drive_spectrum": plans.drive_spectrum,
        "general_node_product": plans.product,
        "general_node_histories": plans.history,
        "general_node_delayed": workspace._total,
        "general_lab": workspace.lab,
        "general_output": workspace.output,
    }
    for row, buffer in owned.items():
        assert ledger[row] == buffer.nbytes > 0, row
    assert set(GENERAL_LEDGER_ROWS) <= set(ledger)
    assert metadata["general_node_chunk"] == workspace.maps.node_chunk
    assert isinstance(metadata["general_fftw_threads"], int)
    assert metadata["general_fftw_threads"] >= 1


def test_a_stage_of_the_source_allocates_no_buffer(cases):
    """Plans and buffers are built once: a stage's source allocates nothing sizeable.

    Measured with ``tracemalloc`` after a warm-up call, against the smallest buffer
    the pass would need if it allocated one -- a single node's three components on
    the fine grid, 12 KiB here -- so what is left is Python's own small objects.
    """
    grid, guide, route, state = cases("raman")
    backend = NumbaCpuBackend(problem_for(grid, guide, route, state), grid=grid)
    try:
        workspace = backend._general
        assert workspace is not None
        workspace.lab[:] = state[:, 0]
        workspace.source()
        tracemalloc.start()
        workspace.source()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        backend.close()
    assert peak < 3 * grid.n_to * 16


def test_a_case_is_built_on_its_own_grid():
    """The helper's cases are what their names say: guard against a silent swap."""
    grid, guide, form = build_case("pair_analytic")
    assert form == "analytic"
    assert len(guide.mode_collection) == 2
    assert grid.n_to == general_grid().n_to


# -------------------------------------------------------------------- end to end


@pytest.mark.medium
def test_a_run_through_the_entry_point_is_the_references_on_numba(cases):
    """``Simulation.from_waveguide(..., backend="numba_cpu")`` end to end.

    The user's door, with a shot-noise draw, the run loop's admission, the output
    and the statistics: the saved field agrees with the reference's run within the
    ten-step bound at the launch's own nonlinear scale over the millimetre, which
    covers the shorter run, and the recorded
    ``physical_energy`` within the same relative bound.
    """
    grid, guide, _, _ = cases("pair")
    # The entry point's default collar is stable in-stage under the default Adams
    # history only below 0.7 um steps, so it is split off and integrated exactly.
    short = LENGTH / 50
    runs = {}
    for backend in ("numpy_ref", "numba_cpu"):
        simulation = Simulation.from_waveguide(
            (),
            short,
            provider=guide,
            grid=grid,
            lambda0=835e-9,
            fwhm=20e-15,
            energy=2e-11,
            pulseshape="gauss",
            noise="input",
            seed=5,
            schedule=FixedSchedule.uniform(0.0, short, short / 10, save_planes=[short]),
            save_planes=[short],
            backend=backend,
        )
        simulation.settings = dataclasses.replace(
            simulation.settings, absorber_scheme="split_step"
        )
        output = simulation.run(progress="none")
        assert isinstance(output, MemoryOutput)
        runs[backend] = output
    problem = Simulation.from_waveguide(
        (),
        short,
        provider=guide,
        grid=grid,
        lambda0=835e-9,
        fwhm=20e-15,
        energy=2e-11,
        pulseshape="gauss",
        save_planes=[short],
        backend="numpy_ref",
        noise=None,
    ).translate()
    maps = _general_maps(problem)
    assert maps is not None
    state = problem.initial.field0[0]
    bound = ten_step_bound(
        nonlinear_scale(maps, state, grid),
        source_bound(maps, grid.n_to, conditioning(maps, state, grid)),
    )
    reference, compiled = (runs[one].planes()[-1] for one in ("numpy_ref", "numba_cpu"))
    assert departure(compiled.field, reference.field) <= bound
    energies = [
        float(np.ravel(plane.stats["physical_energy"])[0])
        for plane in (reference, compiled)
    ]
    assert energies[0] > 0.0
    assert energies[1] == pytest.approx(energies[0], rel=bound, abs=0.0)


def test_the_plan_family_refuses_a_flag_or_a_tail_it_cannot_plan():
    """Both refusals of the fourth family fire before any buffer is planned."""
    with pytest.raises(BackendError, match="is not a planning flag"):
        build_general_plans(4, 0, 16, delayed=False, flag="thorough")
    with pytest.raises(BackendError, match="must be smaller than its chunk of 4"):
        build_general_plans(4, 4, 16, delayed=True, flag="estimate")
