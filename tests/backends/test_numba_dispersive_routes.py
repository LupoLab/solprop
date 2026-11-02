# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 6's dispersive electronic response on ``numba_cpu``'s spatial and general routes.

The Cartesian free-space and line routes, whose nodes are their columns, and the hybrid
planar route, whose ``(p, y)`` rows go to time a block at a time
(``backends/numba_cpu/dispersive_spatial.py``); and the general modal route, whose
frequency-dependent synthesis takes the input multipliers (``backends/numba_cpu/
general_modal.py``). ``numpy_ref``, which applies the multipliers at every node, is the
oracle: one evaluation of the source, one Runge-Kutta step and ten steps of the
default integrator, for both evaluators and the sectors each route admits; a lone
response, one beside Raman, and a mixture; agreement across thread counts and repeated
runs, at the window's rounding budget, and the memory ledger.

**The bounds are derived from summation lengths**, as on the fused path; the derivation
is in ``dispersive_route_cases.py``, with the problems they hold.
"""

import dataclasses

import numpy as np
import pytest

from backends.dispersive_route_cases import (
    ARGON,
    ARGON_FILL,
    CORE_CHI3,
    EVALUATORS,
    FREE_CASES,
    FREE_LENGTH,
    HELIUM,
    STEPS,
    STRANDS,
    disagreement,
    evaluation_bound,
    free_grid,
    free_problem,
    general_problem,
    planar_problem,
    raman,
    run,
    source,
    strand,
)
from backends.matched import fixed_order
from bounds import assert_within, step_evaluations
from modal_provider_cases import GENERAL_Z_END, general_grid
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numba_cpu.dispersive import DISPERSIVE_LEDGER_ROWS
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref import backend as numpy_ref_backend
from solprop.grid import TimeGrid
from solprop.nonlinear.responses import DispersiveKerrResponse
from solprop.physdata import Mixture
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, RungeKutta
from solprop.translate.problem import KernelProblem

# --- the measurements ------------------------------------------------------------


def assert_one_evaluation(problem: KernelProblem, grid: TimeGrid) -> float:
    """One evaluation against the reference; return the measured disagreement."""
    ours = source(NumbaCpuBackend, problem, grid)
    theirs = source(NumpyRefBackend, problem, grid)
    bound = evaluation_bound(problem)
    assert bound > 0
    measured = disagreement(ours, theirs)
    assert measured < bound
    return measured


def assert_steps(problem: KernelProblem, grid: TimeGrid) -> float:
    """A replayed window of fixed steps against the reference; return the measure."""
    ours, summary = run(NumbaCpuBackend, problem, grid)
    theirs, reference = run(NumpyRefBackend, problem, grid)
    assert summary.n_steps == reference.n_steps
    bound = summary.n_steps * step_evaluations(problem) * evaluation_bound(problem)
    assert bound > 0
    measured = disagreement(ours, theirs)
    assert measured < bound
    return measured


# --- free space and the line ----------------------------------------------------------


@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("case", list(FREE_CASES))
@pytest.mark.parametrize("line", [False, True])
def test_one_free_space_evaluation_matches_the_reference(line, case, evaluator):
    """The response alone, its products on the columns of each chunk."""
    grid = free_grid(**FREE_CASES[case])
    response = DispersiveKerrResponse(ARGON, evaluator, thg=FREE_CASES[case]["thg"])
    assert_one_evaluation(free_problem([response], grid, line=line), grid)


@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("case", ["field full", "field conserving"])
def test_one_free_space_evaluation_beside_raman_matches_the_reference(case, evaluator):
    """The column runs beside the response, which adds to the chunk's spectrum."""
    thg = FREE_CASES[case]["thg"]
    grid = free_grid(**FREE_CASES[case])
    response = DispersiveKerrResponse(ARGON, evaluator, thg=thg)
    problem = free_problem([response, raman(thg)], grid)
    assert problem.flags.has_raman
    assert_one_evaluation(problem, grid)


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_one_free_space_runge_kutta_step_matches_the_reference(evaluator):
    """Seven evaluations through the whole Cartesian stage."""
    grid = free_grid("field", True)
    problem = free_problem(
        [DispersiveKerrResponse(ARGON, evaluator)], grid, adams=False
    )
    problem = dataclasses.replace(
        problem,
        stepper=dataclasses.replace(
            problem.stepper,
            schedule=FixedSchedule.uniform(
                0.0, FREE_LENGTH / STEPS, FREE_LENGTH / STEPS
            ),
            save_planes=(FREE_LENGTH / STEPS,),
        ),
    )
    ours, summary = run(NumbaCpuBackend, problem, grid)
    theirs, _ = run(NumpyRefBackend, problem, grid)
    assert summary.n_steps == 1
    bound = step_evaluations(problem) * evaluation_bound(problem)
    assert bound > 0
    assert disagreement(ours, theirs) < bound


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("case", list(FREE_CASES))
@pytest.mark.parametrize("line", [False, True])
def test_ten_free_space_steps_match_the_reference(line, case, evaluator):
    """The default integrator over ten replayed steps."""
    grid = free_grid(**FREE_CASES[case])
    response = DispersiveKerrResponse(ARGON, evaluator, thg=FREE_CASES[case]["thg"])
    assert_steps(free_problem([response], grid, line=line), grid)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("thg", [True, False])
def test_ten_free_space_steps_beside_raman_match_the_reference(thg, evaluator):
    grid = free_grid("field", thg)
    response = DispersiveKerrResponse(ARGON, evaluator, thg=thg)
    assert_steps(free_problem([response, raman(thg)], grid), grid)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_a_free_space_mixture_matches_the_reference(evaluator):
    """Helium and argon in the medium: two channels, each with its own leg bound.

    The symmetric evaluator's one product under the mixture filter, the polynomial's
    grouped per-species sum.
    """
    grid = free_grid("field", True)
    medium = SpatialMedium(Mixture.fill({"He": 0.6, "Ar": 0.4}, 2.0))
    responses = [
        DispersiveKerrResponse(HELIUM, evaluator, species="He"),
        DispersiveKerrResponse(ARGON, evaluator, species="Ar"),
    ]
    problem = free_problem(responses, grid, medium=medium)
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None and leaf.n_channels == 2
    assert_one_evaluation(problem, grid)
    assert_steps(problem, grid)


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_free_space_chunks_accumulate_as_one(evaluator):
    """Chunks of 20 columns, the last a partial one: each lands where it belongs."""
    grid = free_grid("field", True)
    problem = free_problem([DispersiveKerrResponse(ARGON, evaluator)], grid)
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=20)
    )
    assert problem.transform.n_nodes % 20
    assert_one_evaluation(problem, grid)


# --- the planar route -------------------------------------------------------------


@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("thg", [True, False])
def test_one_planar_evaluation_matches_the_reference(thg, evaluator):
    """The block's rows filtered, synthesised in x, multiplied and projected back."""
    problem, grid = planar_problem([DispersiveKerrResponse(ARGON, evaluator, thg=thg)])
    assert_one_evaluation(problem, grid)


@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("thg", [True, False])
def test_one_planar_evaluation_beside_raman_matches_the_reference(thg, evaluator):
    problem, grid = planar_problem(
        [DispersiveKerrResponse(ARGON, evaluator, thg=thg), raman(thg)]
    )
    assert_one_evaluation(problem, grid)


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_a_planar_collar_reads_the_bare_rows(evaluator):
    """The in-stage collar reads the block's rows after the material, unfiltered.

    A lone symmetric response then filters into a buffer of its own rather than in
    place, and synthesises its filtered field over the bare one at the nodes.
    """
    problem, grid = planar_problem(
        [DispersiveKerrResponse(ARGON, evaluator)], absorbed=True
    )
    assert problem.responses.absorber_scheme == "nonlinear"
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        workspace = backend._spatial_dispersive
        assert workspace is not None and backend._in_stage_absorber
        assert not workspace.in_place
    finally:
        backend.close()
    assert_one_evaluation(problem, grid)
    assert_steps(problem, grid)


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_planar_blocks_accumulate_as_one(evaluator):
    """Blocks of three y rows of four, the last a partial one."""
    problem, grid = planar_problem([DispersiveKerrResponse(ARGON, evaluator)])
    geometry = problem.transform.geometry
    assert geometry is not None
    n_x = geometry.shape[0]
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=3 * n_x)
    )
    assert_one_evaluation(problem, grid)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("thg", [True, False])
def test_ten_planar_steps_match_the_reference(thg, evaluator):
    problem, grid = planar_problem([DispersiveKerrResponse(ARGON, evaluator, thg=thg)])
    assert_steps(problem, grid)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_one_planar_runge_kutta_step_matches_the_reference(evaluator):
    problem, grid = planar_problem(
        [DispersiveKerrResponse(ARGON, evaluator)], integrator=RungeKutta(), steps=1
    )
    ours, summary = run(NumbaCpuBackend, problem, grid)
    theirs, _ = run(NumpyRefBackend, problem, grid)
    assert summary.n_steps == 1
    bound = step_evaluations(problem) * evaluation_bound(problem)
    assert bound > 0
    assert disagreement(ours, theirs) < bound


# --- the general modal route ---------------------------------------------------------


@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("form", ["real", "analytic"])
@pytest.mark.parametrize("strand_case", list(STRANDS))
def test_one_general_evaluation_matches_the_reference(strand_case, form, evaluator):
    """The multipliers in synthesis, the products at the gas region's nodes."""
    gas, core = STRANDS[strand_case]
    problem, grid = general_problem(strand(gas, core=core), evaluator, form)
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None and leaf.regions is not None
    assert_one_evaluation(problem, grid)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("form", ["real", "analytic"])
@pytest.mark.parametrize("strand_case", list(STRANDS))
def test_ten_general_steps_match_the_reference(strand_case, form, evaluator):
    gas, core = STRANDS[strand_case]
    problem, grid = general_problem(strand(gas, core=core), evaluator, form)
    assert_steps(problem, grid)


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_one_general_runge_kutta_step_matches_the_reference(evaluator):
    problem, grid = general_problem(
        strand(ARGON_FILL, core=CORE_CHI3), evaluator, "real", adams=False
    )
    problem = dataclasses.replace(
        problem,
        stepper=dataclasses.replace(
            problem.stepper,
            schedule=FixedSchedule.uniform(
                0.0, GENERAL_Z_END / STEPS, GENERAL_Z_END / STEPS
            ),
            save_planes=(GENERAL_Z_END / STEPS,),
        ),
    )
    ours, summary = run(NumbaCpuBackend, problem, grid)
    theirs, _ = run(NumpyRefBackend, problem, grid)
    assert summary.n_steps == 1
    bound = step_evaluations(problem) * evaluation_bound(problem)
    assert bound > 0
    assert disagreement(ours, theirs) < bound


def test_a_general_polynomial_region_groups_its_species_and_forms_one_cube():
    """Krypton and xenon in one region share a leg bound: one group, one cube."""
    problem, _ = general_problem(
        strand(Mixture.fill({"Kr": 0.5, "Xe": 0.5}, 1.0)), "polynomial", "real"
    )
    backend = NumbaCpuBackend(problem, grid=general_grid())
    try:
        general = backend._general
        assert general is not None and general.dispersive is not None
        terms = general.dispersive.terms
        assert terms.groups == ((0, 1),)
        assert terms.n_cubes == 1 and terms.n_products == 3
    finally:
        backend.close()


# --- what is this backend's alone ---------------------------------------------------


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("route", ["free", "planar", "general"])
def test_one_thread_and_four_agree(thread_budget_lifted, route, evaluator):
    """The parallel node loops, forced, against the serial ones, and two runs alike.

    The same fixed-order method on the same fixed schedule, so the runs differ by the
    order of their threaded reductions alone: the window's rounding budget, the
    reference comparison's (:func:`assert_steps`).
    """
    response = DispersiveKerrResponse(ARGON, evaluator)
    if route == "free":
        grid = free_grid("field", True)
        problem = free_problem([response], grid)
    elif route == "planar":
        problem, grid = planar_problem([response])
    else:
        problem, grid = general_problem(
            strand(ARGON_FILL, core=CORE_CHI3), evaluator, "analytic"
        )
    problem = fixed_order(problem)
    serial, one = run(NumbaCpuBackend, problem, grid, threads=1, regime="large")
    parallel, many = run(NumbaCpuBackend, problem, grid, threads=4, regime="large")
    again, _ = run(NumbaCpuBackend, problem, grid, threads=4, regime="large")
    assert one.n_steps == many.n_steps
    bound = one.n_steps * step_evaluations(problem) * evaluation_bound(problem)
    assert_within(parallel, serial, bound, what="four threads")
    assert_within(again, parallel, bound, what="repeat run")


def ledger_of(problem: KernelProblem, grid: TimeGrid) -> dict[str, int]:
    """Return the backend's ledger and check it is released on close."""
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    closed = backend.memory_ledger()
    assert all(closed[row] == 0 for row in DISPERSIVE_LEDGER_ROWS)
    return ledger


def test_a_lone_symmetric_free_response_holds_no_field():
    """Filtered in place, its product is the chunk's own polarisation."""
    grid = free_grid("field", True)
    problem = free_problem([DispersiveKerrResponse(ARGON, "symmetric")], grid)
    ledger = ledger_of(problem, grid)
    for row in (
        "dispersive_filtered_time",
        "dispersive_node_fields",
        "dispersive_products",
        "dispersive_product_spectra",
    ):
        assert ledger[row] == 0, row


def test_a_lone_analytic_symmetric_free_response_filters_in_place():
    """Complex products beside a real polarisation: its own product, no filtered field.

    The chunk's spectrum is filtered in place, so the stage's inverse transform gives
    the filtered analytic signal; the product and its spectrum are the response's.
    """
    grid = free_grid("field", False)
    problem = free_problem(
        [DispersiveKerrResponse(ARGON, "symmetric", thg=False)], grid
    )
    ledger = ledger_of(problem, grid)
    rows = min(problem.transform.node_chunk, problem.transform.n_nodes)
    assert ledger["dispersive_filtered_time"] == 0
    assert ledger["dispersive_products"] == rows * grid.n_to * 16
    assert ledger["dispersive_product_spectra"] == rows * grid.n_to * 16


@pytest.mark.parametrize(
    ("evaluator", "products"), [("symmetric", 1), ("polynomial", 3)]
)
def test_a_free_response_beside_raman_holds_its_products_and_spectra(
    evaluator, products
):
    """Each product of a chunk in time and frequency; the filtered fields in place."""
    grid = free_grid("field", False)
    problem = free_problem(
        [DispersiveKerrResponse(ARGON, evaluator, thg=False), raman(False)], grid
    )
    ledger = ledger_of(problem, grid)
    rows = min(problem.transform.node_chunk, problem.transform.n_nodes)
    field = rows * grid.n_to * 16  # the analytic family: complex products
    assert ledger["dispersive_products"] == products * field
    assert ledger["dispersive_product_spectra"] == products * field
    assert ledger["dispersive_filtered_time"] == 0
    assert ledger["dispersive_node_fields"] == 0


def test_the_general_route_holds_its_filtered_fields_and_products():
    """Per chunk: the filtered fields, the products and their spectra."""
    problem, grid = general_problem(strand(ARGON_FILL), "polynomial", "real")
    ledger = ledger_of(problem, grid)
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        general = backend._general
        assert general is not None
        rows, n_to = general.plans.spectrum.shape
    finally:
        backend.close()
    assert ledger["dispersive_filtered_time"] == 2 * rows * n_to * 16
    assert ledger["dispersive_products"] == 3 * rows * n_to * 8
    assert ledger["dispersive_product_spectra"] == (
        2 * rows * n_to * 16 + 3 * rows * (n_to // 2 + 1) * 16
    )


def test_a_planar_polynomial_holds_two_node_slots():
    problem, grid = planar_problem([DispersiveKerrResponse(ARGON, "polynomial")])
    ledger = ledger_of(problem, grid)
    assert ledger["dispersive_node_fields"] == 2 * ledger["node_field"]
    assert ledger["dispersive_filtered_time"] > 0
    assert ledger["dispersive_products"] > 0


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_a_general_chunk_without_a_gas_node_adds_nothing(monkeypatch, evaluator):
    """A chunk of the solid core's nodes alone: no product is formed or landed.

    The node chunk is cut at the first node of the gas region, so the first chunk
    holds core nodes only; with no regional tensor either, its polarisation spectrum
    is zero, on both backends, and the evaluation still agrees with the reference.
    """
    problem, grid = general_problem(strand(ARGON_FILL), evaluator, "real")
    tables = problem.transform.modal_tables
    assert tables is not None
    first_gas = int(np.flatnonzero(np.asarray(tables.region_index) == 1)[0])
    assert first_gas >= 1
    monkeypatch.setattr(
        numpy_ref_backend, "general_node_chunk", lambda *_, **__: first_gas
    )
    assert_one_evaluation(problem, grid)
