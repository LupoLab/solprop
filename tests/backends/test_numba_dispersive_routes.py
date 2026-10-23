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

**The bounds are derived from summation lengths**, as on the fused path
(``test_numba_dispersive_kerr.py``). One transform of ``n`` points is ``8 eps
ceil(log2 n)``. One evaluation of the constant cubic on a spatial route is two
temporal transforms and two transverse ones (of ``n_columns`` points; the planar
route's y transforms and its slab rule's ``n_x`` sums, the general route's synthesis
and projection over the modes and nodes). The response adds, per product of its
layout, one inverse transform of a filtered field and one forward transform of the
product; the planar route adds a synthesis and a projection for each. The reference
takes as many again at every node, so each count is doubled.
"""

import dataclasses
import math
import warnings
from typing import Any

import numpy as np
import pytest

from backends.matched import fixed_order
from bounds import assert_within, gamma, step_evaluations
from modal_provider_cases import GENERAL_Z_END, WAVELENGTH, general_grid
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numba_cpu.dispersive import DISPERSIVE_LEDGER_ROWS
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref import backend as numpy_ref_backend
from solprop.boundaries import absorber
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import SolpropWarning
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, TimeGrid
from solprop.modes import PlanarGrid, SlabMode, StepIndexMode
from solprop.modes.guide import ModalGuide
from solprop.modes.material import GasKerr, IndexMaterial, MaterialKerr, MaterialRegion
from solprop.nonlinear.raman import response_for
from solprop.nonlinear.responses import DispersiveKerrResponse
from solprop.physdata import MaterialSelection, Mixture
from solprop.physdata.nonlinear_data import OptmatKerr
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, Multistep, PIController, RungeKutta
from solprop.translate.general import general_modal_route
from solprop.translate.problem import KernelProblem, assemble
from solprop.translate.responses import KerrForm

EPS = np.finfo(float).eps
STEPS = 10
FREE_LENGTH = 1e-4
ARGON = OptmatKerr("Ar").dispersive_model()
HELIUM = OptmatKerr("He").dispersive_model()
DENSITY = 4.9e25
"""Two bar of argon at room temperature, m^-3."""

EVALUATORS = ("symmetric", "polynomial")
POLICIES = {"symmetric": "symmetric-kerr", "polynomial": "gas-polynomial"}
FREE_CASES: dict[str, dict[str, Any]] = {
    "field full": {"kind": "field", "thg": True},
    "field conserving": {"kind": "field", "thg": False},
    "envelope conserving": {"kind": "envelope", "thg": False},
    "envelope full": {"kind": "envelope", "thg": True},
}


# --- problems -------------------------------------------------------------------


def free_grid(kind: str, thg: bool) -> TimeGrid:
    """Return a free-space grid; an envelope's full cubic needs harmonic room."""
    if kind == "envelope":
        return EnvelopeGrid.from_limits(
            FREE_LENGTH, 800e-9, (500e-9, 1.6e-6), 100e-15, thg=thg
        )
    return FieldGrid.from_limits(FREE_LENGTH, 800e-9, (400e-9, 2e-6), 100e-15)


def free_problem(
    responses: list[Any],
    grid: TimeGrid,
    *,
    line: bool = False,
    medium: SpatialMedium | None = None,
    adams: bool = True,
) -> KernelProblem:
    """Return a Cartesian (or line) free-space problem of ten fixed steps.

    Vacuum optics and argon's density, or a gas mixture's own medium.
    """
    space = (
        LineGrid(3e-4, 16, window_factor=0)
        if line
        else FreeGrid(3e-4, 9, 4e-4, 12, window_factor=0)
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-6), 8e-5
    )
    return assemble(
        grid,
        None,
        responses,
        pulse.build(grid, space, medium=SpatialMedium()) * grid.active_mask,
        spatial_grid=space,
        medium=SpatialMedium() if medium is None else medium,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[FREE_LENGTH],
        dz_max=FREE_LENGTH,
        dz_init=FREE_LENGTH / STEPS,
        schedule=FixedSchedule.uniform(0.0, FREE_LENGTH, FREE_LENGTH / STEPS),
        integrator=Multistep() if adams else None,
        density=DENSITY if medium is None else None,
    )


HALF_GAP = 63.5e-6
PLANAR_LENGTH = 2e-3


def planar_problem(
    responses: list[Any],
    *,
    integrator: Multistep | RungeKutta | None = None,
    steps: int = STEPS,
    absorbed: bool = False,
) -> tuple[KernelProblem, TimeGrid]:
    """Return a hybrid planar run: TE1 and TE3 of a lossless argon gap, four k_y.

    `absorbed` adds a temporal collar in the in-stage scheme, whose term reads the
    block's bare rows after the material.
    """
    grid = FieldGrid.from_limits(PLANAR_LENGTH, 800e-9, (200e-9, 3e-6), 100e-15)
    modes = [SlabMode(HALF_GAP, "Ar", 2.0, order=p, loss=False) for p in (1, 3)]
    pulse = PlanarPulse(
        GaussianPulse(wavelength=800e-9, fwhm=15e-15, energy=2e-4),
        3e-4,
        waist_x=0.735 * HALF_GAP,
    )
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-12),
        save_planes=[PLANAR_LENGTH],
        dz_max=PLANAR_LENGTH / steps,
        dz_init=PLANAR_LENGTH / steps,
        schedule=FixedSchedule.uniform(0.0, PLANAR_LENGTH, PLANAR_LENGTH / steps),
        backend="numpy_ref",
        stats=StatsSpec(names=("energy",)),
        integrator=Multistep() if integrator is None else integrator,
        absorber=absorber(grid, boundary_length=1.0e3) if absorbed else None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SolpropWarning)
        simulation = Simulation.from_planar(
            grid,
            PlanarGrid(modes, 1e-3, 4),
            responses,
            pulse,
            settings=settings,
            materials=MaterialSelection(),
        )
        return simulation.translate(), simulation.grid


def raman(thg: bool) -> object:
    """Return a delayed response beside the electronic one: the column runs too.

    Silica's shape at a fifth of argon's static coefficient per particle, so that the
    two are of a size and the argon runs' ten steps stay stable.
    """
    return response_for(
        "SiO2", amplitude=0.2 * ARGON.gamma3_static, fraction=0.18, thg=thg
    )


# --- the measurements ------------------------------------------------------------


def source(backend_class: type, problem: KernelProblem, grid: TimeGrid) -> np.ndarray:
    """Return one evaluation of the source at the launch, in the lab frame."""
    backend = backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        return np.array(backend._adams_lab_source(problem.initial.field0, 0.0))
    finally:
        backend.close()


def run(
    backend_class: type, problem: KernelProblem, grid: TimeGrid, **options: object
) -> tuple[np.ndarray, Any]:
    """Propagate to the last save plane and return the field and the summary."""
    backend = backend_class(problem, grid=grid, **options)
    try:
        backend.set_initial(problem.initial.field0, None)
        summary = backend.propagate_to(float(problem.stepper.save_planes[-1]))
        return backend.observe().require_field().copy(), summary
    finally:
        backend.close()


def disagreement(ours: np.ndarray, theirs: np.ndarray) -> float:
    """Return the relative L2 difference of two arrays of the same size."""
    ours, theirs = np.ravel(ours), np.ravel(theirs)
    scale = float(np.linalg.norm(theirs))
    assert scale > 0
    return float(np.linalg.norm(ours - theirs) / scale)


def evaluation_bound(problem: KernelProblem) -> float:
    """Bound one evaluation's rounding on a spatial or general route (the docstring)."""
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None
    groups = len({row.tobytes() for row in np.ascontiguousarray(leaf.filters)})
    products = leaf.n_channels if leaf.evaluator == "symmetric" else 1 + 2 * groups
    temporal = 8 * EPS * math.ceil(math.log2(problem.grids.n_to))
    columns = problem.shape[1] * problem.shape[2]
    transverse = 8 * EPS * math.ceil(math.log2(max(2, columns)))
    sums = gamma(max(2, problem.transform.n_nodes))
    response = products * (2 * temporal + 2 * sums)
    return 2 * (2 * temporal + 2 * transverse + 2 * sums + response)


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


def strand(gas: Mixture | None, *, core: float | None = None) -> ModalGuide:
    """Return a 1.5 um silica-like strand's HE11 guide in a gas exterior.

    The core is a constant solid region (``MaterialKerr``, or linear), the unbounded
    exterior the gas region (``GasKerr``), as ``test_dispersive_kerr_routes.py``
    builds it.
    """
    grid = general_grid()
    mode = StepIndexMode(
        1.5e-6,
        core=1.45,
        cladding=1.0,
        profile_wavelength=WAVELENGTH,
        kind="HE",
        n=1,
        m=1,
        parity="even",
    )
    solid, outside = IndexMaterial(1.45), IndexMaterial(1.0)
    regions = (
        MaterialRegion(
            "core",
            0.0,
            mode.core_radius,
            solid,
            () if core is None else (MaterialKerr(chi3=core),),
        ),
        MaterialRegion(
            "cladding",
            mode.core_radius,
            None,
            outside,
            () if gas is None else (GasKerr(gas),),
        ),
    )
    provider = ModalGuide.from_step_index(
        (mode,), core=solid, cladding=outside, regions=regions
    ).field_provider
    active = grid.omega[grid.active_mask]
    low, high = float(active[0]), float(active[-1])
    return ModalGuide.from_provider(
        provider,
        np.linspace(0.99 * low, 1.01 * high, 9),
        nodes=provider.node_rule(0.99 * low, n_r=16),
    )


ARGON_FILL = Mixture.fill({"Ar": 1.0}, 2.0)


def general_problem(
    guide: ModalGuide, evaluator: str, form: KerrForm, *, adams: bool = True
) -> tuple[KernelProblem, TimeGrid]:
    """Return a general modal problem launching 2 nJ into HE11, ten fixed steps."""
    grid = general_grid()
    route = general_modal_route(
        guide,
        grid,
        z_max=GENERAL_Z_END,
        omega_reference=grid.omega0,
        kerr_form=form,
        selection=MaterialSelection(electronic=POLICIES[evaluator]),
    )
    state = np.zeros((1, 1, 1, grid.n_omega), complex)
    state[0, 0, 0] = GaussianPulse(wavelength=835e-9, fwhm=40e-15, energy=2e-9).build(
        grid
    )
    problem = assemble(
        grid,
        list(guide.mode_collection),
        [],
        state,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[GENERAL_Z_END],
        dz_max=GENERAL_Z_END,
        dz_init=GENERAL_Z_END / STEPS,
        schedule=FixedSchedule.uniform(0.0, GENERAL_Z_END, GENERAL_Z_END / STEPS),
        integrator=Multistep() if adams else None,
        general=route,
    )
    return problem, grid


CORE_CHI3 = 2.2e-22
"""The solid core's susceptibility, m^2/V^2, silica-like."""

STRANDS = {
    "gas alone": (ARGON_FILL, None),
    "beside a solid core": (ARGON_FILL, CORE_CHI3),
    "helium and argon": (Mixture.fill({"He": 0.5, "Ar": 0.5}, 2.0), None),
}


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
def test_one_thread_and_four_agree(route, evaluator):
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
