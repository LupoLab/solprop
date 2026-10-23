# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 6's dispersive electronic response on native CPU's spatial and general routes.

solprop-native runs ``responses.dispersive_kerr`` on the Cartesian free-space and line
routes and the hybrid planar route chunk by chunk (its ``SeparableDispersive``), and on
the general modal route in the route's synthesis (``GeneralDispersive``), with the
structure ``numba_cpu`` takes (``backends/numba_cpu/dispersive_spatial.py`` and
``dispersive_general.py``). ``numpy_ref``, which applies the multipliers at every node,
is the oracle and ``numba_cpu`` is checked beside it: one evaluation of the stage
source, one Runge-Kutta step and ten steps of the default integrator, for both
evaluators and the sectors each route admits, field and envelope grids where the route
has both; a lone response and one beside Raman; a mixture; a general modal guide with
a gas region beside a constant solid one; partial chunks; a coupled wall flow on the
fused path, which slice 3a's native code still refused. Then what is the backend's
own: repeatability, the thread count, the path the response takes and the memory
ledger, which ``solprop.backends.memory.dispersive_memory`` predicts to the byte.

**The bounds are ``numba_cpu``'s** (``test_numba_dispersive_routes.py``'s
``evaluation_bound``), derived from the transforms and node reductions both backends
perform and counted for the reference's node-by-node arithmetic, before any run: the
native response takes the same transforms per product (fewer when a lone symmetric
response is the chunk's own polarisation) and the same sums over the slab rule, the
modes and the nodes. Against ``numba_cpu`` the same bound holds, both compiled
backends sitting inside it from the reference's side. Each test states what it
measured.
"""

import dataclasses
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import flow_parity
from backends.matched import fixed_order, native_thread_fields
from backends.test_numba_dispersive_routes import (
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
    free_grid,
    free_problem,
    general_problem,
    planar_problem,
    raman,
    run,
    source,
    strand,
)
from backends.test_numba_dispersive_routes import (
    evaluation_bound as cpu_evaluation_bound,
)
from bounds import assert_within, dispersive_device_bound, step_evaluations
from modal_provider_cases import GENERAL_Z_END
from solprop.backends.memory import dispersive_memory
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import prop_capillary
from solprop.modes import CoherentWall
from solprop.nonlinear.responses import DispersiveKerrResponse
from solprop.output.memory import MemoryOutput
from solprop.physdata import MaterialSelection, Mixture
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule
from solprop.translate.problem import KernelProblem

solprop_native = pytest.importorskip("solprop_native")
NativeBackend = solprop_native.NativeBackend

ON_DEVICE = solprop_native.build_info().device_backend != "none"
host_only = pytest.mark.skipif(
    ON_DEVICE,
    reason="the thread count is a host notion; a device build is held run to run",
)
pytestmark = [
    # The fine grids reach beyond argon's linear fit; the notice is not the subject.
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
]

PROBE = Path(__file__).parent / "native_dispersive_thread_probe.py"


def evaluation_bound(problem: KernelProblem) -> float:
    """``numba_cpu``'s bound, or on a device build the predeclared device envelope."""
    return dispersive_device_bound(cpu_evaluation_bound(problem), problem)


DISPERSIVE_ROWS = (
    "dispersive_filtered_time",
    "dispersive_node_fields",
    "dispersive_products",
    "dispersive_coarse_products",
    "dispersive_product_spectra",
    "dispersive_projection_chunks",
    "dispersive_multipliers",
)
"""The response's rows of the native memory ledger, named on every problem."""


# --- the measurements ------------------------------------------------------------


def assert_one_evaluation(problem: KernelProblem, grid: Any) -> float:
    """One evaluation against the reference and ``numba_cpu``; return the larger."""
    ours = source(NativeBackend, problem, grid)
    bound = evaluation_bound(problem)
    assert bound > 0
    measured = 0.0
    for backend_class in (NumpyRefBackend, NumbaCpuBackend):
        departure = disagreement(ours, source(backend_class, problem, grid))
        assert departure < bound
        measured = max(measured, departure)
    return measured


def assert_steps(problem: KernelProblem, grid: Any) -> float:
    """A replayed window of fixed steps against both; return the larger departure."""
    ours, summary = run(NativeBackend, problem, grid)
    bound = summary.n_steps * step_evaluations(problem) * evaluation_bound(problem)
    assert bound > 0
    measured = 0.0
    for backend_class in (NumpyRefBackend, NumbaCpuBackend):
        theirs, reference = run(backend_class, problem, grid)
        assert summary.n_steps == reference.n_steps
        departure = disagreement(ours, theirs)
        assert departure < bound
        measured = max(measured, departure)
    return measured


def path_of(problem: KernelProblem, grid: Any) -> dict[str, bool] | None:
    """Return how the native response runs: alone, in place, borrowed."""
    backend = NativeBackend(problem, grid=grid)
    try:
        return backend.metadata()["dispersive_path"]
    finally:
        backend.close()


def one_step(problem: KernelProblem, length: float) -> KernelProblem:
    """Return the problem cut to one Runge-Kutta step of a tenth of `length`."""
    return dataclasses.replace(
        problem,
        stepper=dataclasses.replace(
            problem.stepper,
            schedule=FixedSchedule.uniform(0.0, length / STEPS, length / STEPS),
            save_planes=(length / STEPS,),
        ),
    )


def assert_one_step(problem: KernelProblem, grid: Any) -> None:
    """Seven evaluations through the whole stage, against both backends."""
    ours, summary = run(NativeBackend, problem, grid)
    assert summary.n_steps == 1
    bound = step_evaluations(problem) * evaluation_bound(problem)
    assert bound > 0
    for backend_class in (NumpyRefBackend, NumbaCpuBackend):
        theirs, _ = run(backend_class, problem, grid)
        assert disagreement(ours, theirs) < bound


def route_case(route: str, evaluator: str) -> tuple[KernelProblem, Any]:
    """Return the ten-step case of a route the repeatability tests run."""
    response = DispersiveKerrResponse(ARGON, evaluator)
    if route == "free":
        grid = free_grid("field", True)
        return free_problem([response], grid), grid
    if route == "planar":
        return planar_problem([response])
    return general_problem(strand(ARGON_FILL, core=CORE_CHI3), evaluator, "analytic")


# --- free space and the line -------------------------------------------------------


@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("case", list(FREE_CASES))
@pytest.mark.parametrize("line", [False, True])
def test_one_free_space_evaluation_matches_the_reference(line, case, evaluator):
    """The response alone on each chunk's columns, which are its nodes.

    Measured 2.2-4.6 eps against the reference and 0.5-2.0 eps against
    ``numba_cpu``, under bounds of 640-3100 eps.
    """
    grid = free_grid(**FREE_CASES[case])
    response = DispersiveKerrResponse(ARGON, evaluator, thg=FREE_CASES[case]["thg"])
    problem = free_problem([response], grid, line=line)
    lone = evaluator == "symmetric"
    analytic = case == "field conserving"
    assert path_of(problem, grid) == {
        "alone": True,
        "in_place": lone,
        "borrowed": lone and not analytic,
    }
    assert_one_evaluation(problem, grid)


@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("case", ["field full", "field conserving"])
def test_one_free_space_evaluation_beside_raman_matches_the_reference(case, evaluator):
    """The column runs beside the response, whose spectra are added to the chunk's.

    Measured 3.0-4.4 eps against the reference and 1.4-1.6 eps against ``numba_cpu``.
    """
    thg = FREE_CASES[case]["thg"]
    grid = free_grid(**FREE_CASES[case])
    response = DispersiveKerrResponse(ARGON, evaluator, thg=thg)
    problem = free_problem([response, raman(thg)], grid)
    assert problem.flags.has_raman
    assert path_of(problem, grid) == {
        "alone": False,
        "in_place": False,
        "borrowed": False,
    }
    assert_one_evaluation(problem, grid)


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_one_free_space_runge_kutta_step_matches_the_reference(evaluator):
    """Seven evaluations through the whole Cartesian stage."""
    grid = free_grid("field", True)
    problem = free_problem(
        [DispersiveKerrResponse(ARGON, evaluator)], grid, adams=False
    )
    assert_one_step(one_step(problem, FREE_LENGTH), grid)


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

    The symmetric evaluator's one product under the mixture filter, rebuilt natively
    from the channels' ``g^4``; the polynomial's two groups.
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


# --- the planar route ----------------------------------------------------------------


@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("thg", [True, False])
def test_one_planar_evaluation_matches_the_reference(thg, evaluator):
    """The block's rows filtered, synthesised in x, multiplied and projected back.

    Measured 3.4-5.8 eps against the reference and 1.1-3.0 eps against
    ``numba_cpu``, under 1800-3500 eps.
    """
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
def test_a_planar_collar_leaves_the_lone_response_in_place(evaluator):
    """The in-stage collar is a pass of its own here, which packs the rows afresh.

    So a lone symmetric response filters in place beside it, where ``numba_cpu``'s
    one pass reads the bare rows after the material and filters into a buffer of its
    own; the two agree all the same.
    """
    problem, grid = planar_problem(
        [DispersiveKerrResponse(ARGON, evaluator)], absorbed=True
    )
    assert problem.responses.absorber_scheme == "nonlinear"
    path = path_of(problem, grid)
    assert path is not None
    assert path["in_place"] == (evaluator == "symmetric")
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
    from solprop.stepper import RungeKutta  # noqa: PLC0415

    problem, grid = planar_problem(
        [DispersiveKerrResponse(ARGON, evaluator)], integrator=RungeKutta(), steps=1
    )
    ours, summary = run(NativeBackend, problem, grid)
    assert summary.n_steps == 1
    bound = step_evaluations(problem) * evaluation_bound(problem)
    assert bound > 0
    for backend_class in (NumpyRefBackend, NumbaCpuBackend):
        theirs, _ = run(backend_class, problem, grid)
        assert disagreement(ours, theirs) < bound


# --- the general modal route ---------------------------------------------------------


@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("form", ["real", "analytic"])
@pytest.mark.parametrize("strand_case", list(STRANDS))
def test_one_general_evaluation_matches_the_reference(strand_case, form, evaluator):
    """The multipliers in synthesis, the products at the gas region's nodes.

    Alone (a gas strand with a linear core) the response writes each chunk's source;
    beside the constant solid core it adds to the regional cubic's. Measured 0.5-1.6
    eps against the reference and 0.3-0.8 eps against ``numba_cpu``, under 560-1090
    eps.
    """
    gas, core = STRANDS[strand_case]
    problem, grid = general_problem(strand(gas, core=core), evaluator, form)
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None and leaf.regions is not None
    assert path_of(problem, grid) == {
        "alone": core is None,
        "in_place": False,
        "borrowed": False,
    }
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
    assert_one_step(one_step(problem, GENERAL_Z_END), grid)


def test_a_general_polynomial_region_groups_its_species_and_forms_one_cube():
    """Krypton and xenon in one region share a leg bound: one group, one cube."""
    problem, grid = general_problem(
        strand(Mixture.fill({"Kr": 0.5, "Xe": 0.5}, 1.0)), "polynomial", "real"
    )
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None and leaf.groups() == ((0, 1),)
    ledger = ledger_of(problem, grid)
    # Two complex filtered fields and three real products over the chunk's rows.
    assert 4 * ledger["dispersive_products"] == 3 * ledger["dispersive_filtered_time"]
    assert_one_evaluation(problem, grid)


# --- a coupled flow on the fused path ------------------------------------------------


@pytest.mark.medium
@pytest.mark.parametrize("policy", ["symmetric-kerr", "gas-polynomial"])
def test_a_coherent_wall_flow_beside_the_response_matches_the_reference(policy):
    """The flow carries each stage's input; the response is the fused path's.

    Slice 3a's native code refused the response beside a coupled flow, which
    ``numpy_ref`` and ``numba_cpu`` admit; the stage source is the fused path's own,
    so the flow's ten-step bound holds. Measured 1.0e-15-1.1e-15 against a bound of
    2.2e-13, where the response moves the field from the carrier policy's by 2.2e-7.
    """
    fields = {}
    for name, electronic in (("native", policy), ("numpy_ref", policy)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = cast(
                "MemoryOutput",
                prop_capillary(
                    50e-6,
                    flow_parity.LENGTH,
                    "He",
                    1.0,
                    wall=CoherentWall(),
                    lambda0=800e-9,
                    grid=flow_parity.GRID,
                    absorber=flow_parity.WEAK_COLLAR,
                    fwhm=30e-15,
                    energy=1e-4,
                    modes=3,
                    n_save=2,
                    backend=name,
                    dz_max=flow_parity.LENGTH / flow_parity.STEPS,
                    dz_init=flow_parity.LENGTH / flow_parity.STEPS,
                    schedule=FixedSchedule(
                        np.full(
                            flow_parity.STEPS, flow_parity.LENGTH / flow_parity.STEPS
                        )
                    ),
                    plasma=False,
                    raman=False,
                    materials=MaterialSelection(electronic=electronic),
                ),
            )
        fields[name] = output.planes()[-1].field
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = flow_parity.build("scalar")
        problem = simulation.translate()
    bound = flow_parity.ten_step_bound(
        flow_parity.flow_scales(problem, simulation.grid, fields["numpy_ref"])
    )
    assert bound > 0
    assert flow_parity.departure(fields["native"], fields["numpy_ref"]) <= bound


# --- what is this backend's alone ----------------------------------------------------


def window_bound(problem: KernelProblem) -> float:
    """Bound two runs of a route's ten-step case: every evaluation's rounding, added."""
    return STEPS * step_evaluations(problem) * evaluation_bound(problem)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("route", ["free", "planar", "general"])
def test_two_runs_of_one_build_agree(route, evaluator):
    """Two runs of one build: the same fixed-order method on the same schedule.

    The same discrete computation, held to the window's rounding budget
    (:func:`window_bound`) rather than its bits: a threaded reduction may order its
    sums differently from one run to the next.
    """
    problem, grid = route_case(route, evaluator)
    problem = fixed_order(problem)
    first, _ = run(NativeBackend, problem, grid)
    second, _ = run(NativeBackend, problem, grid)
    assert_within(second, first, window_bound(problem), what="repeat run")


@pytest.mark.medium
@host_only
def test_one_thread_and_four_agree(tmp_path):
    """Ten steps at one thread and at four, each in a fresh process, on every route.

    The same fixed-order method on the same schedule, so the two differ by the order
    of their threaded transforms and reductions alone: the window's rounding budget
    (:func:`window_bound`), as against the reference.
    """
    names = [
        f"{route}:{evaluator}"
        for route in ("free", "planar", "general")
        for evaluator in EVALUATORS
    ]
    one = native_thread_fields(PROBE, 1, names, tmp_path)
    four = native_thread_fields(PROBE, 4, names, tmp_path)
    for name in names:
        problem, _ = route_case(*name.split(":"))
        bound = window_bound(fixed_order(problem))
        assert_within(four[name], one[name], bound, what=name)


# --- the memory ledger -------------------------------------------------------------


def ledger_of(problem: KernelProblem, grid: Any) -> dict[str, int]:
    """Return the backend's ledger, every row of the response named."""
    backend = NativeBackend(problem, grid=grid)
    try:
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    assert all(row in ledger for row in DISPERSIVE_ROWS)
    return ledger


def ledger_cases() -> dict[str, tuple[KernelProblem, Any]]:
    """Return one problem of each route and response layout the ledger is held to."""
    cases: dict[str, tuple[KernelProblem, Any]] = {}
    for evaluator in EVALUATORS:
        for thg in (True, False):
            grid = free_grid("field", thg)
            response = DispersiveKerrResponse(ARGON, evaluator, thg=thg)
            cases[f"free {evaluator} thg={thg}"] = (
                free_problem([response], grid),
                grid,
            )
            cases[f"free {evaluator} thg={thg} beside Raman"] = (
                free_problem([response, raman(thg)], grid),
                grid,
            )
            cases[f"planar {evaluator} thg={thg}"] = planar_problem([response])
            cases[f"planar {evaluator} thg={thg} beside Raman"] = planar_problem(
                [response, raman(thg)]
            )
        envelope = free_grid("envelope", False)
        cases[f"free envelope {evaluator}"] = (
            free_problem(
                [DispersiveKerrResponse(ARGON, evaluator, thg=False)], envelope
            ),
            envelope,
        )
        for form in ("real", "analytic"):
            for name, (gas, core) in STRANDS.items():
                cases[f"general {evaluator} {form} {name}"] = general_problem(
                    strand(gas, core=core), evaluator, form
                )
    return cases


LEDGER_CASES = ledger_cases()


@pytest.mark.parametrize("case", sorted(LEDGER_CASES))
def test_the_estimate_is_the_ledger_to_the_byte(case):
    """``dispersive_memory`` predicts every row the native response holds.

    The rows of :mod:`solprop.backends.memory`'s ``base`` group, one per buffer; the
    multipliers, the per-stage scratch and the plans are named among what it does not
    count.
    """
    problem, grid = LEDGER_CASES[case]
    rows = dispersive_memory(problem, "native")
    ledger = ledger_of(problem, grid)
    assert rows
    for row, size in rows.items():
        assert ledger[row] == size, row
    for row in set(DISPERSIVE_ROWS) - set(rows) - {"dispersive_multipliers"}:
        assert ledger[row] == 0, row


def test_a_lone_symmetric_free_response_holds_no_field():
    """Filtered in place, its product is the chunk's own polarisation."""
    grid = free_grid("field", True)
    problem = free_problem([DispersiveKerrResponse(ARGON, "symmetric")], grid)
    ledger = ledger_of(problem, grid)
    for row in DISPERSIVE_ROWS[:-1]:
        assert ledger[row] == 0, row
    assert ledger["dispersive_multipliers"] > 0


def test_a_free_polynomial_writes_its_filtered_fields_into_its_product_blocks():
    """Three products of a chunk in time and frequency; no filtered field besides."""
    grid = free_grid("field", False)
    problem = free_problem(
        [DispersiveKerrResponse(ARGON, "polynomial", thg=False)], grid
    )
    ledger = ledger_of(problem, grid)
    rows = min(problem.transform.node_chunk, problem.transform.n_nodes)
    field = rows * grid.n_to * 16  # the analytic family: complex products
    assert ledger["dispersive_products"] == 3 * field
    assert ledger["dispersive_product_spectra"] == 3 * field
    assert ledger["dispersive_filtered_time"] == 0
    assert ledger["dispersive_node_fields"] == 0


def test_a_planar_polynomial_holds_two_node_slots():
    problem, grid = planar_problem([DispersiveKerrResponse(ARGON, "polynomial")])
    ledger = ledger_of(problem, grid)
    assert ledger["dispersive_node_fields"] == 2 * ledger["node_field"]
    assert ledger["dispersive_filtered_time"] > 0
    assert ledger["dispersive_products"] > 0
