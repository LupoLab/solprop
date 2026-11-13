# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 6's dispersive electronic response on native: what is the backend's own.

The parity ladder against ``numpy_ref`` on every route is the case catalogue's
(``backends/dispersive_rows.py``). Here: a launch with zero-frequency content, the
embedded rule's finish, a vacuum's exact zero, a coupled flow beside the response,
repeatability, the thread count and the memory ledger, under the bounds of
``dispersive_cases.py`` and ``dispersive_route_cases.py``.
"""

import dataclasses
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import flow_parity
import parity_ladder
from backends.dispersive_cases import (
    GRID_CASES,
    POLICIES,
    averaged,
    capillary,
    radial,
    run,
)
from backends.dispersive_route_cases import (
    ARGON,
    EVALUATORS,
    free_grid,
    free_problem,
    general_problem,
    planar_problem,
    route_case,
    strand,
)
from backends.dispersive_rows import (
    capillary_evaluation,
    falling_mixture,
    route_evaluation,
    window_of,
)
from backends.forced_threads import assert_threaded
from backends.matched import fixed_order, native_thread_fields
from bounds import assert_within
from parity_ladder import Built
from sessions.runs import load_native, native_source
from solprop.backends.node_chunk import plan_node_chunk
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.session import ExecutionOptions
from solprop.driver import prop_capillary
from solprop.modes import CoherentWall
from solprop.nonlinear.responses import DispersiveKerrResponse
from solprop.output.memory import MemoryOutput
from solprop.physdata import MaterialSelection, Mixture
from solprop.stepper import FixedSchedule
from solprop.translate.stage import scalars

solprop_native = pytest.importorskip("solprop_native")
NativeBackend = solprop_native.NativeBackend

ON_DEVICE = solprop_native.build_info().device_backend != "none"
host_only = pytest.mark.skipif(
    ON_DEVICE,
    reason="the thread count is a host notion; a device build is held run to run",
)

PROBE = Path(__file__).parent / "native_dispersive_thread_probe.py"

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


def built(simulation) -> Built:
    return Built(simulation.translate(), simulation._run_request(), simulation.grid)


@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("thg", [True, False])
def test_a_state_with_zero_frequency_content_matches_the_reference(thg, evaluator):
    """A launch with zero-frequency content is evaluated as the state native holds.

    ``Pi_+`` excludes the zero-frequency bin, which the engine's analytic family keeps.
    The bin lies outside the active band here, so the native load masks it, as every
    load and accepted step does: native's evaluation is of the masked launch, which
    the reference's evaluation of the same state is held to.
    """
    case = built(capillary(evaluator, thg=thg))
    problem = case.problem
    active = np.asarray(problem.grids.active_mask)
    assert not active[0]
    start = problem.initial.field0.copy()
    start[..., 0] = 1e-3 * np.abs(start).max()
    backend = NativeBackend(problem, case.run_request, grid=case.grid)
    try:
        held = load_native(backend, start).field_to_host().reshape(problem.shape)
    finally:
        backend.close()
    np.testing.assert_array_equal(held[..., 0], 0.0)
    np.testing.assert_array_equal(held, start * active)
    masked = dataclasses.replace(
        problem, initial=dataclasses.replace(problem.initial, field0=start * active)
    )
    case = dataclasses.replace(case, problem=masked)
    assert parity_ladder.disagreement(
        parity_ladder.projected(NativeBackend, case),
        parity_ladder.projected(NumpyRefBackend, case),
    ) < capillary_evaluation(case)


@pytest.mark.parametrize("evaluator", ["polynomial", "symmetric"])
def test_the_embedded_rule_is_finished_once_and_only_where_it_exists(evaluator):
    """A second finish of the same evaluation, or one without the rule, does nothing.

    The embedded products are transformed once per evaluation, when the rule is read; a
    repeated read must not land them twice. The radial route has no embedded rule, so
    reading its fine rule leaves it untouched. Explicit no-ops, so exact.
    """
    case = built(capillary(evaluator))
    assert parity_ladder.embedded(case.problem)
    backend = NativeBackend(case.problem, case.run_request, grid=case.grid)
    try:
        native_source(backend, case.problem.initial.field0)
        native = backend._native_state
        fine = np.array(native.projected_to_host(False))
        landed = np.array(native.projected_to_host(True))
        assert np.any(landed)
        np.testing.assert_array_equal(native.projected_to_host(True), landed)
        np.testing.assert_array_equal(native.projected_to_host(False), fine)
    finally:
        backend.close()
    radial_problem, radial_request, grid = radial(evaluator, envelope=False, thg=True)
    assert not parity_ladder.embedded(radial_problem)
    backend = NativeBackend(radial_problem, radial_request, grid=grid)
    try:
        native_source(backend, radial_problem.initial.field0)
        native = backend._native_state
        before = np.array(native.projected_to_host(False))
        assert np.any(before)
        np.testing.assert_array_equal(native.projected_to_host(False), before)
    finally:
        backend.close()


@pytest.mark.parametrize("evaluator", sorted(POLICIES))
def test_a_vacuum_has_exactly_no_source(evaluator):
    """Every member at an exact zero: multipliers exactly zero, and no output (class P).

    A vacuum's filter is a unit one by contract, the C++ test's explicit no-op; its
    source, the multiplier's product, must be an exact zero on both, not a rounding.
    """
    case = built(falling_mixture(evaluator, 0.0, 0.0))
    end = float(case.run_request.stepper.save_planes[-1])
    densities = scalars(case.problem, 0.0, end).dispersive_kerr_densities
    assert np.all(np.asarray(densities) == 0.0)
    for backend_class in (NumpyRefBackend, NativeBackend):
        found = parity_ladder.source(backend_class, case, z=end)
        np.testing.assert_array_equal(found, 0.0)
    # Vacuous unless the same evaluation makes one where there is gas.
    assert np.any(parity_ladder.source(NativeBackend, case))


@pytest.mark.medium
@pytest.mark.parametrize("policy", ["symmetric-kerr", "gas-polynomial"])
def test_a_coherent_wall_flow_beside_the_response_matches_the_reference(policy):
    """The flow carries each stage's input; the response is the fused path's.

    The stage source is the fused path's own, so the flow's ten-step bound holds.
    Measured 1.0e-15-1.1e-15 against a bound of 2.2e-13, where the response moves the
    field from the carrier policy's by 2.2e-7.
    """
    fields = {}
    for name in ("native", "numpy_ref"):
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
                    wavelength=800e-9,
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
                    materials=MaterialSelection(electronic=policy),
                    noise=None,
                ),
            )
        fields[name] = output.planes()[-1].field
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = flow_parity.build("scalar")
        problem = simulation.translate()
    bound = flow_parity.ten_step_bound(
        flow_parity.flow_scales(
            problem, simulation._run_request(), simulation.grid, fields["numpy_ref"]
        )
    )
    assert bound > 0
    assert flow_parity.departure(fields["native"], fields["numpy_ref"]) <= bound


# --- repeatability and the thread count ---------------------------------------------


def window_case(route: str, evaluator: str) -> tuple[Built, float]:
    """Return a route's ten-step case at a fixed order, and its window's bound."""
    if route == "capillary":
        case = built(capillary(evaluator, gas="N2", raman=True, stepped=True))
        evaluation = capillary_evaluation
    elif route == "averaged":
        case = Built(*averaged(evaluator, envelope=False, thg=True))
        evaluation = capillary_evaluation
    else:
        case = Built(*route_case(route, evaluator))
        evaluation = route_evaluation
    case = dataclasses.replace(case, run_request=fixed_order(case.run_request))
    return case, window_of(evaluation)(case)


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize(
    "route",
    [
        "capillary",
        pytest.param("free", marks=pytest.mark.medium),
        pytest.param("planar", marks=pytest.mark.medium),
        pytest.param("general", marks=pytest.mark.medium),
    ],
)
def test_two_runs_of_one_build_agree(route, evaluator):
    """Two runs of one build: the same fixed-order method on the same fixed schedule.

    A repeated run is the same discrete computation, so it is held to the window's
    rounding budget, not to its bits: a threaded reduction may order its sums
    differently from one run to the next.
    """
    case, bound = window_case(route, evaluator)
    first, _ = run(NativeBackend, case.problem, case.run_request, case.grid)
    second, _ = run(NativeBackend, case.problem, case.run_request, case.grid)
    assert_within(second, first, bound, what="repeat run")


@pytest.mark.medium
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@host_only
def test_one_thread_and_four_agree(tmp_path):
    """Ten steps at one thread and at four, each in a fresh process, on every route.

    The same fixed-order method on the same schedule, so the two differ by the order
    of their threaded transforms and node reductions alone: the window's rounding
    budget, as against the reference. A repeat at one thread is held to it too.
    """
    averaged_names = ["symmetric", "polynomial"]
    names = averaged_names + [
        f"{route}:{evaluator}"
        for route in ("free", "planar", "general")
        for evaluator in EVALUATORS
    ]
    one = native_thread_fields(PROBE, 1, names, tmp_path)
    again = native_thread_fields(PROBE, 1, averaged_names, tmp_path)
    four = native_thread_fields(PROBE, 4, names, tmp_path)
    assert_threaded(four["sizing"], 4, "kernels.stage", "kernels.response", "fft.fft")
    for name in names:
        route, _, evaluator = name.rpartition(":")
        _, bound = window_case(route or "averaged", evaluator)
        if name in averaged_names:
            assert_within(again[name], one[name], bound, what=f"{name} repeat")
        assert_within(four[name], one[name], bound, what=f"{name} four threads")


# --- the memory ledger -------------------------------------------------------------


def ledger_of(case: Built) -> dict[str, int]:
    """Return the backend's ledger and check its rows are released on close."""
    backend = NativeBackend(
        case.problem, case.run_request, grid=case.grid, options=case.options
    )
    try:
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    # Native's views go with its state, which refuses to report any more once closed.
    with pytest.raises(RuntimeError, match="closed"):
        backend.memory_ledger()
    assert all(row in ledger for row in DISPERSIVE_ROWS)
    assert all(ledger[row] >= 0 for row in DISPERSIVE_ROWS)
    return ledger


def columns_and_samples(case: Built) -> tuple[int, int]:
    return case.problem.shape[0] * case.problem.shape[1], case.problem.grids.n_to


def test_a_lone_symmetric_response_without_an_in_stage_absorber_holds_no_field():
    """SEP 6's memory target: the stage's own buffers, filtered in place."""
    case = built(capillary("symmetric", stepped=True))
    ledger = ledger_of(case)
    for row in DISPERSIVE_ROWS[:-1]:
        assert ledger[row] == 0, row
    n_bins = case.problem.grids.n_to // 2 + 1
    # Its input and output rows, their stage copies and its weights at least.
    assert ledger["dispersive_multipliers"] >= 5 * n_bins * 8


def test_a_lone_symmetric_mixture_holds_no_field_either():
    """A mixture is one product under its filter, so it takes the lone path too."""
    mixture = {"He": 0.6, "Ar": 0.4}
    ledger = ledger_of(built(capillary("symmetric", gas=mixture, stepped=True)))
    for row in DISPERSIVE_ROWS[:-1]:
        assert ledger[row] == 0, row


def test_a_lone_symmetric_response_beside_the_collar_holds_one_field():
    """The in-stage absorber reads the bare modal field: one filtered field besides.

    Its spectrum is the stage's projected spectrum, free until the stage's forward
    transform.
    """
    case = built(capillary("symmetric"))
    assert case.problem.responses.absorber_scheme == "nonlinear"
    ledger = ledger_of(case)
    n_columns, n_to = columns_and_samples(case)
    assert ledger["dispersive_filtered_time"] == n_columns * n_to * 8
    for row in ("dispersive_node_fields", "dispersive_products"):
        assert ledger[row] == 0, row
    assert ledger["dispersive_product_spectra"] == 0


def test_the_polynomial_holds_two_derivative_fields_and_its_products():
    """Two filtered fields, two node slots and three products per rule."""
    case = built(capillary("polynomial", thg=False))
    ledger = ledger_of(case)
    n_columns, n_to = columns_and_samples(case)
    field = n_columns * n_to * 16  # the analytic family: complex fields
    assert ledger["dispersive_filtered_time"] == 2 * field
    assert ledger["dispersive_products"] == 3 * field
    assert ledger["dispersive_coarse_products"] == 3 * field
    assert ledger["dispersive_product_spectra"] == 3 * field
    assert ledger["dispersive_node_fields"] == 2 * ledger["node_field"]
    # One chunk: the stage's scaled projections are read in place.
    assert ledger["dispersive_projection_chunks"] == 0


@pytest.mark.parametrize("case", ["field full", "field conserving", "envelope full"])
def test_species_with_one_leg_bound_hold_one_group(case):
    """Krypton and xenon share ``D E`` and ``D^2 E``: two filtered fields, three
    products."""
    band = {"wavelength_limits": (180e-9, 4e-6), "energy": 5e-6}
    gas = {"Kr": 0.5, "Xe": 0.5}
    one = built(capillary("polynomial", gas=gas, **band, **GRID_CASES[case]))
    ledger = ledger_of(one)
    n_columns, n_to = columns_and_samples(one)
    item = 8 if case == "field full" else 16
    assert ledger["dispersive_filtered_time"] == 2 * n_columns * n_to * item
    assert ledger["dispersive_products"] == 3 * n_columns * n_to * item


def test_a_response_beside_raman_holds_its_own_node_field_and_products():
    """The bare field stays the column's; the products are landed in its spectrum."""
    case = built(capillary("symmetric", gas="N2", raman=True, thg=False))
    ledger = ledger_of(case)
    n_columns, n_to = columns_and_samples(case)
    assert ledger["dispersive_node_fields"] == ledger["node_field"]
    assert ledger["dispersive_products"] == n_columns * n_to * 16
    assert ledger["dispersive_product_spectra"] == n_columns * n_to * 16


def test_chunked_nodes_pack_their_projections():
    """More than one chunk: each chunk's columns of both scaled projections, once."""
    case = dataclasses.replace(
        built(capillary("polynomial")), options=ExecutionOptions(node_chunk=8)
    )
    ledger = ledger_of(case)
    chunks = -(-case.problem.transform.n_nodes // 8)
    per_rule = chunks * case.problem.shape[0] * 8 * case.problem.transform.n_pol * 8
    assert ledger["dispersive_projection_chunks"] == 2 * per_rule


def test_a_problem_without_the_leaf_names_the_rows_at_zero():
    case = built(capillary("carrier-kerr"))
    backend = NativeBackend(case.problem, case.run_request, grid=case.grid)
    try:
        ledger = backend.memory_ledger()
        metadata = backend.metadata()
    finally:
        backend.close()
    assert all(ledger[row] == 0 for row in DISPERSIVE_ROWS)
    assert metadata["dispersive_path"] is None


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_a_lone_symmetric_free_response_holds_no_field():
    """Filtered in place, its product is the chunk's own polarisation."""
    grid = free_grid("field", True)
    response = DispersiveKerrResponse(ARGON, "symmetric")
    ledger = ledger_of(Built(*free_problem([response], grid), grid))
    for row in DISPERSIVE_ROWS[:-1]:
        assert ledger[row] == 0, row
    assert ledger["dispersive_multipliers"] > 0


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_a_free_polynomial_writes_its_filtered_fields_into_its_product_blocks():
    """Three products of a chunk in time and frequency; no filtered field besides."""
    grid = free_grid("field", False)
    response = DispersiveKerrResponse(ARGON, "polynomial", thg=False)
    case = Built(*free_problem([response], grid), grid)
    ledger = ledger_of(case)
    field = plan_node_chunk(case.problem).chunk * grid.n_to * 16  # complex products
    assert ledger["dispersive_products"] == 3 * field
    assert ledger["dispersive_product_spectra"] == 3 * field
    assert ledger["dispersive_filtered_time"] == 0
    assert ledger["dispersive_node_fields"] == 0


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_a_planar_polynomial_holds_two_node_slots():
    case = Built(*planar_problem([DispersiveKerrResponse(ARGON, "polynomial")]))
    ledger = ledger_of(case)
    assert ledger["dispersive_node_fields"] == 2 * ledger["node_field"]
    assert ledger["dispersive_filtered_time"] > 0
    assert ledger["dispersive_products"] > 0


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_a_general_polynomial_region_forms_one_cube():
    """Krypton and xenon in one region: two complex filtered fields, three real
    products."""
    gas = Mixture.fill({"Kr": 0.5, "Xe": 0.5}, 1.0)
    case = Built(*general_problem(strand(gas), "polynomial", "real"))
    ledger = ledger_of(case)
    assert 4 * ledger["dispersive_products"] == 3 * ledger["dispersive_filtered_time"]
