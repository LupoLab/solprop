# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The construction-free memory estimate against ``numba_cpu``'s ledger.

``test_memory.py``'s comparisons on this backend: the route groups to the byte on every
case of ``tests/memory_cases.py``, and its own rows beside the reference's.
"""

import dataclasses
import warnings

import pytest

from backends.dispersive_cases import capillary
from backends.test_memory import (
    CASES,
    DISPERSIVE_CASES,
    GUIDE_GROUPS,
    ROUTE_GROUPS,
    _adams_problem,
    _benchmark,
)
from driver.test_moving_guide import LENGTH, build, grin_taper, kinked_radius
from modal_provider_cases import grin_grid, grin_guide
from solprop.backends.memory import (
    dispersive_memory,
    estimate_guide_memory,
    estimate_memory,
    ledger_groups,
)
from solprop.backends.registry import select
from solprop.exceptions import SolpropWarning
from solprop.stepper import Multistep

BACKEND = "numba_cpu"


@pytest.fixture(scope="module")
def gnlse_problem(soliton_problem, narrow_grid):
    """Return a fundamental soliton's problem on the legacy mode-set route."""
    return soliton_problem(narrow_grid, 1, save_planes=[0.01])


@pytest.fixture(scope="module")
def problems():
    """Build each case's problem and grid once, construction only."""
    built = {}

    def build(case, size):
        if (case, size) not in built:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                built[case, size] = _benchmark().SIZES[case][size]()
        return built[case, size]

    return build


@pytest.mark.medium
@pytest.mark.parametrize(("case", "size"), CASES)
def test_estimate_memory_equals_the_ledger_to_the_byte(problems, case, size):
    problem, grid = problems(case, size)
    estimate = estimate_memory(problem, BACKEND)
    constructed = select(BACKEND, problem, grid=grid)
    try:
        held = ledger_groups(constructed.memory_ledger())
    finally:
        constructed.close()
    for group in ROUTE_GROUPS:
        assert estimate.groups[group] == held[group], group
    # The route the case exercises is present, and the reference is counted as resident.
    route = "coupled_flow" if case in {"wall", "te_tm", "taper"} else "general_maps"
    assert estimate.groups[route] > 0
    assert estimate.resident == estimate.groups["general_reference"]
    assert estimate.not_counted


def test_estimate_memory_of_the_legacy_route_is_its_base(gnlse_problem):
    estimate = estimate_memory(gnlse_problem, BACKEND)
    assert estimate.total == estimate.groups["base"] > 0
    assert estimate.resident == 0
    assert estimate.largest() == (("base", estimate.total),)


@pytest.mark.medium
# A flow on the host engine runs solprop's march on native, with its warning (SEP 39).
@pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning")
def test_estimate_memory_counts_the_bins_formed_by_scaling_and_squaring(
    problems, monkeypatch
):
    monkeypatch.setattr(
        "solprop.translate.coupled.eigenbasis_limit", lambda n_modes: 0.0
    )
    problem, grid = problems("taper", 0)
    estimate = estimate_memory(problem, BACKEND)
    constructed = select(BACKEND, problem, grid=grid)
    try:
        ledger = constructed.memory_ledger()
    finally:
        constructed.close()
    assert estimate.groups["coupled_flow"] == ledger_groups(ledger)["coupled_flow"]


def test_estimate_memory_of_a_moving_guide_equals_the_ledger_to_the_byte():
    """The kinked graded-index taper: every coefficient of each map and the stage maps.

    ``numba_cpu`` holds the reference's map rows, the coefficients and the two maps a
    stage evaluates, and every other route group by its own formulas.
    """
    grid = grin_grid()
    simulation = build(grin_taper(kinked_radius(), kerr=True), grid)
    simulation.settings = dataclasses.replace(simulation.settings, absorber=None)
    problem = simulation.translate()
    estimate = estimate_memory(problem, BACKEND)
    constructed = select(BACKEND, problem, grid=grid)
    try:
        held = ledger_groups(constructed.memory_ledger())
    finally:
        constructed.close()
    for group in ROUTE_GROUPS:
        assert estimate.groups[group] == held[group], group
    tables = problem.transform.modal_tables
    assert tables is not None and tables.synthesis is not None
    one = tables.synthesis[0, 0].nbytes
    assert estimate.groups["general_maps"] >= (2 * 10 + 2) * one


@pytest.mark.parametrize("moving", [False, True], ids=["snapshot", "moving"])
def test_the_guide_estimate_equals_the_translated_one_to_the_byte(moving):
    """The declared shape gives the route's three groups before anything is built.

    Bound: **exact** (``test_memory.py``'s test on the other backends).
    """
    grid = grin_grid()
    guide = (
        grin_taper(kinked_radius(), kerr=True) if moving else grin_guide(grid, knots=9)
    )
    before = estimate_guide_memory(guide, grid, z_max=LENGTH, backend=BACKEND)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = build(guide, grid)
        simulation.settings = dataclasses.replace(
            simulation.settings, absorber_scheme="split_step"
        )
        problem = simulation.translate()
    after = estimate_memory(problem, BACKEND)
    tables = problem.transform.modal_tables
    assert tables is not None and tables.backward_electric is not None
    for group in GUIDE_GROUPS:
        assert before.groups[group] == after.groups[group] > 0, group
    assert before.total == sum(after.groups[group] for group in GUIDE_GROUPS)
    assert before.resident == 0
    assert any("translation fixes" in line for line in before.not_counted)


@pytest.mark.parametrize("formulation", ["interaction", "exponential"])
@pytest.mark.parametrize(
    "case", ["gnlse", "capillary (pressure gradient)", "cartesian"]
)
@pytest.mark.parametrize("order", [1, 5, 6])
@pytest.mark.parametrize("interpolate", [False, True])
def test_the_adams_history_is_estimated_to_the_byte(
    case, formulation, order, interpolate
):
    # SEP 16 phase 3b: numba_cpu's moment scratch and the host engine's store equal
    # their construction-free estimate.
    simulation, problem = _adams_problem(case, formulation)
    assert isinstance(problem.stepper.integrator, Multistep)
    problem = dataclasses.replace(
        problem,
        stepper=dataclasses.replace(
            problem.stepper,
            integrator=dataclasses.replace(problem.stepper.integrator, order=order),
            interpolate_planes=interpolate,
        ),
    )
    estimate = estimate_memory(problem, BACKEND).groups["adams_history"]
    constructed = select(BACKEND, problem, grid=simulation.grid, modes=simulation.modes)
    try:
        held = ledger_groups(constructed.memory_ledger())["adams_history"]
    finally:
        constructed.close()
    assert estimate == held > 0


@pytest.mark.parametrize("case", ["scalar", "taper"])
def test_the_coupled_adams_history_is_estimated_to_the_byte(case):
    # SEP 16 phases 5 and 6: beside a coupled flow the exponential formulation's
    # history, with a frozen-residual flow's compensation rows, equals its estimate.
    import flow_parity  # noqa: PLC0415 - the coupled cases' builder

    with warnings.catch_warnings():
        for category in (*flow_parity.QUIET, SolpropWarning):
            warnings.simplefilter("ignore", category)
        simulation = flow_parity.build(case, schedule=None)
        simulation.settings = dataclasses.replace(
            simulation.settings, integrator=Multistep(formulation="exponential")
        )
        problem = simulation.translate()
    estimate = estimate_memory(problem, BACKEND).groups["adams_history"]
    constructed = select(BACKEND, problem, grid=simulation.grid, modes=simulation.modes)
    try:
        held = ledger_groups(constructed.memory_ledger())["adams_history"]
    finally:
        constructed.close()
    assert estimate == held > 0


@pytest.mark.parametrize("case", sorted(DISPERSIVE_CASES))
def test_the_dispersive_buffers_are_estimated_to_the_byte(case):
    # SEP 6, Phase 2c's open question 3 (decided 2026-10-08): on the fused node path
    # both compiled backends hold the response's filtered fields, node slots, products
    # and spectra by one rule, and the estimate counts them in `base`, row for row.
    policy, options = DISPERSIVE_CASES[case]
    simulation = capillary(policy, **options)
    problem = simulation.translate()
    rows = dispersive_memory(problem, BACKEND)
    constructed = select(BACKEND, problem, grid=simulation.grid)
    try:
        ledger = constructed.memory_ledger()
    finally:
        constructed.close()
    assert rows
    for row, size in rows.items():
        assert ledger[row] == size, row
    assert (sum(rows.values()) == 0) == (case == "lone in place")
    reference = estimate_memory(problem, "numpy_ref")
    assert dispersive_memory(problem, "numpy_ref") == {}
    assert estimate_memory(problem, BACKEND).groups["base"] == reference.groups[
        "base"
    ] + sum(rows.values())


def test_chunked_dispersive_node_slots_are_estimated_to_the_byte():
    # Five chunks of eight nodes: the node slots are a chunk's, the rest the columns'.
    simulation = capillary("gas-polynomial")
    problem = simulation.translate()
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=8)
    )
    rows = dispersive_memory(problem, BACKEND)
    components = problem.transform.n_pol * problem.shape[2]
    assert rows["dispersive_node_fields"] == 2 * 8 * components * problem.grids.n_to * 8
    constructed = select(BACKEND, problem, grid=simulation.grid)
    try:
        ledger = constructed.memory_ledger()
    finally:
        constructed.close()
    for row, size in rows.items():
        assert ledger[row] == size, row


def test_the_spatial_dispersive_buffers_are_the_admitted_ones():
    # The Cartesian route counts, as the general route's chunk does, the buffers per
    # node and component its node budget admits for the response; numba_cpu holds no
    # more than that.
    from backends.test_dispersive_kerr_routes import (  # noqa: PLC0415
        ARGON,
        free_grid,
        free_problem,
    )
    from solprop.nonlinear.responses import DispersiveKerrResponse  # noqa: PLC0415
    from solprop.translate.problem import dispersive_node_buffers  # noqa: PLC0415

    grid = free_grid("field")
    problem = free_problem(DispersiveKerrResponse(ARGON, "polynomial", thg=False), grid)
    buffers = dispersive_node_buffers(problem.responses, spatial=True)
    transform = problem.transform
    chunk = min(transform.node_chunk, transform.n_nodes)
    components = transform.n_pol * problem.shape[2]
    rows = dispersive_memory(problem, BACKEND)
    assert rows == {
        "dispersive_node_buffers": 16 * chunk * components * buffers * grid.n_to
    }
    constructed = select(BACKEND, problem, grid=grid)
    try:
        ledger = constructed.memory_ledger()
    finally:
        constructed.close()
    held = sum(
        size
        for row, size in ledger.items()
        if row.startswith("dispersive_")
        and row not in ("dispersive_multipliers", "dispersive_fft_plan_heap_growth")
    )
    assert 0 < held <= rows["dispersive_node_buffers"]
