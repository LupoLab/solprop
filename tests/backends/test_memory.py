# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The construction-free memory estimate against every backend's ledger.

:func:`solprop.backends.memory.estimate_memory` must equal what a constructed backend's
:meth:`memory_ledger` holds, group by group and to the byte, on every case of
``tests/memory_cases.py`` -- the route groups on all three backends, the ``base`` group
on the reference whose rows it is.
"""

import dataclasses
import importlib.util
import warnings

import numpy as np
import pytest

from driver.test_moving_guide import LENGTH, build, grin_taper, kinked_radius
from modal_provider_cases import grin_grid, grin_guide
from solprop.backends.memory import (
    MEMORY_GROUPS,
    MemoryEstimate,
    dispersive_memory,
    estimate_guide_memory,
    estimate_memory,
    ledger_groups,
    reference_base_memory,
)
from solprop.backends.registry import available, select
from solprop.driver import Simulation
from solprop.exceptions import BackendError, HostMarchWarning, SolpropWarning
from solprop.stepper import FixedSchedule, Multistep
from solprop.stepper.controller import PIController


def _benchmark():
    """Return the memory-estimate cases of ``tests/memory_cases.py``."""
    return importlib.import_module("memory_cases")


@pytest.fixture(scope="module")
def gnlse_problem(soliton_problem, narrow_grid):
    """Return a fundamental soliton's problem on the legacy mode-set route."""
    return soliton_problem(narrow_grid, 1, save_planes=[0.01])


CASES = [
    (case, size)
    for case in ("general", "raman", "wall", "te_tm", "taper", "split")
    for size in (0, 1)
]
BACKENDS = [
    name for name in ("numpy_ref", "numba_cpu", "native") if name in available()
]
ROUTE_GROUPS = tuple(group for group in MEMORY_GROUPS if group != "base")


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
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("case", "size"), CASES)
def test_estimate_memory_equals_the_ledger_to_the_byte(problems, case, size, backend):
    problem, grid = problems(case, size)
    estimate = estimate_memory(problem, backend)
    constructed = select(backend, problem, grid=grid)
    try:
        held = ledger_groups(constructed.memory_ledger())
    finally:
        constructed.close()
    if backend == "native":
        # The adapter leaves the Python-owned reference out of its native ledger; it is
        # still resident, and the estimate counts it at the reference's own bytes, which
        # the numpy_ref case checks against its ledger.
        assert held["general_reference"] == 0
        held["general_reference"] = estimate_memory(problem, "numpy_ref").groups[
            "general_reference"
        ]
    for group in ROUTE_GROUPS:
        assert estimate.groups[group] == held[group], group
    if backend == "numpy_ref":
        assert estimate.groups["base"] == held["base"]
        assert held["measured"] == 0
    # The route the case exercises is present, and the reference is counted as resident.
    route = "coupled_flow" if case in {"wall", "te_tm", "taper"} else "general_maps"
    assert estimate.groups[route] > 0
    assert estimate.resident == estimate.groups["general_reference"]
    assert estimate.not_counted


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_estimate_memory_of_the_legacy_route_is_its_base(gnlse_problem, backend):
    estimate = estimate_memory(gnlse_problem, backend)
    assert estimate.total == estimate.groups["base"] > 0
    assert estimate.resident == 0
    assert estimate.largest() == (("base", estimate.total),)


def test_estimate_memory_matches_the_reference_ledger_on_the_legacy_route(
    gnlse_problem,
):
    constructed = select("numpy_ref", gnlse_problem)
    try:
        held = ledger_groups(constructed.memory_ledger())
    finally:
        constructed.close()
    estimate = estimate_memory(gnlse_problem, "numpy_ref")
    assert {group: estimate.groups[group] for group in MEMORY_GROUPS} == {
        group: held[group] for group in MEMORY_GROUPS
    }


def test_estimate_memory_resolves_auto(gnlse_problem):
    assert estimate_memory(gnlse_problem, "auto").backend in {"native", "numba_cpu"}


def test_estimate_memory_refuses_an_unknown_backend(gnlse_problem):
    with pytest.raises(BackendError, match="no memory estimate for the backend 'jax'"):
        estimate_memory(gnlse_problem, "jax")


def test_ledger_groups_sorts_every_route_row():
    ledger = {
        "state": 1,
        "general_synthesis": 2,
        "general_coordinate_dual": 4,
        "general_node_field": 8,
        "general_node_product": 16,
        "flow_forward": 32,
        "coupled_flow_actions": 64,
        "general_fft_plan_heap_growth": 128,
    }
    assert ledger_groups(ledger) == {
        "base": 1,
        "general_maps": 2,
        "general_reference": 4,
        "general_chunk": 8,
        "general_delayed": 16,
        "coupled_flow": 96,
        "adams_history": 0,
        "measured": 128,
    }


def test_memory_estimate_orders_its_largest_groups():
    estimate = MemoryEstimate("numba_cpu", {"base": 3, "general_maps": 7, "x": 0})
    assert estimate.largest(5) == (("general_maps", 7), ("base", 3))
    assert estimate.total == 10


def test_reference_base_memory_counts_noise_and_the_embedded_projection(
    gnlse_problem,
):
    arguments = {
        "tabulating": True,
        "statistics_weight": 0,
        "spatial_absorber": 0,
    }
    plain = reference_base_memory(
        gnlse_problem, embedded=False, noise=False, **arguments
    )
    loaded = reference_base_memory(
        gnlse_problem, embedded=True, noise=True, **arguments
    )
    field = int(np.prod(gnlse_problem.shape)) * 16
    assert loaded["noise"] - plain["noise"] == field
    assert loaded["projected_coarse"] == loaded["projected"] > 0


@pytest.mark.medium
# A flow on the host engine runs solprop's march on native, with its warning (SEP 39).
@pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning")
@pytest.mark.parametrize("backend", BACKENDS)
def test_estimate_memory_counts_the_bins_formed_by_scaling_and_squaring(
    problems, backend, monkeypatch
):
    """With no eigenbasis admitted, Kokkos holds every bin's Schur factors (E6)."""
    monkeypatch.setattr(
        "solprop.translate.coupled.eigenbasis_limit", lambda n_modes: 0.0
    )
    problem, grid = problems("taper", 0)
    estimate = estimate_memory(problem, backend)
    constructed = select(backend, problem, grid=grid)
    try:
        ledger = constructed.memory_ledger()
    finally:
        constructed.close()
    assert estimate.groups["coupled_flow"] == ledger_groups(ledger)["coupled_flow"]
    if backend == "native":
        assert ledger["flow_fallback"] > 0


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu"])
def test_estimate_memory_of_a_moving_guide_equals_the_ledger_to_the_byte(backend):
    """The kinked graded-index taper: every coefficient of each map and the stage maps.

    ``numpy_ref`` and ``numba_cpu`` hold the same map rows, the coefficients and the
    two maps a stage evaluates, and every other route group by their own formulas;
    Kokkos has its own test below.
    """
    grid = grin_grid()
    simulation = build(grin_taper(kinked_radius(), kerr=True), grid)
    simulation.settings = dataclasses.replace(simulation.settings, absorber=None)
    problem = simulation.translate()
    estimate = estimate_memory(problem, backend)
    constructed = select(backend, problem, grid=grid)
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


@pytest.mark.medium
@pytest.mark.skipif("native" not in available(), reason="native is not installed")
def test_estimate_memory_counts_a_moving_guide_to_the_byte_on_native():
    """Every coefficient of both maps, the two a stage evaluates, and the weights.

    Kokkos holds a moving guide's maps as ``numpy_ref`` does and a trial's
    ``transform_weights`` beside them, one row per stage; ``estimate_memory`` equals
    its native ledger group by group, the Python-owned reference aside (the adapter
    leaves it out of the native ledger).
    """
    grid = grin_grid()
    length, steps = 1e-3, 10
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = Simulation.from_waveguide(
            (),
            length,
            provider=grin_taper(kinked_radius(), kerr=True),
            grid=grid,
            save_planes=[length],
            dz_max=length / steps,
            dz_init=length / steps,
            schedule=FixedSchedule(np.full(steps, length / steps)),
            wavelength=1.03e-6,
            fwhm=60e-15,
            energy=1e-12,
            shape="gauss",
            noise=None,
        )
        simulation.settings = dataclasses.replace(simulation.settings, absorber=None)
        problem = simulation.translate()
    estimate = estimate_memory(problem, "native")
    constructed = select("native", problem, grid=grid)
    try:
        ledger = constructed.memory_ledger()
    finally:
        constructed.close()
    held = ledger_groups(ledger)
    assert held["general_reference"] == 0
    held["general_reference"] = estimate_memory(problem, "numpy_ref").groups[
        "general_reference"
    ]
    for group in ROUTE_GROUPS:
        assert estimate.groups[group] == held[group], group
    tables = problem.transform.modal_tables
    assert tables is not None
    assert tables.synthesis is not None
    one = tables.synthesis[0, 0].nbytes
    terms = int(tables.synthesis.shape[0] * tables.synthesis.shape[1])
    stages = problem.stepper.tableau.n_stages
    assert terms == 10
    assert ledger["general_synthesis"] == terms * one
    assert ledger["general_projection"] == terms * one
    assert ledger["general_stage_maps"] == 2 * one
    assert ledger["general_map_weights"] == 8 * stages * terms


GUIDE_GROUPS = ("general_reference", "general_maps", "general_chunk")
"""The route groups a guide's declared shape fixes before translation."""


@pytest.mark.parametrize("moving", [False, True], ids=["snapshot", "moving"])
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_the_guide_estimate_equals_the_translated_one_to_the_byte(moving, backend):
    """The declared shape gives the route's three groups before anything is built.

    Bound, stated first: **exact**, byte for byte, group by group -- both are integer
    arithmetic on the same sizes, one read from the guide and the grid, the other from
    the arrays translation built. The snapshot is the graded-index
    guide on its own grid; the moving one is the kinked taper, whose maps are a
    section's Chebyshev coefficients and whose reference is stored at the kept
    sections' ends. Both carry the backward partners, as both shipped providers do.
    Formulas only: no backend is constructed, so Kokkos's formulas are checked
    whether or not it is installed.
    """
    grid = grin_grid()
    guide = (
        grin_taper(kinked_radius(), kerr=True) if moving else grin_guide(grid, knots=9)
    )
    before = estimate_guide_memory(guide, grid, z_max=LENGTH, backend=backend)
    # The Kerr guides' ten 100 um steps are far past the in-stage collar's stability
    # limit, which translation refuses; the per-step collar has none and leaves the
    # guide groups compared here unchanged.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = build(guide, grid)
        simulation.settings = dataclasses.replace(
            simulation.settings, absorber_scheme="split_step"
        )
        problem = simulation.translate()
    after = estimate_memory(problem, backend)
    tables = problem.transform.modal_tables
    assert tables is not None and tables.backward_electric is not None
    for group in GUIDE_GROUPS:
        assert before.groups[group] == after.groups[group] > 0, group
    assert before.total == sum(after.groups[group] for group in GUIDE_GROUPS)
    assert before.resident == 0
    assert any("translation fixes" in line for line in before.not_counted)


def test_the_guide_estimate_refuses_what_it_cannot_estimate():
    """An unknown backend and a guide without a sampling are refused by name."""
    grid = grin_grid()
    guide = grin_guide(grid)
    with pytest.raises(BackendError, match="no memory estimate"):
        estimate_guide_memory(guide, grid, z_max=LENGTH, backend="jax")
    bare = dataclasses.replace(guide, sampling=None)
    with pytest.raises(BackendError, match="without a sampling"):
        estimate_guide_memory(bare, grid, z_max=LENGTH, backend="numpy_ref")


def _adams_problem(case: str, formulation: str):
    """Return a translated problem and its simulation for an Adams memory case."""
    from capability_routes import ROUTES  # noqa: PLC0415
    from solprop.stepper.multistep import Multistep  # noqa: PLC0415

    simulation = ROUTES[case](None)
    integrator = Multistep(formulation=formulation)  # pyright: ignore[reportArgumentType]
    simulation.settings = type(simulation.settings)(
        **{
            **{
                name: getattr(simulation.settings, name)
                for name in simulation.settings.__dataclass_fields__
            },
            "integrator": integrator,
        }
    )
    return simulation, simulation.translate()


@pytest.mark.parametrize("formulation", ["interaction", "exponential"])
@pytest.mark.parametrize(
    "case", ["gnlse", "capillary (pressure gradient)", "cartesian"]
)
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
@pytest.mark.parametrize("order", [1, 5, 6])
@pytest.mark.parametrize("interpolate", [False, True])
def test_the_adams_history_is_estimated_to_the_byte(
    backend, case, formulation, order, interpolate
):
    # SEP 16 phase 3b: the history's rows on every backend -- the host engine's store,
    # a varying operator's state and rate rows, numba_cpu's moment scratch and
    # the native backend's resident engine -- equal their construction-free estimate.
    if backend not in available():
        pytest.skip(f"{backend} is not installed")
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
    estimate = estimate_memory(problem, backend).groups["adams_history"]
    constructed = select(backend, problem, grid=simulation.grid, modes=simulation.modes)
    try:
        held = ledger_groups(constructed.memory_ledger())["adams_history"]
    finally:
        constructed.close()
    assert estimate == held > 0


def test_noninterpolating_native_adams_saves_eleven_field_volumes():
    _, problem = _adams_problem("cartesian", "exponential")
    plain = dataclasses.replace(
        problem, stepper=dataclasses.replace(problem.stepper, interpolate_planes=False)
    )
    dense = dataclasses.replace(
        problem, stepper=dataclasses.replace(problem.stepper, interpolate_planes=True)
    )
    # Seven dense slopes and four saved fields, all unused when planes are landed on.
    saving = (
        estimate_memory(dense, "native").groups["adams_history"]
        - estimate_memory(plain, "native").groups["adams_history"]
    )
    assert saving == 11 * problem.initial.field0.nbytes


class HostScaleController(PIController):
    """solprop's controller with its scale overridden: formed on the host."""

    def scale(self, norm_y, norm_y_new):
        """Return the inherited scale, from this subclass's own method."""
        return super().scale(norm_y, norm_y_new)


@pytest.mark.skipif("native" not in available(), reason="needs solprop-native")
def test_a_controller_s_own_scale_is_not_held_in_the_native_engine():
    # solprop's own controller has its scale formed in the resident engine, from a
    # magnitude sum held there; a controller that overrides `scale` has it formed on
    # the host and uploaded, so the engine holds no such row -- and the estimate,
    # still to the byte, is that much smaller.
    simulation, problem = _adams_problem("gnlse", "exponential")
    own = dataclasses.replace(
        problem,
        stepper=dataclasses.replace(
            problem.stepper, controller=HostScaleController(rtol=1e-6, atol=0.0)
        ),
    )
    estimate = estimate_memory(own, "native").groups["adams_history"]
    # A scale of its own runs solprop's march, and says so (SEP 39).
    with pytest.warns(HostMarchWarning, match="norm scale of its own"):
        constructed = select(
            "native", own, grid=simulation.grid, modes=simulation.modes
        )
    try:
        held = ledger_groups(constructed.memory_ledger())["adams_history"]
    finally:
        constructed.close()
    assert estimate == held
    assert estimate == estimate_memory(problem, "native").groups["adams_history"] - 8


@pytest.mark.parametrize("case", ["scalar", "taper"])
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_the_coupled_adams_history_is_estimated_to_the_byte(backend, case):
    # SEP 16 phases 5 and 6: beside a coupled flow the exponential formulation's
    # history -- with a frozen-residual flow's compensation rows, and on ``native`` the
    # resident block functions and the flow weights per sample -- equals its estimate.
    if backend not in available():
        pytest.skip(f"{backend} is not installed")
    import flow_parity  # noqa: PLC0415 - the coupled cases' builder
    from solprop.stepper.multistep import Multistep  # noqa: PLC0415

    with warnings.catch_warnings():
        for category in (*flow_parity.QUIET, SolpropWarning):
            warnings.simplefilter("ignore", category)
        simulation = flow_parity.build(case, schedule=None)
        simulation.settings = dataclasses.replace(
            simulation.settings, integrator=Multistep(formulation="exponential")
        )
        problem = simulation.translate()
    estimate = estimate_memory(problem, backend).groups["adams_history"]
    constructed = select(backend, problem, grid=simulation.grid, modes=simulation.modes)
    try:
        held = ledger_groups(constructed.memory_ledger())["adams_history"]
    finally:
        constructed.close()
    assert estimate == held > 0


DISPERSIVE_CASES = {
    # A lone symmetric response filtering the stage's spectrum in place: nothing held.
    "lone in place": ("symmetric-kerr", {"stepped": True}),
    # Beside the in-stage collar: its filtered field.
    "lone beside the collar": ("symmetric-kerr", {}),
    # A field grid's conserving sector: complex products beside a real polarisation.
    "analytic": ("symmetric-kerr", {"thg": False}),
    "polynomial envelope": ("gas-polynomial", {"envelope": True, "thg": True}),
    "beside Raman": ("symmetric-kerr", {"gas": "N2", "raman": True}),
    "polynomial mixture": ("gas-polynomial", {"gas": {"He": 0.6, "Ar": 0.4}}),
}


@pytest.mark.parametrize("case", sorted(DISPERSIVE_CASES))
@pytest.mark.parametrize("backend", ["numba_cpu", "native"])
def test_the_dispersive_buffers_are_estimated_to_the_byte(backend, case):
    # SEP 6, Phase 2c's open question 3 (decided 2026-10-08): on the fused node path
    # both compiled backends hold the response's filtered fields, node slots, products
    # and spectra by one rule, and the estimate counts them in `base`, row for row.
    if backend not in available():
        pytest.skip(f"{backend} is not installed")
    from backends.test_numba_dispersive_kerr import capillary  # noqa: PLC0415

    policy, options = DISPERSIVE_CASES[case]
    simulation = capillary(policy, **options)
    problem = simulation.translate()
    rows = dispersive_memory(problem, backend)
    constructed = select(backend, problem, grid=simulation.grid)
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
    assert estimate_memory(problem, backend).groups["base"] == reference.groups[
        "base"
    ] + sum(rows.values())


@pytest.mark.parametrize("backend", ["numba_cpu", "native"])
def test_chunked_dispersive_node_slots_are_estimated_to_the_byte(backend):
    # Five chunks of eight nodes: the node slots are a chunk's, the rest the columns'.
    if backend not in available():
        pytest.skip(f"{backend} is not installed")
    from backends.test_numba_dispersive_kerr import capillary  # noqa: PLC0415

    simulation = capillary("gas-polynomial")
    problem = simulation.translate()
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=8)
    )
    rows = dispersive_memory(problem, backend)
    components = problem.transform.n_pol * problem.shape[2]
    assert rows["dispersive_node_fields"] == 2 * 8 * components * problem.grids.n_to * 8
    constructed = select(backend, problem, grid=simulation.grid)
    try:
        ledger = constructed.memory_ledger()
    finally:
        constructed.close()
    for row, size in rows.items():
        assert ledger[row] == size, row


def test_the_spatial_dispersive_buffers_are_the_admitted_ones():
    # The Cartesian route counts, as the general route's chunk does, the buffers per
    # node and component its node budget admits for the response; numba_cpu holds no
    # more than that. Kokkos's own rows are held to its ledger to the byte
    # (`test_native_dispersive_routes.py`, SEP 6 Phase 3b).
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
    rows = dispersive_memory(problem, "numba_cpu")
    assert rows == {
        "dispersive_node_buffers": 16 * chunk * components * buffers * grid.n_to
    }
    native = dispersive_memory(problem, "native")
    assert set(native) == {
        "dispersive_filtered_time",
        "dispersive_node_fields",
        "dispersive_products",
        "dispersive_product_spectra",
    }
    # Free space's filtered fields are blocks of its products: three complex products
    # and their spectra over the chunk's columns.
    field = 16 * chunk * components * grid.n_to
    assert native == {
        "dispersive_filtered_time": 0,
        "dispersive_node_fields": 0,
        "dispersive_products": 3 * field,
        "dispersive_product_spectra": 3 * field,
    }
    constructed = select("numba_cpu", problem, grid=grid)
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
