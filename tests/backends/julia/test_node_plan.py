# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check source-derived Julia node bounds and independent retained array payloads."""

import gc
import os
import subprocess
import sys
import textwrap
import weakref
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import numpy as np
import pytest

from backends.julia.cases import InstantaneousRate, material_case
from backends.julia.spatial_cases import spatial_case
from backends.julia.split_cases import split_case
from backends.julia.test_transverse_records import edge_case
from solprop.backends.julia._node_plan import plan_julia_nodes
from solprop.backends.node_chunk import (
    DEFAULT_NODE_BUDGET,
    DEFAULT_SPATIAL_NODE_BUDGET,
)
from solprop.backends.session import ExecutionOptions, Start
from solprop.boundaries import SPLIT_RATE_BUCKETS, Absorber, SplitCrossing
from solprop.exceptions import BackendError
from solprop.grid import FreeGrid, LineGrid, RadialGrid
from solprop.modes import MarcatiliMode
from solprop.monitor import BoundaryCheck, BoundaryMonitor
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.nonlinear.raman import response_for
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.spatial_absorber import SpatialAbsorber, SpatialAbsorberDesign
from solprop.stats import StatsSpec
from solprop.translate.precision import SINGLE
from solprop.translate.problem import Initial, Translation, assemble
from solprop.transverse_monitor import TransverseEdge


def _modal_nodes(case, count=7):
    """Use a valid constant scalar quadrature to expose independent node capacity."""
    transform = replace(
        case.problem.transform,
        kind="modal",
        synthesis=np.ones((count, 1)),
        projection=np.full((1, count), 1.0 / count),
        projection_coarse=np.full((1, count), 1.0 / count),
        node_weights=np.ones(count),
    )
    return Translation(
        replace(case.problem, route=replace(case.problem.route, transform=transform)),
        case.request,
    )


@pytest.fixture(scope="module")
def modal():
    case, _, _ = material_case("envelope", "kerr")
    return _modal_nodes(case)


def _inventory(plan):
    return {entry.name: entry for entry in plan.inventory}


def _statistics(case, names):
    return Translation(
        case.problem,
        case.request.for_run(stats=StatsSpec(names=tuple(names), periodic=())),
    )


def _plural_raman(problem):
    original = problem.responses
    return replace(
        problem,
        responses=replace(
            original,
            labels=(*original.labels, "additional"),
            species_names=(*original.species_names, "additional"),
            species_densities=(*original.species_densities, original.density),
            raman_terms=(
                *original.raman_terms,
                replace(
                    original.raman_terms[0],
                    label="additional",
                ),
            ),
        ),
    )


def test_default_forced_and_explicit_budget_precedence(modal):
    default = plan_julia_nodes(modal.problem, modal.request)
    assert default.n_nodes == 7 and default.plan.chunk == 7
    assert default.plan.source == "default"
    assert default.plan.budget == DEFAULT_NODE_BUDGET
    per_node = default.per_node_bytes
    selected = plan_julia_nodes(
        modal.problem, modal.request, ExecutionOptions(node_budget=4 * per_node)
    )
    assert selected.plan.chunk == 4 and selected.plan.source == "explicit"
    selected = plan_julia_nodes(
        modal.problem, modal.request, ExecutionOptions(node_budget=3 * per_node)
    )
    assert selected.plan.chunk == 3
    forced = plan_julia_nodes(
        modal.problem,
        modal.request,
        ExecutionOptions(node_chunk=100, node_budget=1),
    )
    assert forced.plan.chunk == 7 and forced.plan.requested == 100
    assert forced.plan.budget == 1 and forced.plan.source == "forced"
    assert forced.chunk_bytes > forced.plan.budget
    assert forced.trace_chunk == forced.plan.chunk


def test_balancing_uses_the_admitted_number_of_passes(modal):
    case = _modal_nodes(modal, 10)
    per_node = plan_julia_nodes(case.problem, case.request).per_node_bytes
    plan = plan_julia_nodes(
        case.problem, case.request, ExecutionOptions(node_budget=6 * per_node)
    )
    assert plan.plan.chunk == 5  # Two balanced passes instead of six plus four.
    assert plan.chunk_bytes <= plan.plan.budget


def test_budget_below_one_complete_node_refuses_without_mutation(modal):
    per_node = plan_julia_nodes(modal.problem, modal.request).per_node_bytes
    original = modal.problem.initial.field0.copy()
    for budget in (1, per_node - 1):
        with pytest.raises(BackendError, match="cannot hold one node"):
            plan_julia_nodes(
                modal.problem, modal.request, ExecutionOptions(node_budget=budget)
            )
    one = plan_julia_nodes(
        modal.problem, modal.request, ExecutionOptions(node_budget=per_node)
    )
    assert one.plan.chunk == 1 and one.chunk_bytes == per_node
    np.testing.assert_array_equal(modal.problem.initial.field0, original)


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("geometry", ["radial", "cartesian", "line"])
def test_route_defaults_and_named_disjoint_inventory(kind, geometry):
    case, _ = spatial_case(kind, geometry)
    result = plan_julia_nodes(case.problem, case.request)
    expected = (
        DEFAULT_NODE_BUDGET if geometry == "radial" else DEFAULT_SPATIAL_NODE_BUDGET
    )
    assert result.plan.budget == expected and result.plan.chunk == result.n_nodes
    assert len(_inventory(result)) == len(result.inventory)
    assert result.per_node_bytes == sum(row.per_node_bytes for row in result.inventory)
    assert result.fixed_workspace_bytes == sum(
        row.fixed_bytes for row in result.inventory
    )
    assert result.full_volume_workspace_bytes == sum(
        row.full_volume_bytes for row in result.inventory
    )
    assert result.per_node_bytes > 0
    assert result.opaque_fft_plans and any(
        "native" in item for item in result.exclusions
    )


@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_response_buffers_are_shared_across_components_and_channels(kind):
    case, _, _ = material_case(kind, "raman")
    first = plan_julia_nodes(case.problem, case.request)
    second = plan_julia_nodes(_plural_raman(case.problem), case.request)
    assert first.per_node_bytes == second.per_node_bytes
    kernel = _inventory(first)["source.raman.0.kernel_and_resonances"].fixed_bytes
    assert second.fixed_workspace_bytes - first.fixed_workspace_bytes == kernel
    assert ("source.analytic" in _inventory(first)) is (kind == "field")
    plasma, _, _ = material_case(kind, "plasma")
    source = plan_julia_nodes(plasma.problem, plasma.request)
    observed = _statistics(plasma, ("electron_density",))
    with_population = plan_julia_nodes(observed.problem, observed.request)
    n = plasma.problem.grids.n_to
    scalar = 8 if kind == "field" else 16
    assert with_population.per_node_bytes - source.per_node_bytes == n * (
        scalar + 5 * 8 + 1
    )
    assert "source.plasma_recurrence" in _inventory(source)


@pytest.mark.parametrize("geometry", ["radial", "cartesian", "line"])
@pytest.mark.parametrize("collar", [False, True])
def test_each_transverse_line_contributes_its_retained_strip_and_current(
    geometry, collar
):
    case, _, edge = edge_case(geometry, collar=collar)
    plain = plan_julia_nodes(case.problem, case.request.for_run(monitor=None))
    result = plan_julia_nodes(case.problem, case.request)
    expected = sum(
        16 * (2 * len(line.strip) + 3 * (line.onset is not None))
        for line in edge.reduction.lines
    )
    assert result.per_node_bytes - plain.per_node_bytes == expected
    assert result.trace_chunk == result.plan.chunk
    assert "transverse.column_and_node_readings" in _inventory(result)


def test_split_monitor_counts_empty_defined_buckets():
    case, grid, _ = material_case("envelope", "kerr")
    crossing = SplitCrossing(
        np.zeros((case.problem.shape[0], case.problem.shape[-1]), dtype=np.uint8),
        width=1e-12,
        fastest=1e-7,
    )
    monitor = BoundaryMonitor.build(grid, None, BoundaryCheck())
    responses = replace(
        case.problem.responses,
        temporal_absorber=np.zeros(grid.n_t),
        temporal_absorber_fine=np.zeros(grid.n_to),
        absorber_scheme="split_step",
    )
    problem = replace(case.problem, responses=responses, split_crossing=crossing)
    request = case.request.for_run(monitor=monitor)
    plan = plan_julia_nodes(problem, request)
    row = _inventory(plan)["observations.temporal_monitor"]
    assert row.fixed_bytes == (grid.n_t + 2 * SPLIT_RATE_BUCKETS) * 8


def test_plan_retains_only_frozen_scalar_inventory(modal):
    case = _modal_nodes(modal)
    case = Translation(case.problem, replace(case.request))
    problem_ref, request_ref = weakref.ref(case.problem), weakref.ref(case.request)
    plan = plan_julia_nodes(case.problem, case.request)
    del case
    gc.collect()
    assert problem_ref() is None
    assert request_ref() is None
    with pytest.raises(FrozenInstanceError):
        plan.__setattr__("n_nodes", 3)
    for entry in plan.inventory:
        assert all(
            isinstance(getattr(entry, field.name), (str, int))
            for field in fields(entry)
        )


def test_binary32_admission_is_named(modal):
    with pytest.raises(BackendError, match="binary32"):
        plan_julia_nodes(replace(modal.problem, precision=SINGLE), modal.request)


def test_unsupported_statistics_refuse_before_planning(modal):
    request = modal.request.for_run(
        stats=StatsSpec(names=("on_axis_intensity",), periodic=())
    )
    with pytest.raises(BackendError, match="unsupported statistics: on_axis_intensity"):
        plan_julia_nodes(modal.problem, request)


def test_vector_polarisation_refusal_is_named(modal):
    original = modal.problem.transform
    transform = replace(
        original,
        _n_pol=2,
        synthesis=np.repeat(original.synthesis, 2, axis=0),
        projection=np.repeat(original.projection, 2, axis=1),
        projection_coarse=np.repeat(original.projection_coarse, 2, axis=1),
    )
    problem = replace(
        modal.problem, route=replace(modal.problem.route, transform=transform)
    )
    with pytest.raises(BackendError, match="vector polarisation"):
        plan_julia_nodes(problem, modal.request)


def test_planning_never_imports_a_numerical_engine():
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys
        class BlockEngine(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                forbidden = {'juliacall', 'juliapkg', 'solprop_native'}
                if fullname.split('.')[0] in forbidden:
                    raise AssertionError('numerical engine imported: ' + fullname)
        sys.meta_path.insert(0, BlockEngine())
        from backends.julia.cases import cpu_case
        from solprop.backends.julia._node_plan import plan_julia_nodes
        case = cpu_case()
        assert plan_julia_nodes(case.problem, case.request).plan.chunk > 0
        """
    )
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root / "tests")))
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _material_responses(kind, response):
    kerr = KerrResponse(chi3=1e-22, thg=False)
    raman = response_for("SiO2", amplitude=1e-22, thg=False)
    plasma_type = PlasmaResponse if kind == "field" else EnvelopePlasmaResponse
    plasma = plasma_type("Ar", InstantaneousRate(scale=1e13), preionisation=0.02)
    return {
        "kerr": [kerr],
        "raman": [kerr, raman],
        "mixed": [kerr, raman, plasma],
    }[response]


def _memory_case(kind, geometry, response):
    """Build complete translated inputs rather than reconstruct Julia work counts."""
    original, grid, mode = material_case(kind, "kerr")
    responses = _material_responses(kind, response)
    pulse = GaussianPulse(wavelength=800e-9, fwhm=20e-15, energy=1e-9)
    kwargs = {}
    initial = pulse.build(grid)
    if geometry != "modal":
        space = {
            "radial": lambda: RadialGrid(4e-4, 8, collar_fraction=0.0),
            "cartesian": lambda: FreeGrid(4e-4, 6, 5e-4, 4, collar_fraction=0.0),
            "line": lambda: LineGrid(4e-4, 8, collar_fraction=0.0),
        }[geometry]()
        medium = SpatialMedium()
        initial = SpatialPulse(pulse, 100e-6, jones=(1,)).build(
            grid, space, medium=medium
        )
        kwargs = {"spatial_grid": space, "medium": medium}
    case = assemble(
        grid,
        mode if geometry == "modal" else None,
        responses,
        initial,
        density=1e24 if response == "mixed" else 1.0,
        controller=original.request.error_control.controller,
        dz_max=grid.length,
        dz_init=grid.length / 8,
        save_planes=[grid.length],
        stats_spec=StatsSpec(names=(), periodic=()),
        ionisation_field_range=(1e7, 1e11) if response == "mixed" else None,
        **kwargs,
    )
    case = Translation(case.problem, original.request)
    if response == "raman":
        # Two separately retained numerical Raman components share their node
        # convolution scratch. No propagation or new mixture physics is tested.
        case = Translation(_plural_raman(case.problem), case.request)
    if geometry == "modal":
        case = _modal_nodes(case)
    names = ("energy", "quadrature_error")
    if response == "mixed":
        names += ("species_ionisation", "electron_density", "peak_ionisation_rate")
    if geometry == "modal":
        names += ("peak_power",)
    else:
        names += (
            "peak_intensity",
            "on_axis_intensity",
            "beam_radius",
            "on_axis_spectrum",
        )
    return _statistics(case, names), grid, mode


def _retained(case, grid, mode, chunk):
    # The public ledger recursively visits owning Julia arrays once. It separately
    # labels FFT plan estimates, which cannot serve as retained-array evidence.
    from solprop.backends.julia.backend import JuliaBackend  # noqa: PLC0415

    owner = JuliaBackend(
        case.problem,
        case.request,
        grid=grid,
        modes=(mode,) if mode is not None else None,
        options=ExecutionOptions(
            node_chunk=chunk,
            backend_options={"setup_id": os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]},
        ),
    )
    try:
        with owner.prepare(case.problem, case.request).start(
            Start.entrance(case.problem)
        ) as session:
            return session.memory_ledger()
    finally:
        owner.close()


def _edge_memory_case(kind, geometry, collar):
    if not collar:
        return edge_case(geometry, kind=kind)
    original, grid = spatial_case(kind, geometry)
    space = {
        "radial": lambda: RadialGrid(4e-4, 32, collar_fraction=0.5),
        "cartesian": lambda: FreeGrid(4e-4, 32, 5e-4, 40, collar_fraction=0.5),
        "line": lambda: LineGrid(4e-4, 32, collar_fraction=0.5),
    }[geometry]()
    medium = SpatialMedium()
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=20e-15, energy=1e-9),
        100e-6,
        jones=(1,),
    )
    case = assemble(
        grid,
        None,
        [],
        pulse.build(grid, space, medium=medium),
        spatial_grid=space,
        medium=medium,
        controller=original.request.error_control.controller,
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length / 8,
        stats_spec=StatsSpec(names=(), periodic=()),
    )
    design = SpatialAbsorberDesign(
        SpatialAbsorber(kappa=3e5),
        np.zeros(case.problem.shape[0]),
        1.0,
        (1e6,),
        collar_rate=np.zeros(case.problem.shape[0]),
        alpha0=1e3,
        width=(1e-4,),
        kappa_p=(3e5,),
    )
    edge = TransverseEdge(
        space,
        design,
        0.1,
        10.0,
        grid.length,
        np.arange(grid.n_omega),
        2 * np.pi / 800e-9,
    )
    monitor = BoundaryMonitor.build(
        grid,
        None,
        BoundaryCheck(budget=1.0),
        columns=space.spectral_weights,
    )
    monitor.transverse = edge
    # Preparation retains noise and clean reductions even before numerical launch;
    # the bounded ledger must cover these optional resumed/fresh-noise buffers.
    noise = np.full(case.problem.shape, 1e-20 + 0j)
    problem = replace(
        case.problem,
        initial=Initial(case.problem.initial.field0, noise, "input", 0.0),
        flags=replace(case.problem.flags, has_noise=True),
    )
    return Translation(problem, original.request.for_run(monitor=monitor)), grid, edge


_MANAGED = pytest.mark.skipif(
    os.environ.get("SOLPROP_TEST_JULIA") != "1",
    reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
)


def _workspace_payload(ledger):
    prefixes = (
        "session.source.",
        "session.services.work.",
        "session.services.transverse.",
        "session.services.split.",
        "session.services.plane_values",
        "session.services.quadrature_work.",
    )
    return sum(
        size
        for name, size in ledger.items()
        if name.startswith(prefixes)
        and not name.endswith(("estimated_plan", ".chunks"))
        and ".execution." not in name
    )


def _check_retained_bound(plan, small, large, request):
    smaller, larger = _workspace_payload(small), _workspace_payload(large)
    # Chunk-descriptor containers shrink as chunks grow; they are object storage,
    # not numerical payload, and must not hide a missing growing numerical array.
    assert 0 < larger - smaller <= 2 * plan.per_node_bytes
    fixed = plan.fixed_workspace_bytes + plan.full_volume_workspace_bytes
    assert smaller <= fixed + plan.per_node_bytes
    assert larger <= fixed + 3 * plan.per_node_bytes
    request.node.user_properties.extend(
        (
            ("chunk_1_retained_bytes", smaller),
            ("chunk_3_retained_bytes", larger),
            ("per_node_bound_bytes", plan.per_node_bytes),
            ("fixed_full_volume_bound_bytes", fixed),
        )
    )


@_MANAGED
@pytest.mark.medium
@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("geometry", ["modal", "radial", "cartesian", "line"])
@pytest.mark.parametrize("response", ["kerr", "raman", "mixed"])
def test_retained_array_chunk_increment_is_within_the_inventory(
    kind, geometry, response, request
):
    case, grid, mode = _memory_case(kind, geometry, response)
    plan = plan_julia_nodes(case.problem, case.request)
    small = _retained(case, grid, mode if geometry == "modal" else None, 1)
    large = _retained(case, grid, mode if geometry == "modal" else None, 3)
    _check_retained_bound(plan, small, large, request)
    assert any("source" in name for name in small)
    assert any("services.work" in name for name in small)


@_MANAGED
@pytest.mark.medium
@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("geometry", ["radial", "cartesian", "line"])
@pytest.mark.parametrize("collar", [False, True])
def test_transverse_trace_increment_is_within_the_joint_node_bound(
    kind, geometry, collar, request
):
    case, grid, edge = _edge_memory_case(kind, geometry, collar)
    plan = plan_julia_nodes(case.problem, case.request)
    small, large = (_retained(case, grid, None, chunk) for chunk in (1, 3))
    _check_retained_bound(plan, small, large, request)
    assert any("services.transverse" in name for name in small)
    if collar:
        assert all(len(line.strip) >= 4 for line in edge.reduction.lines)
        assert any("transverse.lines" in name for name in small)


@_MANAGED
@pytest.mark.medium
@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_optional_modal_windows_axis_coarse_projection_and_split_fit(kind, request):
    original, grid, _ = split_case(kind, observers=("monitor",))
    mode = MarcatiliMode(50e-6, cladding=1.45, loss=False)
    responses = original.problem.responses
    assert responses.temporal_absorber is not None
    assert responses.temporal_absorber_fine is not None
    boundary = Absorber(
        np.zeros_like(grid.omega),
        responses.temporal_absorber,
        responses.temporal_absorber_fine,
        grid.active_mask,
        grid.length,
    )
    case = assemble(
        grid,
        mode,
        [],
        original.problem.initial.field0,
        controller=original.request.error_control.controller,
        save_planes=original.request.observation.save_planes,
        dz_init=grid.length / 8,
        dz_max=grid.length,
        absorber=boundary,
        absorber_scheme="split_step",
        stats_spec=StatsSpec(names=(), periodic=()),
    )
    case = Translation(case.problem, original.request)
    case = _modal_nodes(case)
    transform = replace(
        case.problem.transform,
        projection_coarse=case.problem.transform.projection * 0.5,
    )
    spec = StatsSpec(
        names=(
            "energy",
            "energy_window",
            "peak_power_window",
            "peak_intensity",
            "quadrature_error",
        ),
        periodic=(),
        energy_windows=np.array([[-np.inf, grid.omega0], [grid.omega0, np.inf]]),
    )
    case = Translation(
        replace(case.problem, route=replace(case.problem.route, transform=transform)),
        case.request.for_run(stats=spec, monitor=original.request.boundary.monitor),
    )
    plan = plan_julia_nodes(case.problem, case.request)
    small, large = (_retained(case, grid, mode, chunk) for chunk in (1, 3))
    _check_retained_bound(plan, small, large, request)
    assert any("work.windows" in name for name in small)
    assert any("work.axis" in name for name in small)
    assert any("services.split" in name for name in small)
    assert any("work.monitor_work" in name for name in small)
    assert any("source.coarse_projected" in name for name in small)
