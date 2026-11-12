# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The node chunk is the backend's execution choice, planned at preparation (SEP 47).

Translation records no chunk: each backend plans it from the problem and its
:class:`~solprop.backends.session.ExecutionOptions` (section 6.5), and the plan is
execution metadata, outside every identity (section 4.3). The move changed no chunk:
``tests/data/node_chunks.json`` holds the chunks the translation chose at commit
655e59ca, for every route and response admission at twenty budgets, the default
among them, and the plan reproduces each, refusals included.
"""

import dataclasses
import functools
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import vector_raman_cases
from backends import host_cases_envelope_plasma as envelope_cases
from backends import test_native_edge_collar as collar
from backends import test_radial_nonlinear as radial_cases
from driver.test_prop_capillary import build as capillary
from solprop.backends.node_chunk import (
    BUFFERS_PER_NODE,
    DEFAULT_NODE_BUDGET,
    DEFAULT_SPATIAL_NODE_BUDGET,
    NodeChunkPlan,
    _even_chunk,
    default_node_budget,
    general_planned_chunk,
    plan_node_chunk,
)
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.session import ExecutionOptions, positive_count
from solprop.driver import Simulation, SimulationSettings
from solprop.driver.interface import prop_capillary, prop_waveguide
from solprop.exceptions import BackendError, DriverError
from solprop.grid import FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.nonlinear.raman import response_for
from solprop.nonlinear.responses import DispersiveKerrResponse, PlasmaResponse
from solprop.physdata.nonlinear_data import OptmatKerr
from solprop.profiles import Constant
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController
from solprop.translate.ionisation import build_rate_table
from solprop.translate.problem import assemble
from solprop.translate.stage import terms

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.SolpropWarning"),
]

RECORD = json.loads(
    (Path(__file__).parents[1] / "data" / "node_chunks.json").read_text()
)
"""The translation's chunks before SEP 47, by case and budget."""


def _dispersive_modal(evaluator, raman):
    """A fused modal dispersive response, alone or beside a Raman term."""
    grid = FieldGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 400e-15)
    mode = MarcatiliMode(125e-6, "Ar", 1.0, cladding=1.45)
    declared = [
        DispersiveKerrResponse(OptmatKerr("Ar").dispersive_model(), evaluator),
        *(
            [response_for("SiO2", amplitude=1e-22, fraction=0.18, thg=True)]
            if raman
            else []
        ),
    ]
    start = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6).build(grid)
    return assemble(
        grid,
        [mode],
        declared,
        start,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[0.1],
        dz_max=0.01,
        dz_init=1e-3,
        quadrature=transverse_quadrature([mode], n_r=8),
        density=Constant(2.5e25),
    ).problem


def _dispersive_free(evaluator):
    """The dispersive response on Cartesian columns."""
    grid = FieldGrid.from_limits(1e-4, 800e-9, (400e-9, 2e-6), 100e-15)
    space = FreeGrid(3e-4, 9, 4e-4, 12, collar_fraction=0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-6), 8e-5
    )
    return assemble(
        grid,
        None,
        [DispersiveKerrResponse(OptmatKerr("Ar").dispersive_model(), evaluator)],
        pulse.build(grid, space, medium=SpatialMedium()),
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1e-4],
        dz_max=1e-4,
        dz_init=1e-5,
        density=Constant(2.5e25),
    ).problem


def _held_plasma(kind):
    """Field plasma on a spatial route: the held-current admission."""
    grid = FieldGrid.from_limits(radial_cases.LENGTH, 800e-9, (600e-9, 1.2e-6), 200e-15)
    space = {
        "radial": RadialGrid(6e-4, 16, collar_fraction=0.0),
        "free": FreeGrid(3e-4, 9, 4e-4, 12, collar_fraction=0),
        "line": LineGrid(4e-4, 24, collar_fraction=0.0),
    }[kind]
    pulse = SpatialPulse(
        GaussianPulse(
            wavelength=800e-9, fwhm=30e-15, energy=1e-3 if kind == "line" else 1e-7
        ),
        80e-6,
        jones=(1, 0.5j),
    )
    rate = build_rate_table(radial_cases.QuadraticRate(), (1e3, 1e10))
    return radial_cases.make_problem(
        grid,
        space,
        pulse.build(grid, space, medium=SpatialMedium()),
        [PlasmaResponse("Ar", rate)],
        steps=2,
        density=1e22,
    ).problem


def _builders():
    """Return each recorded case's problem builder, by name."""
    mixture = functools.cache(envelope_cases.build_envelope_mixture)
    builders = {
        "capillary-1-mode": lambda: capillary().translate(),
        "capillary-3-modes": lambda: capillary(modes=3).translate(),
        "capillary-3-modes-plasma": lambda: capillary(modes=3, plasma=True).translate(),
        "capillary-n2-raman": lambda: capillary(
            gas="N2", modes=2, raman=True
        ).translate(),
    }
    for route in ("radial", "line", "free", "planar"):
        builders[f"collar-{route}"] = functools.partial(
            lambda route: collar._case(route).translate(), route
        )
        builders[f"collar-kerr-{route}"] = functools.partial(
            lambda route: collar._case(route, kerr=True).translate(), route
        )
    for kind in ("radial", "free", "line"):
        builders[f"plasma-{kind}"] = functools.partial(_held_plasma, kind)
    for geometry in ("modal", "radial", "free"):
        for n_pol in (1, 2):
            builders[f"envelope-{geometry}-{n_pol}"] = functools.partial(
                lambda geometry, n_pol: envelope_cases.geometry_case(
                    mixture(), geometry, n_pol
                )[0],
                geometry,
                n_pol,
            )
    for route in ("axis", "rows", "scalar"):
        for family in ("envelope", "field_real"):
            builders[f"raman-{route}-{family}"] = functools.partial(
                lambda route, family: vector_raman_cases.vector_problem(
                    family=family, route=route
                )[0],
                route,
                family,
            )
    for evaluator in ("symmetric", "polynomial"):
        for raman in (False, True):
            builders[f"dispersive-modal-{evaluator}-{raman}"] = functools.partial(
                _dispersive_modal, evaluator, raman
            )
        builders[f"dispersive-free-{evaluator}"] = functools.partial(
            _dispersive_free, evaluator
        )
    return builders


BUILDERS = _builders()


def test_every_recorded_case_has_a_builder():
    assert set(BUILDERS) == set(RECORD["cases"])


@pytest.mark.parametrize(
    "name",
    [
        # The envelope plasma cases build their two-population rate tables.
        pytest.param(name, marks=pytest.mark.medium)
        if name.startswith("envelope")
        else name
        for name in sorted(RECORD["cases"])
    ],
)
def test_the_plan_is_the_translation_s_former_chunk(name):
    """At the default budget and nineteen explicit ones, each chunk or refusal."""
    problem = BUILDERS[name]()
    for budget, expected in zip(RECORD["budgets"], RECORD["cases"][name], strict=True):
        options = ExecutionOptions(node_budget=budget)
        if isinstance(expected, dict):
            with pytest.raises(BackendError) as refused:
                plan_node_chunk(problem, options)
            assert str(refused.value).startswith(expected["error"]), (name, budget)
            continue
        plan = plan_node_chunk(problem, options)
        assert plan.chunk == expected, (name, budget)
        assert plan.source == ("default" if budget is None else "explicit")
        assert plan.budget == (
            default_node_budget(problem.transform) if budget is None else budget
        )


# --------------------------------------------------------------- the rule's pieces


@pytest.fixture(scope="module")
def multimode_run():
    """A three-mode capillary on nine nodes, with no response admission."""
    simulation = capillary(modes=3, plasma=False)
    return simulation.translate(), simulation._run_request()


@pytest.fixture(scope="module")
def multimode(multimode_run):
    """The three-mode capillary's problem alone."""
    return multimode_run[0]


def test_a_small_problem_is_not_chunked(multimode):
    assert plan_node_chunk(multimode).chunk == multimode.transform.n_nodes


@pytest.mark.parametrize(("nodes", "chunk"), [(3, 3), (4, 3), (5, 5), (8, 5)])
def test_the_chunk_follows_the_memory_budget(multimode, nodes, chunk):
    """The budget's nodes, spread evenly over the passes they need: 9 = 3 + 3 + 3."""
    per_node = BUFFERS_PER_NODE * multimode.transform.n_pol * multimode.grids.n_to * 16
    options = ExecutionOptions(node_budget=nodes * per_node)
    assert plan_node_chunk(multimode, options).chunk == chunk


@pytest.mark.parametrize(
    ("count", "limit", "chunk"), [(33, 32, 17), (33, 33, 33), (100, 30, 25)]
)
def test_an_even_chunk_covers_the_count_in_the_fewest_passes(count, limit, chunk):
    """33 items under a limit of 32 take two passes, so 17 each, not 32 and 1."""
    assert _even_chunk(count, limit) == chunk


def test_the_chunk_never_falls_below_one(multimode):
    assert plan_node_chunk(multimode, ExecutionOptions(node_budget=1)).chunk == 1


@pytest.mark.parametrize("budget", [0, -1, True])
def test_a_budget_that_is_not_a_positive_integer_is_refused(budget):
    with pytest.raises(BackendError, match="positive number of bytes"):
        ExecutionOptions(node_budget=budget)
    with pytest.raises(BackendError, match="node_chunk must be a positive"):
        ExecutionOptions(node_chunk=budget)


@pytest.mark.parametrize("value", [2.0**20, "1048576", None])
def test_a_count_that_is_not_integral_is_refused_by_name(value):
    """Only an integral value (``operator.index``) is a count: a float, even a whole
    one, or a numeric string is refused naming the option. The annotations refuse
    them first under a runtime type check; the count refuses them without one."""
    with pytest.raises(BackendError, match="node_budget must be a positive number of"):
        positive_count("node_budget", value)
    with pytest.raises(BackendError, match="node_chunk must be a positive number, got"):
        positive_count("node_chunk", value)


@pytest.mark.parametrize("integral", [np.int64(2**20), np.uint32(2**20), 2**20])
def test_any_integral_budget_is_held_as_an_int(integral):
    """A NumPy integer, as a scan axis gives one, is a budget like any other."""
    options = ExecutionOptions(node_budget=integral, node_chunk=integral)
    assert type(options.node_budget) is int and options.node_budget == 2**20
    assert type(options.node_chunk) is int and options.node_chunk == 2**20
    settings = SimulationSettings(
        save_planes=[1.0], dz_max=0.1, dz_init=0.01, node_budget=integral
    )
    assert type(settings.node_budget) is int and settings.node_budget == 2**20
    assert settings.execution_options() == ExecutionOptions(node_budget=2**20)
    with pytest.raises(BackendError, match="positive number of bytes"):
        ExecutionOptions(node_budget=np.int64(0))


def test_the_settings_refuse_a_budget_before_translation():
    """Refused when the settings are made, not after the output opened (SEP 47)."""
    with pytest.raises(DriverError, match="node_budget must be a positive"):
        SimulationSettings(save_planes=[1.0], dz_max=0.1, dz_init=0.01, node_budget=0)


def test_the_constructors_leave_the_budget_to_the_backend():
    """No budget given is the route's default, recorded so; a given one is explicit."""
    simulation = capillary(modes=3)
    assert simulation.settings.node_budget is None
    problem = simulation.translate()
    plan = plan_node_chunk(problem, simulation.settings.execution_options())
    assert (plan.source, plan.budget) == ("default", DEFAULT_NODE_BUDGET)
    explicit = capillary(modes=3, node_budget=DEFAULT_NODE_BUDGET)
    same = plan_node_chunk(problem, explicit.settings.execution_options())
    assert (same.source, same.chunk) == ("explicit", plan.chunk)
    signature = inspect.signature
    for constructor in (
        Simulation.from_capillary,
        Simulation.from_waveguide,
        prop_capillary,
        prop_waveguide,
    ):
        assert signature(constructor).parameters["node_budget"].default is None


def test_the_default_budgets_are_stated_in_bytes():
    assert DEFAULT_NODE_BUDGET == 64 * 2**20
    assert DEFAULT_SPATIAL_NODE_BUDGET == 2**30


def test_a_cartesian_column_charges_twenty_buffers():
    """The conservative inventory stays (#349): a budget buys the chunk it did."""
    problem = BUILDERS["collar-free"]()
    column = 20 * 16 * problem.shape[1] * problem.grids.n_to
    assert plan_node_chunk(problem, ExecutionOptions(node_budget=7 * column)).chunk == 7
    with pytest.raises(BackendError, match="Cartesian node_budget cannot hold"):
        plan_node_chunk(problem, ExecutionOptions(node_budget=column - 1))


def test_a_planar_column_too_large_for_the_budget_is_refused():
    problem = BUILDERS["collar-planar"]()
    with pytest.raises(BackendError, match="planar node_budget cannot hold one"):
        plan_node_chunk(problem, ExecutionOptions(node_budget=16))


def test_an_exact_chunk_overrides_the_budget_and_is_bounded_by_the_nodes(multimode):
    n_nodes = multimode.transform.n_nodes
    plan = plan_node_chunk(multimode, ExecutionOptions(node_budget=1, node_chunk=4))
    assert (plan.chunk, plan.source, plan.requested) == (4, "forced", 4)
    plan = plan_node_chunk(multimode, ExecutionOptions(node_chunk=10 * n_nodes))
    assert plan.chunk == n_nodes and plan.requested == 10 * n_nodes


def test_the_plan_is_recorded_as_execution_metadata(multimode, multimode_run):
    """The backend's record names the chunk, its budget and what chose it."""
    budget = 2**20
    backend = NumpyRefBackend(
        *multimode_run, options=ExecutionOptions(node_budget=budget)
    )
    try:
        record = backend.metadata()
        expected = plan_node_chunk(multimode, ExecutionOptions(node_budget=budget))
        assert backend.chunk_plan == expected
        assert backend.node_chunk == expected.chunk
    finally:
        backend.close()
    assert record["node_chunk"] == expected.chunk
    assert record["node_budget"] == budget
    assert record["node_chunk_source"] == "explicit"
    assert NodeChunkPlan(3, 4, "forced", 5).record() == {
        "node_chunk": 3,
        "node_budget": 4,
        "node_chunk_source": "forced",
    }


def test_an_explicit_budget_wins_over_the_automatic_choice(multimode, multimode_run):
    automatic = NumpyRefBackend(*multimode_run)
    explicit = NumpyRefBackend(*multimode_run, options=ExecutionOptions(node_budget=1))
    try:
        assert automatic.node_chunk == multimode.transform.n_nodes
        assert explicit.node_chunk == 1
        assert automatic.memory_ledger()["node_field"] == (
            multimode.transform.n_nodes * explicit.memory_ledger()["node_field"]
        )
    finally:
        automatic.close()
        explicit.close()


# ------------------------------------------------- identity: the problem is unchanged


def test_the_node_budget_leaves_the_problem_and_its_resume_key_alone():
    """Scientific identity (section 4.3): no chunk in the problem or the record."""
    plain = capillary(modes=3)
    budgeted = capillary(modes=3, node_budget=2**19)
    assert budgeted.settings.node_budget == 2**19
    first, second = plain.translate(), budgeted.translate()
    names = {field.name for field in dataclasses.fields(first.transform)}
    assert "node_chunk" not in names
    np.testing.assert_array_equal(first.transform.synthesis, second.transform.synthesis)
    key = plain.metadata().configuration_hash()
    assert budgeted.metadata().configuration_hash() == key
    assert (
        plan_node_chunk(first, plain.settings.execution_options()).chunk
        != plan_node_chunk(second, budgeted.settings.execution_options()).chunk
    )


def test_the_node_budget_may_change_after_translation():
    """An execution setting: a translated run takes a new budget without refusal."""
    simulation = capillary(modes=3)
    problem = simulation.translate()
    simulation.settings = dataclasses.replace(simulation.settings, node_budget=1)
    assert simulation.translate() is problem
    assert simulation.settings.execution_options() == ExecutionOptions(node_budget=1)


def test_translation_starts_no_native_runtime(tmp_path):
    """No device runtime starts before a scan forks: translation imports no backend."""
    script = tmp_path / "probe.py"
    script.write_text(
        "import sys\n"
        "from solprop.driver import Simulation\n"
        "Simulation.from_capillary(\n"
        "    50e-6, 0.01, 'He', 1.0, wavelength=800e-9,\n"
        "    wavelength_limits=(200e-9, 2e-6), trange=100e-15, fwhm=10e-15,\n"
        "    energy=20e-6, n_r=8, modes=3, noise=None, node_budget=2**16,\n"
        ").translate()\n"
        "print('solprop_native._core' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


# ------------------------------------------------------------- the native lowering


@pytest.fixture(scope="module")
def native_backend():
    return pytest.importorskip("solprop_native.backend")


def test_native_lowering_keeps_the_host_chunk_and_whole_planar_rows(native_backend):
    lower = native_backend.native_node_chunk
    for name in ("collar-free", "collar-radial", "plasma-line", "capillary-3-modes"):
        problem = BUILDERS[name]()
        for budget in (None, 2**21, 2**23):
            plan = plan_node_chunk(problem, ExecutionOptions(node_budget=budget))
            assert lower(problem, plan, device=False) == plan.chunk
            assert lower(problem, plan, device=True) == plan.chunk
    planar = BUILDERS["collar-planar"]()
    geometry = planar.transform.geometry
    assert geometry is not None
    n_x, n_y = geometry.shape
    for chunk, rows in ((1, 1), (n_x + 1, 1), (3 * n_x - 1, 2), (10 * n_x * n_y, n_y)):
        plan = plan_node_chunk(planar, ExecutionOptions(node_chunk=chunk))
        assert lower(planar, plan, device=False) == rows * n_x


def test_the_device_rule_applies_to_an_automatic_plan_alone(
    native_backend, monkeypatch
):
    """A later qualified device default changes `device_node_chunk` and nothing else."""
    problem = BUILDERS["collar-free"]()
    calls = []

    def larger(problem, planned, free):
        del free
        calls.append(planned)
        return problem.transform.n_nodes

    monkeypatch.setattr(native_backend, "device_node_chunk", larger)
    lower = native_backend.native_node_chunk
    automatic = plan_node_chunk(problem, ExecutionOptions())
    explicit = plan_node_chunk(problem, ExecutionOptions(node_budget=2**18))
    forced = plan_node_chunk(problem, ExecutionOptions(node_chunk=3))
    assert explicit.chunk < problem.transform.n_nodes
    assert lower(problem, automatic, device=True) == problem.transform.n_nodes
    assert lower(problem, explicit, device=True) == explicit.chunk
    assert lower(problem, forced, device=True) == 3
    assert lower(problem, automatic, device=False) == automatic.chunk
    assert calls == [automatic.chunk]
    # A host build knows no free device bytes: the host plan's chunk.
    monkeypatch.undo()
    assert native_backend.device_node_chunk(problem, 7, None) == 7


def test_the_device_rule_budgets_a_share_of_the_free_device_bytes(native_backend):
    """One chunk where a quarter of the free bytes holds it, else the largest that
    does, never below the host plan's chunk (solprop issue #361)."""
    rule = native_backend.device_node_chunk
    share = native_backend.DEVICE_BUDGET_SHARE
    problem = BUILDERS["collar-free"]()
    n_nodes = problem.transform.n_nodes
    narrow = plan_node_chunk(problem, ExecutionOptions(node_budget=2**18)).chunk
    assert 1 < narrow < n_nodes
    assert rule(problem, 1, 2**50) == n_nodes
    assert rule(problem, 1, share * 2**18) == narrow
    assert rule(problem, narrow + 1, share * 2**18) == narrow + 1
    # A share that cannot hold one node leaves the host plan's chunk.
    assert rule(problem, 5, share) == 5
    assert rule(problem, 5, None) == 5


def test_the_native_backend_executes_and_records_its_lowered_chunk(native_backend):
    pytest.importorskip("solprop_native")
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        NativeBackend,
    )

    simulation = collar._case("planar")
    problem, run_request = simulation.translate(), simulation._run_request()
    geometry = problem.transform.geometry
    assert geometry is not None
    n_x = geometry.shape[0]
    options = ExecutionOptions(node_chunk=2 * n_x + 1)
    backend = NativeBackend(problem, run_request, options=options)
    try:
        record = backend.metadata()
        assert backend.node_chunk == 2 * n_x
    finally:
        backend.close()
    assert record["node_chunk"] == 2 * n_x
    assert record["node_chunk_source"] == "forced"
    assert record["node_chunk_rule"] in {"host", "device"}


def test_a_general_guide_takes_the_plan_s_budget(native_backend):
    """The general route's own chunk at the plan's budget, the same on both backends."""
    from backends.dispersive_route_cases import (  # noqa: PLC0415
        ARGON_FILL,
        general_problem,
        strand,
    )
    from solprop.backends.numpy_ref.backend import _general_maps  # noqa: PLC0415

    problem, run_request, _ = general_problem(strand(ARGON_FILL), "symmetric", "real")
    automatic = plan_node_chunk(problem)
    assert automatic.chunk == 1 and automatic.budget == DEFAULT_NODE_BUDGET
    small = plan_node_chunk(problem, ExecutionOptions(node_budget=1))
    forced = plan_node_chunk(problem, ExecutionOptions(node_chunk=2))
    maps = _general_maps(problem, small)
    assert maps is not None and maps.node_chunk == general_planned_chunk(problem, small)
    assert maps.node_chunk == 1
    assert general_planned_chunk(problem, forced) == 2
    whole = _general_maps(problem, automatic)
    assert whole is not None and whole.node_chunk > 1
    payload = native_backend._lower(
        problem,
        run_request,
        native_backend._LowerPlan(terms=terms(problem), general_chunk=2),
    )
    assert payload.general.node_chunk == 2
    assert (
        native_backend._lower(
            problem, run_request, native_backend._LowerPlan(terms=terms(problem))
        ).general.node_chunk
        == whole.node_chunk
    )


def test_an_unset_budget_is_the_automatic_choice():
    settings = SimulationSettings(save_planes=[1.0], dz_max=0.1, dz_init=0.01)
    assert settings.node_budget is None
    assert settings.execution_options() == ExecutionOptions()
