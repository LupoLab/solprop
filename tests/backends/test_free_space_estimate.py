# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The estimate counts native's own rows on the Cartesian and line routes (issue #7).

The reference allocates its statistics and absorber buffers per call, so its count of a
free-space run fell furthest short of native's ledger when one chunk holds the whole
grid. Measured against native's ledger before it counted them: 1.41 at 128 x 128 x 80
with a 1 GiB node budget, 1.14 with 64 MiB.
These cases are the measured set of ``solprop.backends.memory`` on these routes.
**Construction only**: nothing is propagated. A ledger and its estimate are exact
accounting, compared to the byte; where a response keeps tables of its own that the
estimate does not name, the bound says how far that goes.
"""

import numpy as np
import pytest

from backends import test_free_ladders
from backends.test_dispersive_kerr_routes import ARGON, free_grid, free_problem
from solprop.backends.memory import (
    LOCAL_FLUX_STATISTICS,
    estimate_memory,
    free_space_memory,
    ledger_groups,
)
from solprop.backends.registry import select
from solprop.boundaries import absorber
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid
from solprop.monitor import DEFAULT_BOUNDARY_CHECK, BoundaryCheck
from solprop.nonlinear.responses import DispersiveKerrResponse, KerrResponse
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec, default_spec
from solprop.stepper import (
    FixedSchedule,
    LowStorageRungeKutta,
    Multistep,
    PIController,
    RungeKutta,
)
from solprop.translate.masks import AbsorberScheme

pytest.importorskip("solprop_native")

free_response = test_free_ladders.free_case

FREE = default_spec("free")
"""The statistics a free-space run records unless told otherwise."""

LENGTH = 1e-3
MIB = 2**20
GIB = 2**30


def beam(  # noqa: PLR0913 - one case's shape, budget and the route's choices
    nx: int,
    ny: int,
    window: float,
    budget: int,
    *,
    integrator: Multistep | RungeKutta | None = None,
    grid: type = EnvelopeGrid,
    collar: AbsorberScheme | None = None,
    n_pol: int = 1,
    stats: StatsSpec | None = None,
    monitor: BoundaryCheck | None = None,
) -> Simulation:
    """Return a Kerr beam on a Cartesian or line grid, from a broadcast placeholder."""
    limits = (LENGTH, 800e-9, (600e-9, 1.2e-6), window)
    time_grid = (
        EnvelopeGrid.from_limits(*limits, thg=False)
        if grid is EnvelopeGrid
        else FieldGrid.from_limits(*limits)
    )
    steps = 100 if collar else 4
    space = (
        FreeGrid(3e-4, nx, 3e-4, ny, collar_fraction=0.0)
        if ny > 1
        else LineGrid(3e-4, nx, collar_fraction=0.0)
    )
    settings = SimulationSettings(
        spatial_absorber=False,
        controller=PIController(rtol=1e-6, atol=1e-14),
        save_planes=[LENGTH],
        dz_max=LENGTH,
        dz_init=LENGTH / steps,
        schedule=FixedSchedule.uniform(
            0.0, LENGTH, LENGTH / steps, save_planes=[LENGTH]
        ),
        integrator=LowStorageRungeKutta() if integrator is None else integrator,
        backend="native",
        boundary_check=monitor,
        node_budget=budget,
        stats=stats,
        absorber=absorber(time_grid) if collar else None,
        absorber_scheme=collar,
    )
    shape = (nx * ny, n_pol, time_grid.n_omega)
    placeholder = np.broadcast_to(np.zeros((1, 1, 1), dtype=np.complex128), shape)
    return Simulation(
        time_grid,
        (),
        [KerrResponse(1e-22, thg=False)],
        placeholder,
        settings=settings,
        spatial_grid=space,
        medium=SpatialMedium(),
    )


def compared(simulation: Simulation) -> tuple[int, int]:
    """Return the estimate's base and the constructed backend's ledger's."""
    problem = simulation.translate()
    backend = select(
        "native",
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    try:
        ledger = ledger_groups(backend.memory_ledger())["base"]
    finally:
        backend.close()
    return estimate_memory(problem, "native").groups["base"], ledger


@pytest.mark.parametrize(
    ("nx", "ny", "window", "budget"),
    [
        pytest.param(64, 64, 144e-15, 64 * MIB, id="64x64x96 several chunks"),
        pytest.param(64, 64, 144e-15, GIB, id="64x64x96 one chunk"),
        pytest.param(
            128, 128, 120e-15, 64 * MIB, id="128x128 several", marks=pytest.mark.medium
        ),
        pytest.param(
            128, 128, 120e-15, GIB, id="128x128 one chunk", marks=pytest.mark.medium
        ),
        pytest.param(
            192,
            192,
            450e-15,
            64 * MIB,
            id="192x192x288 several",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            192, 192, 450e-15, GIB, id="192x192x288 one chunk", marks=pytest.mark.slow
        ),
    ],
)
def test_the_estimate_is_the_low_storage_ledger_to_the_byte(nx, ny, window, budget):
    estimate, ledger = compared(beam(nx, ny, window, budget))
    assert estimate == ledger


@pytest.mark.parametrize(
    "keywords",
    [
        pytest.param({"integrator": RungeKutta()}, id="runge-kutta"),
        pytest.param({"collar": "nonlinear"}, id="in-stage collar"),
        pytest.param({"collar": "split_step"}, id="split collar"),
        pytest.param({"grid": FieldGrid, "window": 40e-15}, id="field"),
        pytest.param(
            {"grid": FieldGrid, "window": 40e-15, "budget": 16 * MIB},
            id="field several chunks",
        ),
        pytest.param({"n_pol": 2}, id="two polarisations"),
    ],
)
@pytest.mark.parametrize("stats", [None, FREE], ids=["no statistics", "default free"])
def test_the_estimate_holds_across_the_route_s_marches_and_terms(keywords, stats):
    options = {"window": 144e-15, "budget": GIB} | keywords
    window, budget = options.pop("window"), options.pop("budget")
    estimate, ledger = compared(beam(64, 64, window, budget, stats=stats, **options))
    assert estimate == ledger


@pytest.mark.parametrize("budget", [16 * MIB, GIB], ids=["several chunks", "one chunk"])
@pytest.mark.parametrize(
    "stats",
    [
        pytest.param(FREE, id="default free"),
        *(
            pytest.param(StatsSpec(names=(name,)), id=name)
            for name in sorted(LOCAL_FLUX_STATISTICS)
        ),
        pytest.param(StatsSpec(names=("energy", "on_axis_spectrum")), id="no closure"),
    ],
)
def test_the_estimate_counts_what_each_statistic_asks_for(stats, budget):
    """A local flux statistic holds the magnetic closure's rows; the others none."""
    simulation = beam(64, 64, 144e-15, budget, stats=stats)
    problem = simulation.translate()
    rows = free_space_memory(problem, "native")
    closure = bool(set(stats.names) & LOCAL_FLUX_STATISTICS)
    assert ("statistics_weighted_time" in rows) == closure
    assert ("statistics_flux_magnitude" in rows) == closure
    estimate, ledger = compared(simulation)
    assert estimate == ledger


def test_a_run_adds_its_monitor_s_weights_and_the_far_field_s_tables_alone():
    """The estimate is of what is constructed; these two arrive when a run starts."""
    simulation = beam(32, 32, 144e-15, GIB, stats=FREE, monitor=DEFAULT_BOUNDARY_CHECK)
    problem = simulation.translate()
    field = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-9), 4e-5
    ).build(
        simulation.grid,
        FreeGrid(3e-4, 32, 3e-4, 32, collar_fraction=0.0),
        medium=SpatialMedium(),
    )
    estimate = estimate_memory(problem, "native").groups["base"]
    with simulation.runner() as runner:
        before = runner.backend.memory_ledger()
        assert ledger_groups(before)["base"] == estimate
        runner.run(field, progress="none")
        ledger = runner.backend.memory_ledger()
    weights = ledger["monitor_weights"]
    tables = ledger["far_field_tables"] - before["far_field_tables"]
    assert 0 < weights + tables < 0.01 * estimate
    assert ledger_groups(ledger)["base"] == estimate + weights + tables


def test_the_line_route_is_counted_the_same_way():
    for stats in (None, FREE):
        estimate, ledger = compared(beam(128, 1, 120e-15, GIB, stats=stats))
        assert estimate == ledger


def test_the_adams_march_differs_by_its_norms_alone():
    """The trial's norms are held in the march's own count, and are not field-sized."""
    estimate, ledger = compared(beam(64, 64, 144e-15, 64 * MIB, integrator=Multistep()))
    assert ledger - estimate == 24


def test_a_response_of_its_own_stays_close(free_response):
    """Kerr is exact; Raman and plasma keep small rows of their own, not all named.

    Measured: Raman's kernels and rates fall 0.18 % short, and the reference's plasma
    node rows, which native forms in its own scratch, put plasma 1.1 % over. Under-count
    is what could admit a run that then exceeds its budget, so it has the tighter bound.
    """
    family, grid, _, problem = free_response
    backend = select("native", problem, grid=grid)
    try:
        ledger = ledger_groups(backend.memory_ledger())["base"]
    finally:
        backend.close()
    estimate = estimate_memory(problem, "native").groups["base"]
    if family.endswith("kerr"):
        assert estimate == ledger
    assert -0.02 * ledger <= ledger - estimate <= 0.005 * ledger


@pytest.mark.parametrize("evaluator", ["polynomial", "symmetric"])
@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_a_dispersive_beam_keeps_its_node_rows_apart(evaluator, kind):
    """The dispersive pass needs its nodes apart, so they do not alias the chunk's."""
    grid = free_grid(kind)
    problem = free_problem(DispersiveKerrResponse(ARGON, evaluator, thg=False), grid)
    rows = free_space_memory(problem, "native")
    assert rows["node_field"] > 0
    assert "node_polarisation" not in rows
    backend = select("native", problem, grid=grid)
    try:
        ledger = ledger_groups(backend.memory_ledger())["base"]
    finally:
        backend.close()
    estimate = estimate_memory(problem, "native").groups["base"]
    # The response's multipliers are not counted, by the estimate's own statement.
    assert -0.02 * ledger <= ledger - estimate <= 0.005 * ledger


def test_only_native_on_free_space_has_rows_of_its_own():
    simulation = beam(64, 64, 144e-15, GIB)
    problem = simulation.translate()
    assert free_space_memory(problem, "numpy_ref") == {}
    assert free_space_memory(problem, "native")["statistics_spectrum"] > 0
    capillary = Simulation.from_gnlse(
        0.11,
        0.01,
        [0.0, 0.0, -1.183e-26],
        wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=9.8,
        shape="sech",
    )
    assert free_space_memory(capillary.translate(), "native") == {}
