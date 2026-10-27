# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 6's dispersive response on numpy_ref's free-space, planar and general routes.

Slice 1b of the SEP's phase 1: the Cartesian free-space pipeline (and the line grid
that shares it) and the hybrid planar pipeline evaluate ``responses.dispersive_kerr``
node by node, and the general modal route evaluates a gas region's channels at the
nodes of that region. Per route:

- one nonlinear evaluation, caught where the route calls it, equals the reference
  :meth:`~solprop.nonlinear.responses.DispersiveKerrResponse.evaluate` at a node, bit
  for bit;
- with the dispersion switched off, the whole stage derivative equals the constant
  cubic's on the same route;
- a short lossless run changes the route's own energy measure by what the constant
  cubic's run changes it: none of these routes' measures is conserved exactly by the
  constant cubic itself (below), and the dispersive response adds no work to that.

The compiled backends run the leaf on these routes against this one
(``test_numba_dispersive_routes.py``, ``test_native_dispersive_routes.py``).

The general modal route also takes a guide with a constant solid region beside a gas
one, the carrier policy's constant cubic for a gas region, and the record of both.
"""

import dataclasses
import logging
import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from backends.matched import dispersive_response_bound
from bounds import assert_within
from modal_provider_cases import GENERAL_Z_END, WAVELENGTH, general_grid
from solprop.backends.numpy_ref import NumpyRefBackend, general_modal, passes
from solprop.backends.numpy_ref.backend import _general_maps
from solprop.boundaries import Absorber
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import (
    BackendError,
    ModeError,
    SolpropWarning,
    TranslationError,
)
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, TimeGrid
from solprop.modes import PlanarGrid, SlabMode, StepIndexMode
from solprop.modes.guide import ModalGuide
from solprop.modes.material import (
    GasKerr,
    IndexMaterial,
    MaterialKerr,
    MaterialRegion,
)
from solprop.nonlinear.raman import SilicaTensorResponse
from solprop.nonlinear.responses import (
    DispersiveKerrResponse,
    DispersiveWeights,
    KerrResponse,
    dispersive_cubic,
    mixture_filter,
)
from solprop.output import MemoryOutput
from solprop.physdata import MaterialSelection, Mixture, species
from solprop.physdata.nonlinear_data import DispersiveKerrModel, OptmatKerr
from solprop.physdata.species import Species
from solprop.profiles import Linear
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec, default_spec
from solprop.stepper import FixedSchedule, PIController, RungeKutta
from solprop.translate.general import (
    _log_gas_regions,
    general_modal_route,
    modal_identity,
)
from solprop.translate.modal import DEFAULT_NODE_BUDGET
from solprop.translate.problem import assemble
from solprop.translate.responses import KerrForm, dispersive_terms

pytestmark = [
    # The fine grids reach beyond argon's linear fit; the notice is not the subject.
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
]

ARGON = OptmatKerr("Ar").dispersive_model()
FLAT = DispersiveKerrModel(
    "Ar", ARGON.gamma3_static, 0.0, 0.0, 0.0, ARGON.leg_bound, None, None,
    ARGON.material_records,
)  # fmt: skip
"""Argon's model without its dispersion: the constant cubic of ``gamma3_0``."""

DENSITY = 4.9e25
"""Number density of the free-space runs' nonlinear medium, m^-3: 2 bar of argon."""

POLICIES = {"polynomial": "gas-polynomial", "symmetric": "symmetric-kerr"}

CARRIER = MaterialSelection()


def carrier_gamma3(wavelength: float) -> float:
    """Return argon's Kerr coefficient at a carrier, m^5/V^2: the carrier policy's."""
    value = species("Ar").kerr_coefficient(wavelength)
    assert value is not None
    return value


def caught(monkeypatch, module: object, name: str) -> list[tuple[Any, Any, Any]]:
    """Record every call of a module function as ``(args, kwargs, result)``."""
    calls: list[tuple[Any, Any, Any]] = []
    original: Callable[..., Any] = getattr(module, name)

    def recording(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        calls.append((args, kwargs, result))
        return result

    monkeypatch.setattr(module, name, recording)
    return calls


def unit_collar(problem):
    """Return a problem whose temporal collar is one everywhere."""
    return dataclasses.replace(
        problem,
        grids=dataclasses.replace(
            problem.grids, t_window_fine=np.ones(problem.grids.n_to)
        ),
    )


def stage_derivative(problem, grid):
    """Return ``numpy_ref``'s stage derivative at the launch."""
    backend = NumpyRefBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend._hoist(0.0)
        return backend._nonlinear(
            backend._require_state(),
            backend._stage_scalars(0.0, 0.0),
            np.ones(problem.shape, complex),
        )
    finally:
        backend.close()


def assert_same_derivative(got, expected):
    """Hold two stage derivatives to the few FFT round trips the leaf adds.

    The dispersive cubic transforms the field and the products where the constant
    cubic multiplies samples; 1e-12 of the largest bin bounds a few round trips of a
    few thousand points. Measured below 3e-16 on every route.
    """
    scale = float(np.abs(expected).max())
    assert scale > 0.0
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12 * scale)


def lossless(grid: TimeGrid) -> Absorber:
    """Return a collar that absorbs nothing, spectrally or in time."""
    return Absorber(
        np.zeros(grid.n_omega),
        np.zeros(grid.n_t),
        np.zeros(grid.n_to),
        grid.active_mask,
        1.0,
    )


def relative_change(output: MemoryOutput, name: str) -> float:
    """Return the relative change of a summed statistic from first to last plane."""
    planes = output.planes()
    first, last = (
        float(np.sum(plane.stats[name])) for plane in planes[:: len(planes) - 1]
    )
    return last / first - 1.0


def assert_no_added_work(dispersive: float, constant: float):
    """The dispersive run's energy change is the constant cubic's, the route's own.

    Each route's energy measure departs from the quantity its lossless flow conserves
    by an amount set by the nonlinear exchange (the free and planar measures are
    paraxial, the general route's metric is the identity to its Gram defect): the
    constant cubic's run shows it, independent of the step (checked when the cases
    were set up) and in proportion to the pulse energy. A response that did net work
    would move the change by its own share; the dispersive coefficients move the
    exchange by a few per cent, which moves this change by less than one.
    """
    assert constant != 0.0
    assert abs(dispersive - constant) <= 0.01 * abs(constant)


def assert_flux_conserved(dispersive: float, constant: float, budget: float):
    """Both lossless runs keep the axial-flux energy within the route's budget.

    The free and planar routes measure energy by the axial flux of their medium
    (SEP 34), which is the quantity the lossless linear flow conserves; an
    instantaneous cubic source does no net work on it, since
    ``sum_omega omega Im(E* P)`` is ``integral(E dP/dt dt)``, zero for ``P ~ E^3``.
    What remains is the fixed-step integrator's truncation of a quadratic invariant
    and rounding, which ``budget`` bounds. A response that did net work, or a
    measure that were not the conserved one, would move the change by its own
    share of the nonlinear exchange, far above the budget.
    """
    assert abs(constant) <= budget
    assert abs(dispersive) <= budget


# ----------------------------------------------------------------- free space


FREE_CASES: dict[str, dict[str, Any]] = {
    "field full": {"kind": "field", "thg": True},
    "field conserving": {"kind": "field", "thg": False},
    "envelope conserving": {"kind": "envelope", "thg": False},
}


def free_grid(kind: str) -> TimeGrid:
    """Return a free-space test grid of either kind."""
    if kind == "envelope":
        return EnvelopeGrid.from_limits(1e-4, 800e-9, (500e-9, 1.6e-6), 100e-15)
    return FieldGrid.from_limits(1e-4, 800e-9, (400e-9, 2e-6), 100e-15)


def sub_octave(grid: TimeGrid) -> np.ndarray:
    """Return the coarse bins from 300 to 450 THz: a launch confined to them.

    The number-conserving sector equals the constant ``thg=False`` rule exactly when
    ``|u|^2 u`` has no negative-frequency content (SEP 6, decision Q4), which a field
    inside ``[a, b]`` with ``b < 2a`` guarantees: its cubic lies in ``[2a - b,
    2b - a]``. A Gaussian's tails reach the whole active band at the 1e-6 level, and
    there the two rules differ, as they should.
    """
    frequency = grid.omega / (2 * np.pi)
    return (frequency > 300e12) & (frequency < 450e12)


def free_problem(
    response: object,
    grid: TimeGrid,
    space: FreeGrid | None = None,
    *,
    band: np.ndarray | None = None,
):
    """Return a Cartesian free-space problem: vacuum optics, an argon density."""
    space = FreeGrid(3e-4, 9, 4e-4, 12, window_factor=0) if space is None else space
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-6), 8e-5
    )
    return assemble(
        grid,
        None,
        [response],
        pulse.build(grid, space, medium=SpatialMedium())
        * (grid.active_mask if band is None else band),
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1e-4],
        dz_max=1e-4,
        dz_init=1e-5,
        schedule=FixedSchedule.uniform(0.0, 1e-4, 1e-5),
        density=DENSITY,
    )


@pytest.mark.parametrize("evaluator", ["polynomial", "symmetric"])
@pytest.mark.parametrize("case", list(FREE_CASES))
def test_the_free_route_evaluates_the_reference_at_a_node(monkeypatch, case, evaluator):
    """The pass the free pipeline calls is the reference at each node, to rounding."""
    kind, thg = FREE_CASES[case]["kind"], FREE_CASES[case]["thg"]
    grid = free_grid(kind)
    response = DispersiveKerrResponse(ARGON, evaluator, thg=thg)
    problem = free_problem(response, grid)
    calls = caught(monkeypatch, passes, "nonlinear_sources")
    stage_derivative(problem, grid)
    args, kwargs, result = calls[0]
    field_t = args[0]
    node = int(np.argmax(np.max(np.abs(field_t), axis=(1, 2))))
    (density,) = kwargs["scalars"].dispersive_kerr_densities
    expected = response.evaluate(
        field_t[node],
        density,
        kind=kind,
        omega=grid.omega_fine,
        window=grid.t_window_fine,
    )
    bound = dispersive_response_bound(problem.grids.n_to, evaluator)
    assert result.current is None
    assert_within(result.polarisation[node], expected, bound, what="polarisation")
    assert density == pytest.approx(DENSITY, rel=1e-12)


@pytest.mark.parametrize("thg", [True, False])
@pytest.mark.parametrize("geometry", ["plane", "line"])
def test_without_dispersion_the_free_derivative_is_the_constant_one(thg, geometry):
    """The transverse transforms and the prefactor around the leaf: Kerr's, measured.

    The launch is confined below an octave (:func:`sub_octave`), where the
    conserving sector is the ``thg=False`` rule. The collar is unit: the dispersive
    response takes it on its products before ``Pi_+`` and its output multipliers, the
    constant cubic on its polarisation, and the conserving sector's ``Pi_+`` removes
    what the collar's edges leak below zero frequency, 1.2e-9 of the largest bin
    here (:func:`test_the_collar_order_matters_only_where_the_collar_varies`).
    """
    grid = free_grid("field")
    space = (
        FreeGrid(3e-4, 9, 4e-4, 12, window_factor=0)
        if geometry == "plane"
        else LineGrid(3e-4, 16, window_factor=0)
    )
    band = sub_octave(grid)
    got = stage_derivative(
        unit_collar(
            free_problem(
                DispersiveKerrResponse(FLAT, "polynomial", thg=thg),
                grid,
                space,
                band=band,
            )
        ),
        grid,
    )
    expected = stage_derivative(
        unit_collar(
            free_problem(
                KerrResponse(FLAT.gamma3_static, thg=thg), grid, space, band=band
            )
        ),
        grid,
    )
    assert_same_derivative(got, expected)


def free_energy_change(response: object, materials: MaterialSelection) -> float:
    """Return the energy change of ten fixed steps of a lossless free-space run."""
    length = 2e-4
    grid = FieldGrid.from_limits(length, 800e-9, (200e-9, 3e-6), 100e-15)
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[length],
        dz_max=length / 10,
        dz_init=length / 10,
        schedule=FixedSchedule.uniform(0.0, length, length / 10),
        integrator=RungeKutta(),
        backend="numpy_ref",
        density=DENSITY,
        stats=default_spec("free"),
    )
    simulation = Simulation.from_free(
        grid,
        FreeGrid(3e-4, 9, 4e-4, 12, window_factor=0),
        SpatialMedium(),
        [response],
        SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=15e-15, energy=3e-6), 6e-5),
        settings=settings,
        materials=materials,
        noise=None,
    )
    output = MemoryOutput()
    simulation.run(output)
    return relative_change(output, "energy")


@pytest.mark.medium
def test_a_lossless_free_run_does_no_more_work_than_the_constant_cubic():
    """Ten fixed steps of a 15 fs, 3 uJ beam, the number-conserving sector.

    The free route's energy is the vacuum axial flux, which weights each column by
    its obliquity ``c k_z / omega``: the quantity the UPPE's lossless flow conserves.
    Both runs change it by about 1e-16, rounding; the former paraxial measure
    ``eps0 c / 2 int |E|^2`` changed by -2.09e-11, which the 1e-12 budget rejects.
    """
    dispersive = free_energy_change(
        DispersiveKerrResponse(ARGON, "polynomial", thg=False),
        MaterialSelection(electronic="gas-polynomial"),
    )
    constant = free_energy_change(
        KerrResponse(carrier_gamma3(800e-9), thg=False), CARRIER
    )
    assert_flux_conserved(dispersive, constant, 1e-12)


# ---------------------------------------------------------------- planar hybrid


HALF_GAP = 63.5e-6
PLANAR_LENGTH = 2e-3


def planar_simulation(
    response: object, *, steps: int = 1, materials: MaterialSelection = CARRIER
):
    """Return a hybrid planar run: TE1 and TE3 of a lossless argon gap, four k_y."""
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
        integrator=RungeKutta(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SolpropWarning)
        return Simulation.from_planar(
            grid,
            PlanarGrid(modes, 1e-3, 4),
            [response],
            pulse,
            settings=settings,
            materials=materials,
        )


@pytest.mark.parametrize("evaluator", ["polynomial", "symmetric"])
@pytest.mark.parametrize("thg", [True, False])
def test_the_planar_route_evaluates_the_reference_at_a_node(
    monkeypatch, thg, evaluator
):
    """The slab synthesis's nodes see the reference, to its rounding."""
    response = DispersiveKerrResponse(ARGON, evaluator, thg=thg)
    simulation = planar_simulation(response)
    problem = simulation.translate()
    calls = caught(monkeypatch, passes, "nonlinear_sources")
    stage_derivative(problem, simulation.grid)
    args, kwargs, result = calls[0]
    field_t = args[0]
    node = int(np.argmax(np.max(np.abs(field_t), axis=(1, 2))))
    (density,) = kwargs["scalars"].dispersive_kerr_densities
    expected = response.evaluate(
        field_t[node],
        density,
        kind="field",
        omega=simulation.grid.omega_fine,
        window=simulation.grid.t_window_fine,
    )
    bound = dispersive_response_bound(problem.grids.n_to, evaluator)
    assert result.current is None
    assert_within(result.polarisation[node], expected, bound, what="polarisation")


@pytest.mark.parametrize("thg", [True, False])
def test_without_dispersion_the_planar_derivative_is_the_constant_one(thg):
    """The slab synthesis, projection and y transforms around the leaf: Kerr's.

    The launch is confined below an octave, as on the free route.
    """
    derivatives = []
    for response in (
        DispersiveKerrResponse(FLAT, "polynomial", thg=thg),
        KerrResponse(FLAT.gamma3_static, thg=thg),
    ):
        simulation = planar_simulation(response)
        launch = simulation._field0 * sub_octave(simulation.grid)
        problem = simulation.with_input(launch).translate()
        derivatives.append(stage_derivative(problem, simulation.grid))
    assert_same_derivative(*derivatives)


@pytest.mark.medium
def test_a_lossless_planar_run_does_no_more_work_than_the_constant_cubic():
    """Ten fixed steps over 2 mm of a 127 um argon gap, the number-conserving sector.

    The planar measure weighs the slab amplitudes by ``c Re(q)/omega``, the axial
    flux of each mode and y column, which the lossless flow conserves (SEP 34 §4.3):
    both runs change it by -4.3e-14 at ten steps and -2.2e-15 at twenty, the fixed
    step's truncation falling as about its fourth power. The former measure, the
    amplitudes' plain ``|a|^2``, changed by 6.48e-9, which the 1e-12 budget rejects.
    """
    changes = []
    for response, materials in (
        (
            DispersiveKerrResponse(ARGON, "polynomial", thg=False),
            MaterialSelection(electronic="gas-polynomial"),
        ),
        (KerrResponse(carrier_gamma3(800e-9), thg=False), CARRIER),
    ):
        output = MemoryOutput()
        planar_simulation(response, steps=10, materials=materials).run(output)
        changes.append(relative_change(output, "energy"))
    dispersive, constant = changes
    assert_flux_conserved(dispersive, constant, 1e-12)


# -------------------------------------------------------------- general modal


FILL = Mixture.fill({"Ar": 1.0}, 2.0)
"""The gas region's fill: 2 bar of argon at room temperature."""

CORE_CHI3 = 2.2e-22
"""The solid core's declared susceptibility, m^2/V^2, silica-like."""


def strand(
    cladding: MaterialKerr | GasKerr | None, *, core: float | None = None
) -> ModalGuide:
    """Return a 1.5 um silica-like strand's HE11 guide, sampled over the run's band.

    The core is a constant solid region (``MaterialKerr``, or linear), the unbounded
    exterior the gas region (``GasKerr``) or anything else declared for it.
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
            () if cladding is None else (cladding,),
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


GAS_STRAND = strand(GasKerr(FILL))


def general_problem(guide: ModalGuide, policy: str, form: KerrForm, *, leaf=None):
    """Return an assembled general modal problem launching 2 nJ into HE11."""
    grid = general_grid()
    route = general_modal_route(
        guide,
        grid,
        z_max=GENERAL_Z_END,
        omega_reference=grid.omega0,
        kerr_form=form,
        selection=MaterialSelection(electronic=policy),
    )
    if leaf is not None:
        route = dataclasses.replace(route, dispersive=leaf)
    state = np.zeros((1, 1, 1, grid.n_omega), complex)
    state[0, 0, 0] = GaussianPulse(wavelength=835e-9, fwhm=40e-15, energy=2e-9).build(
        grid
    )
    return route, assemble(
        grid,
        list(guide.mode_collection),
        [],
        state,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[GENERAL_Z_END],
        dz_max=1e-4,
        dz_init=1e-5,
        general=route,
    )


@pytest.mark.parametrize("evaluator", ["polynomial", "symmetric"])
@pytest.mark.parametrize("form", ["real", "analytic"])
def test_a_gas_region_evaluates_the_reference_at_its_nodes(
    monkeypatch, form, evaluator
):
    """The gas region's nodes see the reference; the solid core's see none of it.

    ``kerr_form="real"`` takes the full cubic and ``"analytic"`` the
    number-conserving sector (SEP 6, decision Q4), on the real three-component
    field the synthesis builds.
    """
    guide = strand(GasKerr(FILL), core=CORE_CHI3)
    route, problem = general_problem(guide, POLICIES[evaluator], form)
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None and leaf.regions is not None
    assert leaf.labels == ("cladding:Ar",) and list(leaf.regions) == [1]
    assert leaf.sector == ("full" if form == "real" else "conserving")
    regional = problem.responses.regional_responses
    assert regional is not None and list(regional.response_region) == [0]
    calls = caught(monkeypatch, general_modal, "dispersive_polarisation")
    stage_derivative(problem, general_grid())
    (field, _, node_region, densities), _, result = calls[0]
    outside = np.flatnonzero(node_region == 1)
    node = int(outside[np.argmax(np.max(np.abs(field[outside]), axis=(1, 2)))])
    (response,) = route.electronic
    # `evaluate` takes the state's one or two components and this node has three,
    # so its arithmetic is applied here as it stands: the density times the static
    # coefficient times the cubic of the response's own multipliers.
    expected = (densities[0] * response.coefficient) * dispersive_cubic(
        field[node],
        response.weights(general_grid().omega_fine, kind="field"),
        kind="field",
        sector=response.sector,
    )
    bound = dispersive_response_bound(problem.grids.n_to, evaluator)
    assert_within(result[node], expected, bound, what="polarisation")
    # Structural: the solid core's nodes are never written.
    assert not np.any(result[node_region == 0])
    assert densities[0] == pytest.approx(FILL.density(np.zeros(1))[0, 0], rel=1e-12)


@pytest.mark.parametrize("form", ["real", "analytic"])
def test_without_dispersion_a_gas_region_is_the_constant_region(form):
    """The synthesis, the projection and the region's nodes around the leaf.

    The flat channel against a region declaring ``chi3 = N gamma3_0``: the full cubic
    is the real form's tensor and the conserving sector the analytic form's below an
    octave. Measured 1.2e-16 and 2.3e-16 of the largest bin.
    """
    route, problem = general_problem(GAS_STRAND, "gas-polynomial", form)
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None
    flat = dispersive_terms(
        [DispersiveKerrResponse(FLAT, "polynomial", thg=form == "real")],
        leaf.labels,
        leaf.densities,
        general_grid(),
        regions=leaf.regions,
    )
    _, flat_problem = general_problem(GAS_STRAND, "gas-polynomial", form, leaf=flat)
    density = GasKerr(FILL).densities()[0]
    constant = strand(MaterialKerr(chi3=density * FLAT.gamma3_static))
    _, expected = general_problem(constant, "carrier-kerr", form)
    assert expected.responses.dispersive_kerr is None
    assert_same_derivative(
        stage_derivative(flat_problem, general_grid()),
        stage_derivative(expected, general_grid()),
    )
    assert route.electronic[0].model == ARGON


def test_a_solid_region_and_a_gas_region_add():
    """Two regions, two laws: the stage is the sum of each region's alone.

    The derivative is linear in the polarisation and the regions own disjoint nodes,
    so the two-region guide's stage is the sum of the solid-only and the gas-only
    guides' to a few roundings of each (measured 1.4e-15 of the largest bin). The
    solid is weak here, 1e-29 m^2/V^2, so that the gas region's evanescent share is
    a fifth of the whole rather than the 1e-8 beside a silica core, and the
    comparison sees it.
    """
    weak = 1e-29
    both, solid, gas = (
        stage_derivative(
            general_problem(guide, "gas-polynomial", "real")[1], general_grid()
        )
        for guide in (
            strand(GasKerr(FILL), core=weak),
            strand(None, core=weak),
            GAS_STRAND,
        )
    )
    scale = float(np.abs(both).max())
    assert float(np.abs(gas).max()) > 0.1 * scale
    assert float(np.abs(solid).max()) > 0.1 * scale
    np.testing.assert_allclose(both, solid + gas, rtol=0, atol=1e-14 * scale)


def test_under_a_carrier_policy_a_gas_region_is_a_constant_cubic():
    """The Kerr coefficient at the run's carrier, times the fill's density."""
    route, problem = general_problem(GAS_STRAND, "carrier-kerr", "real")
    assert problem.responses.dispersive_kerr is None and not route.electronic
    regional = problem.responses.regional_responses
    assert regional is not None and list(regional.response_region) == [1]
    density = GasKerr(FILL).densities()[0]
    carrier = 2 * math.pi * 299792458.0 / general_grid().omega0
    expected = density * carrier_gamma3(carrier)
    assert float(regional.chi3[0, 0, 0]) == pytest.approx(expected, rel=1e-14)


def test_a_mixture_region_has_one_channel_per_species():
    """Each population of the fill is its own channel, in the region's nodes."""
    mixture = strand(GasKerr(Mixture.fill({"He": 0.5, "Ar": 0.5}, 2.0)))
    route, problem = general_problem(mixture, "symmetric-kerr", "analytic")
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None and leaf.regions is not None
    assert leaf.labels == ("cladding:He", "cladding:Ar")
    assert list(leaf.regions) == [1, 1]
    assert [one.model.material for one in route.electronic] == ["He", "Ar"]


HELIUM_ARGON = strand(GasKerr(Mixture.fill({"He": 0.5, "Ar": 0.5}, 2.0)))


@pytest.mark.parametrize("form", ["real", "analytic"])
def test_a_symmetric_mixture_region_takes_its_fills_mixture_filter(monkeypatch, form):
    """The region's nodes see one cubic under the fill's mixture filter, to rounding.

    ``P = F G_mix[(G_mix E)^3]`` with ``F`` and ``g_mix`` built from the two channels'
    coefficients and densities by ``mixture_filter``; the solid core sees none of it.
    """
    _, problem = general_problem(HELIUM_ARGON, "symmetric-kerr", form)
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None and leaf.groups() == ((0, 1),)
    calls = caught(monkeypatch, general_modal, "dispersive_polarisation")
    stage_derivative(problem, general_grid())
    (field, _, node_region, densities), _, result = calls[0]
    total, curve = mixture_filter(
        leaf.filters,
        np.array(leaf.coefficients) * np.array(densities),
    )
    outside = np.flatnonzero(node_region == 1)
    expected = total * dispersive_cubic(
        field[outside],
        DispersiveWeights("symmetric", leaf.positive, curve),
        kind="field",
        sector=leaf.sector,
    )
    bound = dispersive_response_bound(problem.grids.n_to, "symmetric")
    assert_within(result[outside], expected, bound, what="polarisation")
    # Structural: the solid core's nodes are never written.
    assert not np.any(result[node_region == 0])


@pytest.mark.parametrize(
    ("guide", "policy", "buffers"),
    [
        (GAS_STRAND, "symmetric-kerr", 4),
        (HELIUM_ARGON, "symmetric-kerr", 4),
        (GAS_STRAND, "gas-polynomial", 10),
        (HELIUM_ARGON, "gas-polynomial", 10),
    ],
    ids=["argon-symmetric", "mixture-symmetric", "argon-polynomial", "mixture-poly"],
)
def test_the_general_chunk_counts_the_dispersive_buffers(guide, policy, buffers):
    """Each filtered field and product holds two buffers per node and component.

    The symmetric evaluator one field and one product per gas region, a mixture's
    under its filter; the polynomial two fields and two products per group of equal
    leg weight and a cube per region -- on this guide's 815-860 nm band no leg reaches
    a bound, so helium and argon share one group. The chunk
    admits two nodes at twice a node's bytes and one a byte short, as the fused
    path's admission is tested.
    """
    _, problem = general_problem(guide, policy, "real")
    leaf = problem.responses.dispersive_kerr
    assert general_modal.general_dispersive_buffers(leaf) == buffers
    assert general_modal.general_dispersive_buffers(None) == 0
    grid = general_grid()
    n_to, n_fine = grid.n_to, int(grid.omega_fine.size)
    per_node = (
        (
            (general_modal.BUFFERS_PER_GENERAL_NODE + buffers) * 3
            + general_modal.PAIR_BUFFERS_PER_GENERAL_NODE * 9
        )
        * max(n_to, n_fine)
        * 16
    )
    chunk = general_modal.general_node_chunk
    assert chunk(3, n_to, n_fine, 2 * per_node, dispersive=buffers) == 2
    assert chunk(3, n_to, n_fine, 2 * per_node - 1, dispersive=buffers) == 1
    # Without the response the same bytes hold more nodes.
    assert chunk(3, n_to, n_fine, 10 * per_node) > 10
    # The route's own maps take the response's buffers into their chunk.
    tables = problem.transform.modal_tables
    assert tables is not None
    expected = min(
        chunk(3, n_to, n_fine, DEFAULT_NODE_BUDGET, dispersive=buffers),
        int(tables.region_index.size),
    )
    maps = _general_maps(problem)
    assert maps is not None and maps.node_chunk == expected


def test_the_route_records_the_gas_region_and_its_channels():
    """The guide's identity names the fill; the modal identity the dispersive law."""
    route, problem = general_problem(GAS_STRAND, "gas-polynomial", "real")
    identity: Any = route.guide_identity
    gas = identity["regions"][1]["gas"]
    assert [one["species"] for one in gas["populations"]] == ["Ar"]
    assert "gas" not in identity["regions"][0]
    channels = list(GAS_STRAND.mode_collection.channel_labels)
    record: Any = modal_identity(
        route, general_grid(), channels=channels, has_noise=False
    )
    assert record["dispersive"] == {
        "evaluator": "polynomial",
        "sector": "full",
        "tensor": "isotropic-kleinman",
        "channels": [{"label": "cladding:Ar", "region": 1}],
    }
    carrier_route, _ = general_problem(GAS_STRAND, "carrier-kerr", "real")
    assert "dispersive" not in modal_identity(
        carrier_route, general_grid(), channels=channels, has_noise=False
    )
    assert problem.responses.dispersive_kerr is not None


def general_simulation(
    guide: ModalGuide, policy: str, form: KerrForm, *, steps=10, energy=2e-9
):
    """Return a lossless general modal run of fixed steps over the millimetre."""
    grid = general_grid()
    return Simulation.from_waveguide(
        (),
        GENERAL_Z_END,
        provider=guide,
        grid=grid,
        lambda0=835e-9,
        fwhm=20e-15,
        energy=energy,
        materials=MaterialSelection(electronic=policy),
        kerr_form=form,
        schedule=FixedSchedule.uniform(0.0, GENERAL_Z_END, GENERAL_Z_END / steps),
        integrator=RungeKutta(),
        absorber=lossless(grid),
        stats=StatsSpec(names=("physical_energy",)),
        backend="numpy_ref",
        save_planes=[GENERAL_Z_END],
        dz_max=GENERAL_Z_END / steps,
        dz_init=GENERAL_Z_END / steps,
        noise=None,
    )


def test_the_manifest_records_each_gas_region_and_the_vector_edge():
    """The law, each channel's support and regions, with the any-polarisation edge."""
    simulation = general_simulation(GAS_STRAND, "symmetric-kerr", "real")
    manifest = simulation.material_manifest()
    (entry,) = manifest.electronic
    assert entry["label"] == "cladding:Ar"
    assert entry["band_edge_any"] == ARGON.band_edge_any
    regions: Any = entry["regions"]
    assert regions["polarisation"] == "any"
    policy: Any = manifest.policies["electronic_response"]  # pyright: ignore[reportOptionalSubscript]
    assert policy["law"] == "symmetric-kerr"
    assert [one["label"] for one in policy["species"]] == ["cladding:Ar"]
    assert "mixture_filters" not in policy
    carrier = general_simulation(GAS_STRAND, "carrier-kerr", "real").material_manifest()
    assert carrier.policies["electronic_response"]["law"] == "carrier-kerr"  # pyright: ignore[reportOptionalSubscript, reportIndexIssue]


def test_the_manifest_records_a_regions_mixture_filter_and_its_composition():
    """The filter of the region's fill, its region and each population's density."""
    manifest = general_simulation(
        HELIUM_ARGON, "symmetric-kerr", "real"
    ).material_manifest()
    policy: Any = manifest.policies["electronic_response"]  # pyright: ignore[reportOptionalSubscript]
    (recorded,) = policy["mixture_filters"]
    assert recorded["region"] == 1
    assert [one["label"] for one in recorded["composition"]] == [
        "cladding:He",
        "cladding:Ar",
    ]
    assert all(len(one["density"]) == 64 for one in recorded["composition"])
    assert [entry["filter"] for entry in manifest.electronic] == [
        {"kind": "mixture", "labels": ["cladding:He", "cladding:Ar"]}
    ] * 2


@pytest.mark.medium
@pytest.mark.parametrize("form", ["real", "analytic"])
def test_a_lossless_gas_region_does_no_more_work_than_the_constant_cubic(form):
    """Ten fixed steps of 20 uJ in the strand, its gas region the only response.

    The route's ``physical_energy`` takes the identity metric, which the basis meets
    to its Gram defect: the constant cubic's run changes it by -6.246e-12, ten steps
    or twenty alike and in proportion to the pulse energy, and the dispersive run by
    -6.253e-12 in both sectors.
    """
    changes = []
    for policy in ("gas-polynomial", "carrier-kerr"):
        output = MemoryOutput()
        general_simulation(GAS_STRAND, policy, form, energy=2e-5).run(output)
        changes.append(relative_change(output, "physical_energy"))
    assert_no_added_work(*changes)


def test_a_gas_region_without_a_hyperpolarizability_model_is_refused_by_name():
    """A supplied species has no optmat model to evaluate: the region says which."""
    supplied = Species("X", species("Ar").susceptibility, gamma3=1e-51)
    guide = strand(GasKerr(Mixture.fill({supplied: 1.0}, 1.0)))
    with pytest.raises(TranslationError, match=r"region 'cladding'.*'X'"):
        general_problem(guide, "gas-polynomial", "real")
    _, problem = general_problem(guide, "carrier-kerr", "real")
    regional = problem.responses.regional_responses
    assert regional is not None and list(regional.response_region) == [1]


def test_a_gas_region_declares_one_constant_fill_and_no_other_cubic():
    with pytest.raises(ModeError, match="varies along z"):
        GasKerr(Mixture.fill({"Ar": 1.0}, Linear(1.0, 2.0, 1e-3), temperature=294.0))
    solid = IndexMaterial(1.0)
    with pytest.raises(ModeError, match="one electronic response"):
        MaterialRegion(
            "cladding", 1e-6, None, solid, (MaterialKerr(chi3=1e-25), GasKerr(FILL))
        )
    with pytest.raises(ModeError, match="no MaterialKerr"):
        MaterialRegion(
            "cladding", 1e-6, None, solid, (GasKerr(FILL), SilicaTensorResponse())
        )


def test_the_general_source_refuses_densities_that_do_not_match_its_channels():
    _, problem = general_problem(GAS_STRAND, "symmetric-kerr", "real")
    maps = _general_maps(problem)
    assert maps is not None and maps.dispersive is not None
    grids = problem.grids
    with pytest.raises(BackendError, match="1 dispersive channel"):
        general_modal.general_source(
            problem.initial.field0[0],
            maps,
            grids.crop_index,
            n_to=grids.n_to,
            pad_scale=grids.pad_scale,
            dispersive_densities=(),
        )


def test_a_gas_fill_describes_itself_by_its_populations():
    fill = GasKerr(Mixture.fill({"He": 0.5, "Ar": 0.5}, 1.0))
    assert repr(fill) == "GasKerr(He, Ar)"
    text = str(fill)
    assert text.startswith("GasKerr: the electronic response of a region's gas fill")
    assert "fill" in text


def test_a_general_leaf_naming_an_undeclared_region_is_refused():
    _, problem = general_problem(GAS_STRAND, "symmetric-kerr", "real")
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None and leaf.regions is not None
    regions = np.array([7], dtype=np.int32)
    regions.setflags(write=False)
    spoiled = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses,
            dispersive_kerr=dataclasses.replace(leaf, regions=regions),
        ),
    )
    with pytest.raises(TranslationError, match="names region 7"):
        spoiled.validate()


def test_a_gas_region_beyond_its_support_is_logged_once(caplog):
    """The general route names the gas and its band once, with the vector edge."""
    grid = FieldGrid.from_limits(1e-3, 800e-9, (100e-9, 2e-6), 200e-15)
    helium = DispersiveKerrResponse(
        OptmatKerr("He").dispersive_model(), "symmetric", species="cladding:He"
    )
    with caplog.at_level(logging.INFO, logger="solprop.translate.general"):
        _log_gas_regions([helium], grid)
    (message,) = [r.getMessage() for r in caplog.records if "cladding:He" in r.message]
    assert "best effort" in message and "is not refused" in message
