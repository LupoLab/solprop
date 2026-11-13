# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 6's dispersive electronic response: the spatial and general routes' cases.

The problems, measurements and rounding bound of a compiled backend's ladder on these
routes (``dispersive_rows.py``) against ``numpy_ref``, which applies the
multipliers at every node and is the oracle.

**The bounds are derived from summation lengths**, as on the fused path
(``dispersive_cases.py``). One transform of ``n`` points is ``8 eps
ceil(log2 n)``. One evaluation of the constant cubic on a spatial route is two
temporal transforms and two transverse ones (of ``n_columns`` points; the planar
route's y transforms and its slab rule's ``n_x`` sums, the general route's synthesis
and projection over the modes and nodes). The response adds, per product of its
layout, one inverse transform of a filtered field and one forward transform of the
product; the planar route adds a synthesis and a projection for each. The reference
takes as many again at every node, so each count is doubled.
"""

import math
import warnings
from typing import Any

import numpy as np

from bounds import gamma
from legacy_geometry import LEGACY_GRID, bare_absorber
from modal_provider_cases import GENERAL_Z_END, WAVELENGTH, general_grid
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
from solprop.translate.problem import PropagationProblem, Translation, assemble
from solprop.translate.responses import KerrForm
from solprop.translate.run_request import RunRequest

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
) -> Translation:
    """Return a Cartesian (or line) free-space problem of ten fixed steps, and its
    request.

    Vacuum optics and argon's density, or a gas mixture's own medium.
    """
    space = (
        LineGrid(3e-4, 16, collar_fraction=0)
        if line
        else FreeGrid(3e-4, 9, 4e-4, 12, collar_fraction=0)
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
) -> tuple[PropagationProblem, RunRequest, TimeGrid]:
    """Return a hybrid planar run: TE1 and TE3 of a lossless argon gap, four k_y.

    `absorbed` adds a temporal collar in the in-stage scheme, whose term reads the
    block's bare rows after the material.
    """
    grid = FieldGrid.from_limits(
        PLANAR_LENGTH, 800e-9, (200e-9, 3e-6), 100e-15, **LEGACY_GRID
    )
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
        absorber=(
            absorber(grid, boundary_length=1.0e3) if absorbed else bare_absorber(grid)
        ),
        # The in-stage scheme, which the planar route no longer takes by default.
        absorber_scheme="nonlinear",
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
        return simulation.translate(), simulation._run_request(), simulation.grid


def raman(thg: bool) -> object:
    """Return a delayed response beside the electronic one: the column runs too.

    Silica's shape at a fifth of argon's static coefficient per particle, so that the
    two are of a size and the argon runs' ten steps stay stable.
    """
    return response_for(
        "SiO2", amplitude=0.2 * ARGON.gamma3_static, fraction=0.18, thg=thg
    )


# --- the measurements ------------------------------------------------------------


def evaluation_bound(problem: PropagationProblem) -> float:
    """Bound one evaluation's rounding on a spatial or general route (the docstring)."""
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None
    groups = len({row.tobytes() for row in np.ascontiguousarray(leaf.filters)})
    products = leaf.n_channels if leaf.evaluator == "symmetric" else 1 + 2 * groups
    temporal = 8 * EPS * math.ceil(math.log2(problem.grids.n_to))
    columns = problem.shape[0] * problem.shape[1]
    transverse = 8 * EPS * math.ceil(math.log2(max(2, columns)))
    sums = gamma(max(2, problem.transform.n_nodes))
    response = products * (2 * temporal + 2 * sums)
    return 2 * (2 * temporal + 2 * transverse + 2 * sums + response)


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
) -> tuple[PropagationProblem, RunRequest, TimeGrid]:
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
    state = np.zeros((1, 1, grid.n_omega), complex)
    state[0, 0] = GaussianPulse(wavelength=835e-9, fwhm=40e-15, energy=2e-9).build(grid)
    problem, run_request = assemble(
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
    return problem, run_request, grid


CORE_CHI3 = 2.2e-22
"""The solid core's susceptibility, m^2/V^2, silica-like."""

STRANDS = {
    "gas alone": (ARGON_FILL, None),
    "beside a solid core": (ARGON_FILL, CORE_CHI3),
    "helium and argon": (Mixture.fill({"He": 0.5, "Ar": 0.5}, 2.0), None),
}


def route_case(
    route: str, evaluator: str
) -> tuple[PropagationProblem, RunRequest, Any]:
    """Return the ten-step case of a route the repeatability and thread tests run."""
    response = DispersiveKerrResponse(ARGON, evaluator)
    if route == "free":
        grid = free_grid("field", True)
        problem, run_request = free_problem([response], grid)
        return problem, run_request, grid
    if route == "planar":
        return planar_problem([response])
    return general_problem(strand(ARGON_FILL, core=CORE_CHI3), evaluator, "analytic")
