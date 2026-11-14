# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The low-storage march and the general z table: their rows of the case catalogue.

**What is compared.** Two marches that native runs in its own session, on the routes
they reach, each case one fixed schedule replayed on native and on ``numpy_ref``
(``steps``: the field at the schedule's end, normwise):

- **the low-storage RK4IP march** (SEP 24 section 7.2, ``low storage ...`` rows): the
  mode-averaged soliton, plain and under the modified noise, in both propagator modes;
  the modal route with no response, or the in-stage collar alone; a nitrogen gradient
  capillary with Raman and plasma (a varying operator); a dispersion declared along
  z, which takes the operator's z table; the radial route with its in-stage collar;
  the general modal route; a Cartesian Kerr beam in every flow treatment, with and
  without the collars and the noise; a recomputed flow beside a small loss; the line;
  the hybrid planar kind on every case of ``planar_cases.py``; the linear Cartesian
  and planar flows; SEP 6's dispersive response; and the general route's collar and
  linear guide;
- **the general z table** (SEP 29 section 6.6, ``z table ...`` rows): the helium
  gradient capillary with ``Phi``, the prefactor or both on the table, ten fixed steps
  of each march -- the fixed-order Adams method, Runge-Kutta and the low-storage
  march.

The native-only claims -- the march's registers, ledger and estimate, the held load,
``backend="auto"``, threads and the trial refusal; the table's slots, rate slot, ledger
and refusals -- stay in ``test_native_low_storage.py`` and ``test_native_z_table.py``.
The driver's z-table runs (``tests/driver/test_native_z_table.py``) stay too: they hold
every saved plane to its own norm, which the ``run`` rung does not.

**The bounds are the old tests', fixed before native runs.** A low-storage row is
bounded by ``low_storage_cases.parity_bound``: four evaluations a step, each a
temporal and a transverse transform pair, a response's cumulative sums and the fold,
and the flows' arguments, carried by a stated growth (one where there is no source);
the general route adds its source's bound under the run's nonlinear scale
(:func:`general_bound`), the dispersive response four of its evaluations a step
(:func:`dispersive_bound`). A z-table row is bounded by ``march_bounds.march_bound``
at the reference's launch phase, at least a tenth (:func:`z_table_bound`).

**The guard.** Each row's term is switched off on the translated problem: the Kerr
term, the Raman rows or the plasma density (``optmat_rows``), the dispersive response,
the in-stage collar, the loss, or the table's variation along z (:func:`frozen_table`).
A row with no source holds the linear flow, and its term is a part of the flow: the
fibre's dispersion, the Cartesian frame, the slab modes' confinement or the general
guide's exponent (:func:`without_dispersion`, :func:`without_frame`,
:func:`without_confinement`, :func:`without_modal_flow`). Each guarded rung's
reference must then move by more than ``parity_ladder.DETECTION`` (a thousand) of its
bounds.

**How to read a row.** Every row is a :func:`session_row` call, for example::

    session_row(
        name="low storage cartesian tabulate in-stage collar plain",
        route="cartesian",
        stepper="low storage",
        response="kerr",
        features="tabulated flows, in-stage collars",
        why="The low-storage march carries a Cartesian Kerr beam ...",
        build=partial(cartesian, "tabulate", collar=True, noisy=False),
        switch_off=without_kerr,
        rungs={"steps": parity(transverse="columns")},
    )

``name`` makes the test id; ``route``, ``stepper``, ``response`` and ``features`` are
the axes the row covers. ``build`` makes the problem: here the Cartesian march's case
with its collars. ``rungs`` names what is compared, with its bound: the field at the
schedule's end, inside the parity bound with the transverse transform's levels.
``switch_off`` removes the row's term for the guard; ``why`` is the claim a failure
quotes.
"""

import dataclasses
import functools
import math
import warnings
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

from backends.conftest import (
    BETA2,
    DISPERSION_LENGTH,
    GAMMA,
    GNLSE_SETTINGS,
    WAVELENGTH,
    replace_groups,
    soliton_field,
)
from backends.dispersive_rows import without_response
from backends.low_storage_cases import (
    CARTESIAN_CHUNK,
    MARCH,
    STEPS,
    cartesian_march,
    marched,
    parity_bound,
    uniform,
    varying_capillary,
    with_mode,
)
from backends.optmat_rows import without_kerr, without_plasma, without_raman
from backends.planar_cases import planar_case
from backends.spatial_cases import CHI3, QuadraticRate
from backends.spatial_cases import LENGTH as RADIAL_LENGTH
from backends.spatial_cases import make_problem as radial_problem
from general_parity import conditioning, nonlinear_scale, problem_for, scaled_case
from general_parity import source_bound as general_source_bound
from march_bounds import FIXED_ORDER, march_bound
from parity_ladder import Built, Row
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.backend import _general_maps
from solprop.backends.session import ExecutionOptions
from solprop.boundaries import Absorber, absorber
from solprop.constants import C
from solprop.driver import Simulation
from solprop.exceptions import GeneralModalBasisWarning, MaterialRangeWarning
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.modes.capillary import MarcatiliMode
from solprop.noise import modified_noise, spatial_noise
from solprop.nonlinear.gnlse import SpecifiedMode, problem_from
from solprop.nonlinear.raman import DampedOscillator, RamanResponse, undamped_dephasing
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.profiles import PressureGradient
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialFlux, SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, PIController
from solprop.translate.assembly import assemble
from solprop.translate.chebyshev import build as chebyshev
from solprop.translate.ionisation import build_rate_table
from solprop.translate.linop import delta_phi, phi_rate
from solprop.translate.problem import PropagationProblem, PropagatorMode, Translation
from solprop.translate.responses import prefactor_at
from solprop.translate.routes import flux_of

# --- the low-storage march's cases ----------------------------------------------------

NARROW_GRID = EnvelopeGrid.from_limits(1.0, WAVELENGTH, (500e-9, 2e-6), 1e-12)
"""A short-window envelope grid, 768 bins (``narrow_grid``)."""


def _fibre(beta2: float = BETA2) -> Any:
    """Return the soliton problems' dispersion-specified fibre (``gnlse``)."""
    return problem_from(
        [0.0, 0.0, beta2], omega_ref=NARROW_GRID.omega0, gamma=GAMMA, shock=False
    )


def soliton(mode: PropagatorMode, *, noisy: bool) -> Built:
    """Return the third-order soliton over half a dispersion length, ten steps.

    The modified noise is drawn once (seed 3) and assembled with the problem, so that
    each pass reads it at its own position.
    """
    length = DISPERSION_LENGTH / 2
    schedule = uniform(length)
    noise = modified_noise(NARROW_GRID, np.random.default_rng(3)) if noisy else None
    fibre = _fibre()
    problem, run_request = assemble(
        NARROW_GRID,
        fibre.mode,
        [fibre.kerr],
        soliton_field(NARROW_GRID, 3),
        save_planes=[length],
        **GNLSE_SETTINGS,
        controller=PIController(rtol=1e-10, atol=1e-18),
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        schedule=schedule,
        integrator=MARCH,
        propagator_mode=mode,
        noise=noise,
    )
    return Built(problem, run_request, NARROW_GRID)


def linear_fibre(*, collar: bool, beta2: float = BETA2) -> Built:
    """Return the fundamental soliton with no response, over 5 cm in ten steps.

    `collar` adds the default temporal collar in-stage, whose term is then the whole
    source.
    """
    extra: dict[str, Any] = (
        {"absorber": absorber(NARROW_GRID), "absorber_scheme": "nonlinear"}
        if collar
        else {}
    )
    length = 0.05
    schedule = uniform(length)
    problem, run_request = assemble(
        NARROW_GRID,
        _fibre(beta2).mode,
        [],
        soliton_field(NARROW_GRID, 1),
        save_planes=[length],
        **GNLSE_SETTINGS,
        controller=PIController(rtol=1e-8, atol=1e-18),
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        schedule=schedule,
        integrator=MARCH,
        **extra,
    )
    return Built(problem, run_request, NARROW_GRID)


def varying(mode: str) -> Built:
    """Return the nitrogen gradient capillary with Raman and plasma, a varying operator.

    The states are compared, not the statistics, whose profiles a session would need
    at construction: the request observes none.
    """
    simulation = varying_capillary()
    problem, run_request = with_mode(
        simulation.translate(),
        simulation._run_request().for_run(stats=StatsSpec()),
        mode,
    )
    return Built(problem, run_request, simulation.grid, modes=simulation.modes)


def tabled_fibre(mode: str) -> Built:
    """Return a fibre whose dispersion is declared along z, a millimetre in ten steps.

    ``beta_1`` rises by one per cent over the millimetre and the area by half, so the
    operator takes the z table and its flows are exponentiated from the table's held
    difference.
    """
    length = 1e-3
    grid = EnvelopeGrid.from_limits(length, 835e-9, (600e-9, 1300e-9), 2e-12)
    beta0 = 1.45 * 2 * np.pi / 835e-9
    beta1 = 1.47 / C
    nodes = np.array([0.0, length])
    fibre = SpecifiedMode.from_taylor_path(
        [[beta0, beta1 * (1 + 0.01 * z / length)] for z in nodes],
        nodes,
        omega_ref=grid.omega0,
        aeff=[1e-10, 1.5e-10],
    )
    simulation = Simulation.from_gnlse(
        0.1,
        length,
        fibre=fibre,
        shock=False,
        n_save=2,
        noise=None,
        wavelength=835e-9,
        wavelength_limits=(600e-9, 1300e-9),
        trange=2e-12,
        fwhm=200e-15,
        energy=1e-9,
        shape="gauss",
        schedule=uniform(length),
        dz_max=length / STEPS * 1.0000001,
        dz_init=length / STEPS,
        integrator=MARCH,
        backend="numpy_ref",
        # Ten replayed steps: no collar, as before the default one.
        absorber=False,
    )
    problem, run_request = with_mode(
        simulation.translate(), simulation._run_request(), mode
    )
    assert problem.linear.phi_representation == "table"
    return Built(problem, run_request, simulation.grid)


def radial(family: str) -> Built:
    """Return the radial Hankel route's Kerr beam beside its in-stage collar."""
    limits = (RADIAL_LENGTH, 800e-9, (600e-9, 1.2e-6), 150e-15)
    grid = (
        EnvelopeGrid.from_limits(*limits, thg=False)
        if family == "envelope"
        else FieldGrid.from_limits(*limits)
    )
    space = RadialGrid(3e-4, 16, collar_fraction=0.5)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), 80e-6
    )
    problem, run_request = radial_problem(
        grid,
        space,
        pulse.build(grid, space, medium=SpatialMedium()),
        [KerrResponse(CHI3, thg=False)],
        steps=STEPS,
    )
    problem, run_request = marched(problem, run_request, run_request.stepper.schedule)
    assert problem.transform.geometry is not None
    return Built(problem, run_request, grid)


@functools.cache
def _scaled(name: str) -> tuple[Any, Any, Any, np.ndarray]:
    """Return a general modal case's grid, guide, route and scaled launch, once."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeneralModalBasisWarning)
        return scaled_case(name)


def general(
    name: str, length: float, *, collar: bool = False, linear: bool = False
) -> Built:
    """Return a general modal guide's case over `length` in ten steps.

    `collar` adds the in-stage collar; `linear` drops the route's responses, so that no
    source is evaluated at all.
    """
    grid, guide, route, state = _scaled(name)
    schedule = uniform(length)
    options: dict[str, Any] = (
        {"absorber": absorber(grid), "absorber_scheme": "nonlinear"} if collar else {}
    )
    if linear:
        route = dataclasses.replace(route, responses=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeneralModalBasisWarning)
        problem, run_request = problem_for(
            grid,
            guide,
            route,
            state,
            save_planes=[length],
            schedule=schedule,
            dz_max=schedule.max_step,
            dz_init=schedule.max_step,
            integrator=MARCH,
            **options,
        )
    return Built(problem, run_request, grid)


def cartesian(mode: str, *, collar: bool, noisy: bool) -> Built:
    """Return the Cartesian march's Kerr beam, in chunks of 7 of its 108 columns.

    With the noise, the modified draw (seed 7) is assembled with the problem and the
    beam launched beside it, so that the Kerr source carries the draw.
    """
    noise = (
        spatial_noise(
            SpatialFlux.resolve(
                EnvelopeGrid.from_limits(
                    1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
                ),
                # The collar's window factor, as `cartesian_march` builds its grid.
                FreeGrid(3e-4, 9, 4e-4, 12, collar_fraction=0.25 if collar else 0),
                SpatialMedium(),
            ),
            np.random.default_rng(7),
            model="modified",
        )
        if noisy
        else None
    )
    problem, run_request, grid, space = cartesian_march(
        mode, noise=noise, collar=collar
    )
    if noisy:
        pulse = SpatialPulse(
            GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), 80e-6
        )
        problem = dataclasses.replace(
            problem,
            initial=dataclasses.replace(
                problem.initial,
                field0=pulse.build(grid, space, medium=SpatialMedium()),
            ),
        )
    return Built(problem, run_request, grid, options=CARTESIAN_CHUNK)


def small_loss(rate: float = 1e-3) -> Built:
    """Return a recomputed Cartesian flow beside a loss of `rate` nepers per metre.

    A thousandth of a neper per metre puts about 6e-8 in each flow's exponent: only an
    exact zero's magnitude is one, so a real part past it takes its exponential.
    """
    length = 1e-3
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
    )
    space = FreeGrid(3e-4, 9, 4e-4, 12, collar_fraction=0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), 80e-6
    )
    schedule = uniform(length, 4)
    problem, run_request = assemble(
        grid,
        None,
        [KerrResponse(1e-22, thg=False)],
        pulse.build(grid, space, medium=SpatialMedium()),
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[length],
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        schedule=schedule,
        integrator=MARCH,
        propagator_mode="recompute",
        absorber=Absorber(
            np.full(grid.n_omega, rate),
            np.zeros(grid.n_t),
            np.zeros(grid.n_to),
            grid.active_mask,
            5 * length,
        ),
        absorber_scheme="nonlinear",
    )
    return Built(problem, run_request, grid)


def line(family: str) -> Built:
    """Return the line kind's beam, ``n_y = 1``, through the Cartesian kernels.

    The held field-plasma source needs resolved collar samples (a 200 fs window); the
    other families keep the shorter one.
    """
    field = family.startswith("field")
    window = 200e-15 if family == "field_plasma" else 150e-15
    limits = (RADIAL_LENGTH, 800e-9, (600e-9, 1.2e-6), window)
    grid = (
        FieldGrid.from_limits(*limits)
        if field
        else EnvelopeGrid.from_limits(*limits, thg=False)
    )
    collar = family.endswith("collar")
    space = LineGrid(
        3e-4 if collar else 4e-4, 24, collar_fraction=0.3 if collar else 0.0
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-3),
        80e-6,
        jones=(1,) if family.endswith("raman") else (1, 0.5j),
    )
    density = 1.0
    if family.endswith("plasma"):
        responses: list[Any] = [
            PlasmaResponse("Ar", build_rate_table(QuadraticRate(), (1e3, 1e10)))
        ]
        density = 1e22
    elif family.endswith("raman"):
        responses = [
            RamanResponse(
                (DampedOscillator(1e-8, 2e14, undamped_dephasing()),), thg=False
            )
        ]
    else:
        responses = [KerrResponse(CHI3, thg=False)]
    problem, run_request = radial_problem(
        grid,
        space,
        pulse.build(grid, space, medium=SpatialMedium()),
        responses,
        steps=STEPS,
        density=density,
    )
    problem, run_request = marched(problem, run_request, run_request.stepper.schedule)
    if collar:
        problem = dataclasses.replace(
            problem,
            responses=dataclasses.replace(
                problem.responses, absorber_scheme="nonlinear"
            ),
        )
    return Built(problem, run_request, grid, options=CARTESIAN_CHUNK)


def planar(name: str, *, chunk: int | None = 2) -> Built:
    """Return a case of ``planar_cases.py`` on ten steps of the low-storage march."""
    grid, _, problem, run_request = planar_case(name, steps=STEPS)
    problem, run_request = marched(problem, run_request, run_request.stepper.schedule)
    options = None if chunk is None else ExecutionOptions(node_chunk=chunk)
    return Built(problem, run_request, grid, options=options)


def planar_flow() -> Built:
    """Return the planar Kerr case with no response and no collar: the flow alone."""
    built = planar("kerr", chunk=None)
    problem = built.problem
    problem = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses,
            kerr_coefficients=None,
            temporal_absorber=None,
            temporal_absorber_fine=None,
        ),
    )
    return dataclasses.replace(built, problem=problem)


def cartesian_flow() -> Built:
    """Return the Cartesian march's case with no response and no collar.

    Without the Kerr coefficients too, which a compiled backend may read: the pass is
    the fold alone, no transform.
    """
    built = cartesian("tabulate", collar=False, noisy=False)
    problem = built.problem
    problem = dataclasses.replace(
        problem,
        responses=dataclasses.replace(problem.responses, kerr_coefficients=None),
    )
    assert len(problem.responses.response_order) == 1
    return dataclasses.replace(built, problem=problem)


def dispersive() -> Built:
    """Return SEP 6's dispersive response on the Cartesian route, ten steps."""
    from backends.dispersive_route_cases import (  # noqa: PLC0415
        ARGON,
        free_grid,
        free_problem,
    )
    from solprop.nonlinear.responses import DispersiveKerrResponse  # noqa: PLC0415

    grid = free_grid("envelope", thg=False)
    problem, run_request = free_problem(
        [DispersiveKerrResponse(ARGON, "symmetric", thg=False)], grid, adams=False
    )
    problem, run_request = marched(problem, run_request, run_request.stepper.schedule)
    assert problem.responses.dispersive_kerr is not None
    return Built(problem, run_request, grid)


# --- the general z table's cases ------------------------------------------------------

Z_TABLE_LENGTH = 0.02
"""Two centimetres of the gradient capillary, in :data:`STEPS` fixed steps."""

FIELD_GRID = FieldGrid.from_limits(0.2, 800e-9, (200e-9, 4e-6), 100e-15)
"""The field grid of the parity and gradient-capillary tests (``field_grid``)."""

MARCHES = {
    "adams": FIXED_ORDER,
    "runge_kutta": None,
    "low_storage": LowStorageRungeKutta(),
}
"""Each march on the prescribed schedule: the exponential Adams method (a varying
operator's compensated march), Runge-Kutta (``None`` replays the schedule with it) and
the low-storage march."""

TABLES = {
    "both": (True, True),
    "phi": (True, False),
    "prefactor": (False, True),
}
"""Which of the two quantities take the z table; the other keeps its expansion."""


def capillary(field_grid: FieldGrid, integrator: Any) -> Translation:
    """Return the helium gradient capillary on ten fixed steps, its expansions intact.

    The fixture of ``tests/backends/conftest.py`` (``capillary_problem``) on a
    prescribed schedule, with the collar integrated exactly so that the explicit
    marches are stable at its steps.
    """
    mode = MarcatiliMode(
        125e-6, "He", PressureGradient.two_point(0.2, 2.0, 0.2), cladding=1.45
    )
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-6).build(field_grid)
    schedule = FixedSchedule.uniform(
        0.0,
        Z_TABLE_LENGTH,
        Z_TABLE_LENGTH / STEPS * 1.0000001,
        save_planes=[Z_TABLE_LENGTH],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return assemble(
            field_grid,
            mode,
            [KerrResponse(chi3=2.2e-52, thg=True)],
            start,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[Z_TABLE_LENGTH],
            dz_max=schedule.max_step,
            dz_init=schedule.max_step,
            schedule=schedule,
            integrator=integrator,
            absorber=absorber(field_grid),
            absorber_scheme="split_step",
        )


def _phi_table(linear: Any, values: Callable[[float], Any]) -> Any:
    """Return the operator with ``Phi`` on a z table built from `values`."""
    table = chebyshev(
        lambda z: np.stack([values(float(one)) for one in z]),
        *linear.z_span,
        tol=1e-10,
    )
    return dataclasses.replace(
        linear,
        phi_expansion=None,
        phi_representation="table",
        phi_breakpoints=table.breakpoints,
        phi_coefficients=table.coefficients,
        phi_degrees=table.degrees,
        hard_kinks=table.hard_kinks,
        phi_constant=None,
    )


def _prefactor_table(responses: Any, values: Callable[[float], Any]) -> Any:
    """Return the responses with the prefactor on a z table built from `values`."""
    table = chebyshev(
        lambda z: np.stack([values(float(one)) for one in z]),
        *responses.prefactor_span,
        tol=1e-10,
    )
    for array in (
        table.coefficients,
        table.breakpoints,
        table.degrees,
        table.hard_kinks,
    ):
        array.setflags(write=False)
    return dataclasses.replace(
        responses,
        prefactor_representation="table",
        prefactor=table,
        prefactor_expansion=None,
        prefactor_constant=None,
    )


def tabulate(
    translation: Translation, *, phi: bool = True, prefactor: bool = True
) -> Translation:
    """Return a translation with ``Phi``, the prefactor or both on the z table.

    The tables are built from the values the expansions give, as
    ``tests/backends/conftest.py``'s ``tabulated_problem`` builds them, the way a mode
    set that declares no parameters would present them.
    """
    problem, run_request = translation
    linear, responses = problem.linear, problem.responses
    if phi:
        original = linear
        linear = _phi_table(linear, lambda z: delta_phi(original, 0.0, z))
    if prefactor:
        held = responses
        responses = _prefactor_table(responses, lambda z: prefactor_at(held, z))
    return Translation(
        replace_groups(
            problem, run_request=run_request, linear=linear, responses=responses
        ),
        run_request,
    )


def z_table(march: str, tables: str) -> Built:
    """Return the gradient capillary under `march` with `tables` on the z table."""
    phi, prefactor = TABLES[tables]
    problem, run_request = tabulate(
        capillary(FIELD_GRID, MARCHES[march]), phi=phi, prefactor=prefactor
    )
    assert not problem.device_ready
    return Built(problem, run_request, FIELD_GRID)


def launch_phase(problem: PropagationProblem, run_request: Any, grid: Any) -> float:
    """Return the run's nonlinear phase ``L ||N(a_0)|| / ||a_0||`` by the reference.

    ``tests/march_bounds.py``'s ``measured_scales``, at the seam.
    """
    backend = NumpyRefBackend(problem, run_request, grid=grid)
    try:
        state = problem.initial.field0[None]
        backend._hoist(0.0)
        source = backend._nonlinear(
            state, backend._stage_scalars(0.0, 0.0), np.ones_like(state)
        )
    finally:
        backend.close()
    return Z_TABLE_LENGTH * float(np.linalg.norm(source) / np.linalg.norm(state))


# --- the bounds ----------------------------------------------------------------------


def _rows_in_y(problem: PropagationProblem) -> int:
    """Return the planar kind's y rows, its transverse transform's length."""
    geometry = problem.transform.geometry
    assert geometry is not None
    return int(geometry.shape[1])


TRANSVERSE: dict[str, Callable[[PropagationProblem], int]] = {
    "none": lambda problem: 1,
    "columns": lambda problem: problem.transform.n_nodes,
    "rows": _rows_in_y,
}
"""The transverse transform's length in the parity bound: none, the free kinds'
columns, the planar kind's y rows."""


def parity(
    *, growth: float = 2.0, transverse: str = "none", sums: bool = False
) -> Callable[[Built], float]:
    """Return the bound ``low_storage_cases.parity_bound`` with the case's sizes.

    `growth` is one where there is no source; `transverse` names the transverse
    transform (:data:`TRANSVERSE`); `sums` adds the response's cumulative sums over
    ``n_to`` (a plasma's prefix).
    """

    def bound(built: Built) -> float:
        problem = built.problem
        return parity_bound(
            problem,
            built.run_request,
            growth=growth,
            transverse=TRANSVERSE[transverse](problem),
            depth=problem.grids.n_to if sums else 0,
        )

    return bound


def general_bound(name: str) -> Callable[[Built], float]:
    """Return the general route's ten-step bound with RK4's weights, which sum to one.

    ``exp(Phi) (Phi source + parity_bound)``: the route's source bound
    (``general_parity.source_bound``) at the scaled launch's conditioning, carried by
    the run's nonlinear scale ``Phi``.
    """

    def bound(built: Built) -> float:
        grid, _, _, state = _scaled(name)
        maps = _general_maps(built.problem)
        assert maps is not None
        phi = nonlinear_scale(maps, state, grid)
        source = general_source_bound(maps, grid.n_to, conditioning(maps, state, grid))
        return math.exp(phi) * (
            phi * source + parity_bound(built.problem, built.run_request, growth=1.0)
        )

    return bound


def dispersive_bound(built: Built) -> float:
    """Return four of the dispersive response's evaluations a step, and the parity.

    ``dispersive_route_cases.evaluation_bound`` per evaluation.
    """
    from backends.dispersive_route_cases import evaluation_bound  # noqa: PLC0415

    return 4 * STEPS * evaluation_bound(built.problem) + parity_bound(
        built.problem, built.run_request
    )


def z_table_bound(built: Built) -> float:
    """Return the matched-march budget at the reference's launch phase, at least 0.1."""
    phase = launch_phase(built.problem, built.run_request, built.grid)
    return march_bound(
        built.problem, built.run_request, phi=max(phase, 0.1), backend="native"
    )


# --- the guards ----------------------------------------------------------------------


def without_collar(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem without its in-stage temporal collar's term."""
    return dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses, temporal_absorber=None, temporal_absorber_fine=None
        ),
    )


def without_y_collar(problem: PropagationProblem) -> PropagationProblem:
    """Return a planar problem without its y collar's in-stage rate.

    SEP 40's real-space collar in the y room, the geometry's absorber rate, is zeroed;
    the safety ramp and the temporal collar stay.
    """
    transform = problem.route.transform
    geometry = transform.geometry
    assert geometry is not None
    quiet = dataclasses.replace(
        geometry, absorber_rate=np.zeros_like(geometry.absorber_rate)
    )
    return dataclasses.replace(
        problem,
        route=dataclasses.replace(
            problem.route,
            transform=dataclasses.replace(transform, geometry=quiet),
        ),
    )


def without_raman_and_plasma(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem with no Raman rows and no plasma density."""
    return without_plasma(without_raman(problem))


def without_loss(problem: PropagationProblem) -> PropagationProblem:
    """Return :func:`small_loss`'s problem rebuilt with its loss at zero."""
    rebuilt = small_loss(0.0).problem
    assert rebuilt.shape == problem.shape
    return rebuilt


def without_dispersion(problem: PropagationProblem) -> PropagationProblem:
    """Return the linear fibre's problem rebuilt with no group-velocity dispersion."""
    rebuilt = linear_fibre(collar=False, beta2=0.0).problem
    assert rebuilt.shape == problem.shape
    return rebuilt


def _with_separable(problem: PropagationProblem, **changes: Any) -> PropagationProblem:
    """Return a separable problem with its operator's fields changed everywhere.

    In the operator, in the responses' copy and in the flux plan's base, wherever the
    problem holds it.
    """
    separable = problem.linear.separable
    assert separable is not None
    flat = dataclasses.replace(separable, **changes)
    route = problem.route
    flux = flux_of(route)
    if flux is not None:
        route = dataclasses.replace(route, flux=dataclasses.replace(flux, base=flat))
    return dataclasses.replace(
        problem,
        linear=dataclasses.replace(problem.linear, separable=flat),
        responses=dataclasses.replace(problem.responses, separable=flat),
        route=route,
    )


def without_frame(problem: PropagationProblem) -> PropagationProblem:
    """Return a separable problem whose flow no longer subtracts its moving frame.

    The fold of a linear Cartesian step is the flow ``exp(i (k_z - frame) h)`` alone;
    without the frame each column turns by ``k_0 h`` more a step.
    """
    separable = problem.linear.separable
    assert separable is not None
    return _with_separable(problem, frame=np.zeros_like(separable.frame))


def without_confinement(problem: PropagationProblem) -> PropagationProblem:
    """Return a planar problem whose slab modes have no transverse wavenumber in x.

    ``kappa_p^2``, each channel's confinement between the plates, is zeroed: the flow
    keeps the y diffraction and the frame.
    """
    separable = problem.linear.separable
    assert separable is not None and separable.kappa_sq is not None
    return _with_separable(problem, kappa_sq=np.zeros_like(separable.kappa_sq))


def without_modal_flow(problem: PropagationProblem) -> PropagationProblem:
    """Return a constant modal operator's problem with its exponent zeroed."""
    linear = problem.linear
    assert linear.phi_representation == "constant"
    assert linear.phi_constant is not None
    return dataclasses.replace(
        problem,
        linear=dataclasses.replace(
            linear, phi_constant=np.zeros_like(linear.phi_constant)
        ),
    )


def without_general_source(name: str) -> Callable[..., Any]:
    """Return a switch-off rebuilding a general case with its route's responses off."""

    def switch(problem: PropagationProblem) -> PropagationProblem:
        rebuilt = general(name, 1e-3, linear=True).problem
        assert rebuilt.shape == problem.shape
        return rebuilt

    return switch


def frozen_table(tables: str) -> Callable[[PropagationProblem], PropagationProblem]:
    """Return a switch-off that freezes the tabulated quantities at the launch.

    ``Phi`` becomes ``z D(0)`` and the prefactor its launch value, each on a z table
    of the same tolerance: the table's variation along z, which the row's march must
    read at every stage, is gone.
    """
    phi, prefactor = TABLES[tables]

    def switch(problem: PropagationProblem) -> PropagationProblem:
        linear, responses = problem.linear, problem.responses
        if phi:
            rate = np.asarray(phi_rate(linear, 0.0))
            linear = _phi_table(linear, lambda z: z * rate)
        if prefactor:
            launch = np.asarray(prefactor_at(responses, 0.0))
            responses = _prefactor_table(responses, lambda z: launch)
        return replace_groups(problem, linear=linear, responses=responses)

    return switch


# --- the rows ------------------------------------------------------------------------

QUIET = (
    # The cases launch noise-free into time windows kept tight for speed, which hold
    # more than a millionth of the energy in their guards or collar: the boundary
    # monitor warns (SEP 21), and the window's edge is not what these rows hold.
    "ignore::solprop.exceptions.BoundaryWarning",
    # The planar cases are argon-filled; the FFT padding reaches beyond the fit.
    "ignore::solprop.exceptions.MaterialRangeWarning",
    "ignore:the temporal absorber attenuates a packet walking off"
    ":solprop.exceptions.SolpropWarning",
)
"""The warning filters the rows' runs need."""


def session_row(
    *,
    name: str,
    route: str,
    stepper: str,
    response: str,
    features: str,
    build: Callable[[], Built],
    **fields: Any,
) -> Row:
    """Return a row of a native march's session; it marches, so it is medium.

    Unless it is slow by cost (``slow``, the reason). Its id is `name`; `fields` are
    the rest of :class:`parity_ladder.Row`.
    """
    return Row(
        id=name,
        axes={
            "route": route,
            "stepper": stepper,
            "response": response,
            "features": features,
        },
        build=build,
        medium=not fields.get("slow"),
        quiet=QUIET,
        **fields,
    )


MODES: tuple[PropagatorMode, ...] = ("tabulate", "recompute")
"""The march's two propagator modes: flows held, or formed per element."""

LINE_FAMILIES = {
    "field_kerr": ("kerr", "field grid, analytic family", without_kerr),
    "envelope_raman": ("raman", "envelope grid", without_raman),
    "field_plasma": ("plasma", "field grid, held current", without_plasma),
    "field_kerr_collar": ("kerr", "field grid, 0.3 collar in-stage", without_kerr),
}
"""Each line family's response, features and the switch-off of its term."""

PLANAR_CASES = {
    "kerr": ("kerr", "six y rows", without_kerr),
    "kerr-collar": ("kerr", "y collar, temporal collar in-stage", without_kerr),
    "kerr-split": ("kerr", "y collar, split-step temporal collar", without_kerr),
    "collar-only": ("none", "y collar and in-stage temporal collar", without_y_collar),
    "raman": ("raman", "six y rows", without_raman),
    "field-plasma": ("plasma", "field grid", without_plasma),
    "envelope-plasma": ("plasma", "envelope grid", without_plasma),
}
"""Each planar case's response, features and the switch-off of its term."""

PLANAR_VACUOUS = {
    "envelope-plasma": "The envelope plasma of this weak launch moves the reference's "
    "ten steps by 1.9e-13 of the field, 1.9e-4 of the bound (measured): the old test "
    "held it as a case the march must carry, and it is kept, but nothing here sees the "
    "plasma (#806). Its Kerr term moves the run by 7.6e-2 (8e7 bounds).",
}
"""The planar cases whose term reaches none of the row's rungs, with what was
measured."""

PLANAR_SLOW_BY_COST = {
    "envelope-plasma": "slow by cost (#792): ~105 s on CI macOS (44 s serially on the "
    "Mac); a parity check, not a witness",
}
"""The planar cases whose low-storage row is slow by cost, and why (as #805 tiered
the old test's case)."""

TABLE_WORDS = {
    "both": "the operator and the prefactor",
    "phi": "the operator",
    "prefactor": "the prefactor",
}
"""What each table variant puts on the table, in words."""


def _soliton_rows() -> tuple[Row, ...]:
    """Return the mode-averaged soliton's rows, plain and noisy, in both modes."""
    return tuple(
        session_row(
            name=f"low storage soliton {mode} {'modified noise' if noisy else 'plain'}",
            route="gnlse",
            stepper="low storage",
            response="kerr",
            features=f"{mode}d flows" + (", modified noise" if noisy else ""),
            why="The low-storage march carries the third-order soliton over half a "
            "dispersion length as the reference does"
            + (", the noise drawn at each pass's own position." if noisy else "."),
            build=partial(soliton, mode, noisy=noisy),
            switch_off=without_kerr,
            rungs={"steps": parity()},
        )
        for mode in MODES
        for noisy in (False, True)
    )


def _modal_rows() -> tuple[Row, ...]:
    """Return the modal route's rows: no source, a varying operator, a z table."""
    return (
        session_row(
            name="low storage fibre linear",
            route="gnlse",
            stepper="low storage",
            response="none",
            features="no source",
            why="With no response the low-storage march is the reference's linear "
            "flow on the modal route.",
            build=partial(linear_fibre, collar=False),
            switch_off=without_dispersion,
            rungs={"steps": parity(growth=1.0)},
        ),
        session_row(
            name="low storage fibre in-stage collar",
            route="gnlse",
            stepper="low storage",
            response="none",
            features="the in-stage collar's term alone",
            why="With the in-stage collar's term as its whole source the low-storage "
            "march is the reference's on the modal route.",
            build=partial(linear_fibre, collar=True),
            switch_off=without_collar,
            rungs={"steps": parity(growth=1.0)},
        ),
        *(
            session_row(
                name=f"low storage varying capillary {mode}",
                route="capillary (pressure gradient)",
                stepper="low storage",
                response="kerr, raman, plasma",
                features=f"{mode}d flows, varying operator, split-step collar",
                why="The low-storage march carries a nitrogen gradient capillary with "
                f"Raman and plasma, a varying operator whose flows are {mode}d, as "
                "the reference does.",
                build=partial(varying, mode),
                switch_off=without_raman_and_plasma,
                rungs={"steps": parity(sums=True)},
            )
            for mode in MODES
        ),
        *(
            session_row(
                name=f"low storage z table operator {mode}",
                route="gnlse",
                stepper="low storage",
                response="kerr",
                features=f"{mode}d flows, dispersion declared along z (z table)",
                why="A dispersion declared along z takes the operator's z table, whose "
                "held difference the low-storage march's flows are exponentiated "
                "from as the reference's are.",
                build=partial(tabled_fibre, mode),
                switch_off=frozen_table("both"),
                rungs={"steps": parity()},
                vacuous="The table's variation along z is the term, and it reaches "
                "nothing: the frame follows beta_0 and beta_1 along the path and the "
                "fibre has no beta_2, so the tabulated Phi is 3.2e-12 rad over the "
                "run, "
                "and freezing it and the prefactor at the launch moves the reference's "
                "ten steps by 0.95 of the bound (measured). Kept as the old test's one "
                "low-storage run on a z table (#806); the Kerr term moves the run by "
                "2.8e11 bounds, and the capillary's z-table rows carry the guard.",
            )
            for mode in MODES
        ),
    )


def _general_rows() -> tuple[Row, ...]:
    """Return the general modal route's rows: two cases, its collar, a linear guide."""
    return (
        *(
            session_row(
                name=f"low storage general {name}",
                route="general guide (constant)",
                stepper="low storage",
                response="kerr" if name == "pair" else "kerr, raman",
                features=f"general modal case {name!r}, coarse source",
                why=f"On the general modal route's {name} case the low-storage march "
                "is the reference's, on the case's own ten-step bound.",
                build=partial(general, name, 1e-3),
                switch_off=without_general_source(name),
                rungs={"steps": general_bound(name)},
            )
            for name in ("pair", "raman")
        ),
        session_row(
            name="low storage general collar",
            route="general guide (constant)",
            stepper="low storage",
            response="kerr",
            features="the general route's legacy transform for the in-stage collar",
            why="The general route's in-stage collar rides the low-storage march as "
            "on the reference.",
            build=partial(general, "pair", 2e-5, collar=True),
            switch_off=without_collar,
            rungs={"steps": general_bound("pair")},
        ),
        session_row(
            name="low storage general linear",
            route="general guide (constant)",
            stepper="low storage",
            response="none",
            features="no source at all",
            why="A general guide with no response is the reference's linear flow "
            "under the low-storage march.",
            build=partial(general, "pair", 2e-5, linear=True),
            switch_off=without_modal_flow,
            rungs={"steps": parity(growth=1.0)},
        ),
    )


def _spatial_rows() -> tuple[Row, ...]:
    """Return the radial, Cartesian, line and planar rows."""
    radial_rows = tuple(
        session_row(
            name=f"low storage radial {family}",
            route="radial",
            stepper="low storage",
            response="kerr",
            features=f"{family} family, in-stage spatial collar",
            why="On the radial Hankel route the low-storage march carries the Kerr "
            "column and the composed spatial collar as the reference does.",
            build=partial(radial, family),
            switch_off=without_kerr,
            rungs={"steps": parity()},
        )
        for family in ("envelope", "field_analytic")
    )
    cartesian_rows = tuple(
        session_row(
            name=f"low storage cartesian {mode} "
            f"{'in-stage collar' if collar else 'no collar'} "
            f"{'modified noise' if noisy else 'plain'}",
            route="cartesian",
            stepper="low storage",
            response="kerr",
            features=f"{mode}d flows"
            + (", in-stage collars" if collar else "")
            + (", modified noise" if noisy else ""),
            why="The low-storage march carries a Cartesian Kerr beam in chunks of 7 of "
            "its 108 columns as the reference does, with the exchange of SEP 24 "
            "section 4.3"
            + (" beside the spatial and temporal collars" if collar else "")
            + (" and the fused noise" if noisy else "")
            + ".",
            build=partial(cartesian, mode, collar=collar, noisy=noisy),
            switch_off=without_kerr,
            rungs={"steps": parity(transverse="columns")},
        )
        for mode in MODES
        for collar in (False, True)
        for noisy in (False, True)
    )
    line_rows = tuple(
        session_row(
            name=f"low storage line {family}",
            route="line",
            stepper="low storage",
            response=response,
            features=features,
            why=f"The low-storage march carries the line kind's {family} beam through "
            "the Cartesian kernels as the reference does.",
            build=partial(line, family),
            switch_off=switch_off,
            rungs={
                "steps": parity(transverse="columns", sums=family.endswith("plasma"))
            },
        )
        for family, (response, features, switch_off) in LINE_FAMILIES.items()
    )
    planar_rows = tuple(
        session_row(
            name=f"low storage planar {name}",
            route="planar",
            stepper="low storage",
            response=response,
            features=features + ", node chunks of two",
            why=f"The low-storage march folds both halves of the planar {name} case's "
            "column volume at once as the reference does.",
            build=partial(planar, name),
            switch_off=switch_off,
            rungs={"steps": parity(transverse="rows", sums="plasma" in name)},
            window=STEPS,
            vacuous=PLANAR_VACUOUS.get(name, ""),
            slow=PLANAR_SLOW_BY_COST.get(name, ""),
        )
        for name, (response, features, switch_off) in PLANAR_CASES.items()
    )
    return (*radial_rows, *cartesian_rows, *line_rows, *planar_rows)


def _remaining_rows() -> tuple[Row, ...]:
    """Return the remaining branches: a small loss, the flows, the dispersive term."""
    return (
        session_row(
            name="low storage cartesian recomputed small loss",
            route="cartesian",
            stepper="low storage",
            response="kerr",
            features="recomputed flows, a loss of 1e-3 1/m",
            why="A recomputed flow keeps a loss of a thousandth of a neper per metre: "
            "only an exact zero's magnitude is one.",
            build=small_loss,
            switch_off=without_loss,
            rungs={"steps": parity(transverse="columns")},
        ),
        session_row(
            name="low storage cartesian linear",
            route="cartesian",
            stepper="low storage",
            response="none",
            features="no response, no collar: the fold alone",
            why="With no response and no collar the low-storage march's pass is the "
            "fold alone, the reference's linear Cartesian flow.",
            build=cartesian_flow,
            switch_off=without_frame,
            rungs={"steps": parity(growth=1.0)},
        ),
        session_row(
            name="low storage cartesian dispersive response",
            route="cartesian",
            stepper="low storage",
            response="dispersive kerr",
            features="SEP 6 symmetric evaluator, begun once per pass",
            why="SEP 6's dispersive response rides the low-storage march on the "
            "elided Cartesian route as on the reference.",
            build=dispersive,
            switch_off=without_response,
            rungs={"steps": dispersive_bound},
        ),
        session_row(
            name="low storage planar linear",
            route="planar",
            stepper="low storage",
            response="none",
            features="no response, no collar: the fold of a zero source",
            why="With neither a response nor a collar the low-storage march on the "
            "planar kind is the reference's linear flow.",
            build=planar_flow,
            switch_off=without_confinement,
            rungs={"steps": parity(growth=1.0, transverse="rows")},
            window=STEPS,
        ),
    )


def _z_table_rows() -> tuple[Row, ...]:
    """Return the general z table's rows: each march, each table."""
    return tuple(
        session_row(
            name=f"z table {march} {tables}",
            route="capillary (pressure gradient)",
            stepper={"adams": "adams", "runge_kutta": "rk"}.get(march, "low storage"),
            response="kerr",
            features=f"{tables} on the z table, varying operator",
            why=f"Ten fixed steps of the {march.replace('_', '-')} march read "
            f"{TABLE_WORDS[tables]} from native's resident z table as the reference "
            "reads its own.",
            build=partial(z_table, march, tables),
            switch_off=frozen_table(tables),
            rungs={"steps": z_table_bound},
        )
        for march in MARCHES
        for tables in TABLES
    )


@functools.cache
def rows() -> tuple[Row, ...]:
    """Return every row of the low-storage march and the general z table."""
    return (
        *_soliton_rows(),
        *_modal_rows(),
        *_general_rows(),
        *_spatial_rows(),
        *_remaining_rows(),
        *_z_table_rows(),
    )
