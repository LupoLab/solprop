# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The low-storage RK4IP march's resident engine on native CPU: SEP 24 7.2 and 7.3.

Phase 3 runs the march in solprop-native: one native call a step, the four register
passes in the route's own scratch, accepted in place. Its parity with ``numpy_ref`` runs
on every route the translation admits the march on, each case one schedule of at most
ten steps within a rounding bound fixed before the comparison
(:func:`low_storage_cases.parity_bound`). Then what is the engine's own: the ledger and
the estimate, which agree to the byte, the varying operator's flows, and the load. No
agreement here is bitwise (SEP 25). Every run is the native session's (SEP 47 phase
5), which owns the loop: a step left in flight and the poisoned session are
``tests/sessions/test_native_session_low_storage.py``'s, and a signal during a step,
which lands after its native call has committed it,
``tests/backends/test_low_storage.py``'s
``test_an_atomic_engine_commits_its_step_whole[atomic engine]``.
"""

import dataclasses
import functools
import gc
import math
import warnings
import weakref
from pathlib import Path

import numpy as np
import pytest

from backends.forced_threads import assert_threaded
from backends.low_storage_cases import (
    STEPS,
    parity_bound,
    uniform,
    varying_capillary,
    with_mode,
)
from backends.low_storage_cases import assert_compiled as compiled_against_reference
from backends.low_storage_cases import assert_parity as parity_against_reference
from backends.matched import native_thread_fields
from backends.planar_cases import planar_case
from backends.test_low_storage import CARTESIAN_CHUNK, MARCH, cartesian_march, marched
from backends.test_radial_nonlinear import CHI3, QuadraticRate
from backends.test_radial_nonlinear import LENGTH as RADIAL_LENGTH
from backends.test_radial_nonlinear import make_problem as radial_problem
from bounds import assert_within
from general_parity import conditioning as general_conditioning
from general_parity import nonlinear_scale, problem_for, scaled_case, source_bound
from sessions.runs import launch, load_native, run
from solprop.backends.memory import (
    low_storage_flows,
    low_storage_stage_register,
)
from solprop.backends.numpy_ref.backend import _general_maps
from solprop.backends.session import ExecutionOptions, Start
from solprop.backends.sessions import start_session
from solprop.boundaries import Absorber, absorber
from solprop.constants import C
from solprop.driver import Simulation
from solprop.exceptions import GeneralModalBasisWarning
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.noise import modified_noise, spatial_noise
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.nonlinear.raman import DampedOscillator, RamanResponse, undamped_dephasing
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.output.memory import MemoryOutput
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialFlux, SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import PIController
from solprop.translate.assembly import assemble
from solprop.translate.ionisation import build_rate_table

# The fixtures here launch noise-free into time windows kept tight for speed, which
# hold more than a millionth of the energy in their guards or collar: the boundary
# monitor warns (SEP 21), and the window's edge is not what this module tests.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

solprop_native = pytest.importorskip("solprop_native")
NativeBackend = solprop_native.NativeBackend

assert_compiled = functools.partial(compiled_against_reference, NativeBackend)
"""Replay on the reference and ``native``; assert it within the limit."""

assert_parity = functools.partial(parity_against_reference, NativeBackend)
"""Replay on both backends and assert the states within the parity bound."""

MARCH_ROWS = (
    "state",
    "transverse_scratch",
    "register_accumulator",
    "register_stage",
    "propagators",
)
"""The ledger's field-sized rows of the march on a Cartesian grid."""

PROBE = Path(__file__).with_name("native_low_storage_thread_probe.py")

RUNGE_KUTTA_ROWS = (
    "advanced",
    "stages",
    "accumulator_solution",
    "accumulator_error",
    "weighted_residual",
    "recycled_stage",
    "stage_inverse",
)
"""The Runge-Kutta march's native views, none allocated under the low-storage one."""


# --- the modal routes ---------------------------------------------------------------


@pytest.mark.parametrize("mode", ["tabulate", "recompute"])
@pytest.mark.parametrize("noisy", [False, True], ids=["plain", "modified noise"])
def test_the_soliton_agrees_with_the_reference(
    soliton_problem, narrow_grid, scales, mode, noisy
):
    """The mode-averaged soliton, the noise drawn at each pass's own position."""
    length = scales.dispersion_length / 2
    schedule = uniform(length)
    noise = modified_noise(narrow_grid, np.random.default_rng(3)) if noisy else None
    problem, run_request = soliton_problem(
        narrow_grid,
        3,
        save_planes=[length],
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        schedule=schedule,
        integrator=MARCH,
        propagator_mode=mode,
        noise=noise,
    )
    assert_parity(problem, run_request, narrow_grid, noise=noise)


@pytest.mark.parametrize("collar", [False, True], ids=["linear", "in-stage collar"])
def test_a_problem_without_a_response_agrees(linear_problem, narrow_grid, collar):
    """No source at all, or the in-stage collar's term alone, on the modal route."""
    extra = (
        {"absorber": absorber(narrow_grid), "absorber_scheme": "nonlinear"}
        if collar
        else {}
    )
    length = 0.05
    schedule = uniform(length)
    problem, run_request = linear_problem(
        narrow_grid,
        save_planes=[length],
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        schedule=schedule,
        integrator=MARCH,
        **extra,
    )
    assert_parity(problem, run_request, narrow_grid, growth=1.0)


@pytest.mark.medium
@pytest.mark.parametrize("mode", ["tabulate", "recompute"])
def test_a_varying_capillary_with_raman_and_plasma_agrees(mode):
    simulation = varying_capillary()
    # The states are compared, not the statistics, whose profiles a session would
    # need at construction.
    problem, run_request = with_mode(
        simulation.translate(),
        simulation._run_request().for_run(stats=StatsSpec()),
        mode,
    )
    assert not problem.flags.constant_linop
    assert problem.flags.has_raman and problem.flags.has_plasma
    assert_parity(
        problem,
        run_request,
        simulation.grid,
        modes=simulation.modes,
        depth=problem.grids.n_to,
    )


@pytest.mark.parametrize("mode", ["tabulate", "recompute"])
def test_a_z_table_operator_agrees(mode):
    """A dispersion declared along z takes the operator's z table, and its flows.

    The modal flows are then exponentiated from the table's held difference rather
    than from the terms of ``Phi``: on the native CPU build the difference of two
    native slots (SEP 29 section 6.6).
    """
    length = 1e-3
    grid = EnvelopeGrid.from_limits(length, 835e-9, (600e-9, 1300e-9), 2e-12)
    beta0 = 1.45 * 2 * np.pi / 835e-9
    beta1 = 1.47 / C
    nodes = np.array([0.0, length])
    # beta_1 rises by one per cent over the millimetre, the area by half.
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
    assert_compiled(
        problem,
        run_request,
        simulation.grid,
        parity_bound(problem, run_request),
        what=f"the z table, {mode}",
    )


@pytest.mark.parametrize("family", ["envelope", "field_analytic"])
def test_the_radial_route_with_its_in_stage_collar_agrees(family):
    """The radial Hankel route: the Kerr column and the composed spatial collar."""
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
    assert_parity(problem, run_request, grid)


@pytest.mark.parametrize(
    "name", ["pair", pytest.param("raman", marks=pytest.mark.medium)]
)
def test_the_general_modal_route_agrees(name):
    """A general modal guide's coarse source, on its case's own ten-step bound."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeneralModalBasisWarning)
        grid, guide, route, state = scaled_case(name)
        length = 1e-3
        schedule = uniform(length)
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
        )
    maps = _general_maps(problem)
    assert maps is not None
    phi = nonlinear_scale(maps, state, grid)
    source = source_bound(maps, grid.n_to, general_conditioning(maps, state, grid))
    # The general route's ten-step bound with RK4's weights, which sum to one.
    limit = math.exp(phi) * (
        phi * source + parity_bound(problem, run_request, growth=1.0)
    )
    assert_compiled(problem, run_request, grid, limit, what="general route")


# --- the Cartesian, line and planar routes ------------------------------------------


@pytest.mark.parametrize("mode", ["tabulate", "recompute"])
@pytest.mark.parametrize("collar", [False, True], ids=["no collar", "in-stage collar"])
@pytest.mark.parametrize("noisy", [False, True], ids=["plain", "modified noise"])
def test_a_cartesian_kerr_beam_agrees(mode, collar, noisy):
    """Every flow treatment, with the exchange of section 4.3 and the fused noise."""
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
        # A beam beside the draw, so that the Kerr source carries it.
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
    assert_parity(
        problem,
        run_request,
        grid,
        noise=noise,
        options=CARTESIAN_CHUNK,
        transverse=problem.transform.n_nodes,
    )


def test_a_recomputed_flow_keeps_a_small_loss():
    """Only an exact zero's magnitude is one: a real part past it takes its exponential.

    A loss of a thousandth of a neper per metre puts about 6e-8 in each flow's exponent,
    well above the parity bound and far below any bin a coarser test would notice.
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
            np.full(grid.n_omega, 1e-3),
            np.zeros(grid.n_t),
            np.zeros(grid.n_to),
            grid.active_mask,
            5 * length,
        ),
        absorber_scheme="nonlinear",
    )
    assert_parity(problem, run_request, grid, transverse=problem.transform.n_nodes)


LINE_FAMILIES = ("field_kerr", "envelope_raman", "field_plasma", "field_kerr_collar")


@pytest.mark.parametrize("family", LINE_FAMILIES)
def test_a_line_beam_agrees(family):
    """The line kind, ``n_y = 1``, through the Cartesian kernels."""
    field = family.startswith("field")
    # The held field-plasma source needs resolved collar samples; retain the
    # shorter lattice for the unrelated envelope, Kerr and Raman controls.
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
    density, depth = 1.0, 0
    if family.endswith("plasma"):
        responses = [
            PlasmaResponse("Ar", build_rate_table(QuadraticRate(), (1e3, 1e10)))
        ]
        density, depth = 1e22, grid.n_to
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
    assert_parity(
        problem,
        run_request,
        grid,
        options=CARTESIAN_CHUNK,
        transverse=problem.transform.n_nodes,
        depth=depth,
    )


PLANAR = (
    "kerr",
    "kerr-collar",
    "kerr-split",
    "collar-only",
    pytest.param("raman", marks=pytest.mark.medium),
    pytest.param("field-plasma", marks=pytest.mark.medium),
    # slow by cost (#792): ~105 s on CI macOS; a parity check, not a witness
    pytest.param("envelope-plasma", marks=pytest.mark.slow),
)


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.filterwarnings(
    "ignore:the temporal absorber attenuates a packet walking off"
    ":solprop.exceptions.SolpropWarning"
)
@pytest.mark.parametrize("name", PLANAR)
def test_a_planar_beam_agrees(name):
    """The hybrid planar kind: both halves of the column volume folded at once."""
    grid, _, problem, run_request = planar_case(name, steps=STEPS)
    problem, run_request = marched(problem, run_request, run_request.stepper.schedule)
    depth = grid.n_to if "plasma" in name else 0
    geometry = problem.transform.geometry
    assert geometry is not None
    n_y = geometry.shape[1]
    assert_parity(
        problem,
        run_request,
        grid,
        options=ExecutionOptions(node_chunk=2),
        transverse=n_y,
        depth=depth,
    )


# --- the remaining branches of each route -------------------------------------------


def test_a_linear_cartesian_beam_agrees():
    """No response and no collar: the pass is the fold alone, no transform."""
    problem, run_request, grid, _ = cartesian_march("tabulate")
    problem = dataclasses.replace(
        problem,
        # Without the Kerr coefficients too, which a compiled backend may read.
        responses=dataclasses.replace(
            problem.responses,
            kerr_coefficients=None,
        ),
    )
    assert len(problem.responses.response_order) == 1
    assert low_storage_stage_register(problem, "native")
    assert_parity(problem, run_request, grid, options=CARTESIAN_CHUNK, growth=1.0)


@pytest.mark.medium
def test_the_dispersive_response_rides_the_cartesian_march():
    """SEP 6's dispersive response, begun once per pass, on the elided route."""
    from backends.dispersive_route_cases import (  # noqa: PLC0415
        ARGON,
        evaluation_bound,
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
    limit = 4 * STEPS * evaluation_bound(problem) + parity_bound(problem, run_request)
    assert_compiled(problem, run_request, grid, limit, what="dispersive response")


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.filterwarnings(
    "ignore:the temporal absorber attenuates a packet walking off"
    ":solprop.exceptions.SolpropWarning"
)
def test_a_planar_beam_with_neither_term_is_the_flow():
    """No response and no collar on the planar kind: the fold of a zero source."""
    grid, _, problem, run_request = planar_case("kerr", steps=STEPS)
    problem, run_request = marched(problem, run_request, run_request.stepper.schedule)
    problem = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses,
            kerr_coefficients=None,
            temporal_absorber=None,
            temporal_absorber_fine=None,
        ),
    )
    geometry = problem.transform.geometry
    assert geometry is not None
    n_y = geometry.shape[1]
    assert_parity(problem, run_request, grid, transverse=n_y, growth=1.0)


@pytest.mark.parametrize("variant", ["collar", "linear"])
def test_the_general_route_s_collar_and_a_linear_guide_agree(variant):
    """The general route's legacy transform for the collar, or no source at all."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeneralModalBasisWarning)
        grid, guide, route, state = scaled_case("pair")
        length = 2e-5
        schedule = uniform(length)
        extra = (
            {"absorber": absorber(grid), "absorber_scheme": "nonlinear"}
            if variant == "collar"
            else {}
        )
        if variant == "linear":
            route = dataclasses.replace(route, responses=None)
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
            **extra,
        )
    limit = parity_bound(problem, run_request, growth=1.0)
    if variant == "collar":
        maps = _general_maps(problem)
        assert maps is not None
        phi = nonlinear_scale(maps, state, grid)
        source = source_bound(maps, grid.n_to, general_conditioning(maps, state, grid))
        limit = math.exp(phi) * (phi * source + limit)
    assert_compiled(problem, run_request, grid, limit, what=f"general route, {variant}")


# --- the engine's own --------------------------------------------------------------


@pytest.mark.parametrize("collar", [False, True], ids=["no collar", "in-stage collar"])
@pytest.mark.parametrize("mode", ["tabulate", "recompute"])
def test_the_native_state_holds_the_registers_and_no_runge_kutta_view(mode, collar):
    """Section 7.3 on Kokkos: three or four volumes beside the state, none of DOPRI5's.

    The accumulator, the stage register under an in-stage collar only (Q4), the table
    when tabulating, the state and the transverse scratch:
    four volumes tabulated without a collar and three recomputed, five and four with
    one. The estimate is the ledger to the byte for these rows.
    """
    problem, run_request, grid, _ = cartesian_march(mode, collar=collar)
    field = 16 * math.prod(problem.shape)
    backend = NativeBackend(problem, run_request, grid=grid, options=CARTESIAN_CHUNK)
    record: dict[str, object] = {}
    held: list[dict[str, int]] = []

    def at_end(_session, batch):
        # The ledger while the session holds the backend, at its terminal batch.
        if batch.terminal is not None:
            held.append(backend.memory_ledger())

    run(backend, problem, run_request, record=record, between=at_end)
    [ledger] = held
    assert ledger is not None
    volumes = {row: ledger[row] for row in MARCH_ROWS}
    assert volumes == {
        "state": field,
        "transverse_scratch": field,
        "register_accumulator": field,
        "register_stage": field if collar else 0,
        "propagators": field if mode == "tabulate" else 0,
    }
    assert low_storage_stage_register(problem, "native") == collar
    assert low_storage_flows(problem, run_request.stepper, "native") == (
        mode == "tabulate"
    )
    assert sum(volumes.values()) == field * (3 + int(collar) + (mode == "tabulate"))
    for row in (*RUNGE_KUTTA_ROWS, "noise", "adams_history"):
        assert ledger[row] == 0, row
    schedule = run_request.stepper.schedule
    assert schedule is not None
    assert record["propagator_mode"] == mode
    assert record["source_evaluations"] == 4 * schedule.n_steps
    backend.close()


def test_a_varying_operator_holds_its_pair_tabulated_and_none_recomputed():
    """A varying modal operator's ``P`` and ``Q``: two volumes, or formed per element.

    Recomputed, the flows are formed in the finish from the phase terms, so nothing
    of the state's size is held for them: no transient outside the ledger.
    """
    simulation = varying_capillary()
    for mode, held in (("tabulate", 2), ("recompute", 0)):
        problem, run_request = with_mode(
            simulation.translate(), simulation._run_request(), mode
        )
        backend = NativeBackend(
            problem, run_request, grid=simulation.grid, modes=simulation.modes
        )
        field = 16 * math.prod(problem.shape)
        ledger = backend.memory_ledger()
        assert ledger["propagators"] == held * field
        assert low_storage_flows(problem, run_request.stepper, "native") == held
        # The modal route holds C: its input is a padded fine spectrum.
        assert ledger["register_stage"] == field
        assert all(ledger[row] == 0 for row in RUNGE_KUTTA_ROWS)
        backend.close()


@pytest.mark.parametrize("model", [None, "input"], ids=["no noise", "input noise"])
def test_a_loaded_state_is_not_held_after_the_load(monkeypatch, model):
    """The upload copies and fences, so a temporary dies with the load; no ledger row.

    A delay scan's input, the input noise model's sum and a resumed plane are all
    temporaries: each would otherwise stay alive until the next load. The field is read
    back to show that the load took.
    """
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    noise = (
        None
        if model is None
        else spatial_noise(
            SpatialFlux.resolve(
                grid, FreeGrid(3e-4, 9, 4e-4, 12, collar_fraction=0), SpatialMedium()
            ),
            np.random.default_rng(7),
            model=model,
        )
    )
    problem, run_request, grid, _ = cartesian_march("tabulate", noise=noise)
    backend = NativeBackend(problem, run_request, grid=grid, options=CARTESIAN_CHUNK)
    temporary = np.full(problem.shape, 1.0 + 0.5j)
    given = weakref.ref(temporary)
    start = Start(field=temporary, z=0.0, noise=noise)
    del temporary
    session = start_session(backend, problem, run_request, start, owns=False)
    gc.collect()
    assert given() is None
    session.close()
    # A resumed plane, which takes no input draw: the field crosses unchanged.
    loaded_again = np.full(problem.shape, 2.0 + 0.5j)
    again = weakref.ref(loaded_again)
    start = Start(field=loaded_again, z=0.0, noise=noise, resume=True)
    del loaded_again
    observed = launch(backend, problem, run_request, start).field
    gc.collect()
    assert again() is None
    # Exact: the state crosses twice without arithmetic, and the mask keeps or zeroes.
    mask = problem.grids.active_mask
    assert np.all(observed[..., mask] == 2.0 + 0.5j)
    assert np.all(observed[..., ~mask] == 0.0)
    backend.close()


def test_auto_runs_the_march_on_native_when_installed():
    """``backend="auto"`` resolves to Kokkos for the march, as for every integrator."""
    from solprop.driver import Simulation  # noqa: PLC0415

    length = 0.01
    schedule = uniform(length, 5)
    simulation = Simulation.from_gnlse(
        0.11,
        length,
        [0.0, 0.0, -11.83e-27],
        wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=9.8,
        shape="sech",
        n_save=2,
        schedule=schedule,
        integrator=MARCH,
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        # Five replayed steps: no collar, as before the default one.
        absorber=False,
        noise=None,
    )
    assert simulation.settings.backend == "native"
    output = simulation.run()
    assert isinstance(output, MemoryOutput)
    assert output.outcome is not None
    assert output.outcome.execution["backend"] == "native"
    assert output.outcome.execution["source_evaluations"] == 4 * schedule.n_steps


# Medium, though ~135 s on CI macOS against ~6 s on Windows x64: #804.
@pytest.mark.medium
def test_the_march_agrees_across_thread_counts(tmp_path):
    """One thread twice and four threads, each in a fresh process (SEP 24 7.2).

    The same passes on the same schedule, differing by the order of their threaded
    transforms, folds and reductions alone: within the parity bound fixed for the
    case, as against the reference.
    """
    names = ["tabulate-plain", "recompute-collar"]
    one = native_thread_fields(PROBE, 1, names, tmp_path)
    again = native_thread_fields(PROBE, 1, names, tmp_path)
    four = native_thread_fields(PROBE, 4, names, tmp_path)
    assert_threaded(four["sizing"], 4, "kernels.stage", "fft.fft")
    for name in names:
        mode, collar = name.split("-")
        problem, run_request, _, _ = cartesian_march(mode, collar=collar == "collar")
        bound = parity_bound(problem, run_request, transverse=problem.transform.n_nodes)
        assert_within(again[name], one[name], bound, what=f"repeat, {name}")
        assert_within(four[name], one[name], bound, what=f"four threads, {name}")


def test_a_trial_step_on_a_low_storage_state_is_refused_by_name():
    # #492: this state allocates none of the views the trial step writes to.
    problem, run_request, grid, _ = cartesian_march("tabulate")
    backend = NativeBackend(problem, run_request, grid=grid, options=CARTESIAN_CHUNK)
    try:
        state = load_native(backend, problem.initial.field0, clear_fsal=False)
        backend._staged_rows()
        nodes = [float(node) for node in run_request.stepper.tableau.nodes]
        with pytest.raises(RuntimeError, match=r"low-storage state .* no Runge-Kutta"):
            state.rk_trial(0.0, run_request.stepper.dz_init, nodes, prepare_only=False)
    finally:
        backend.close()
