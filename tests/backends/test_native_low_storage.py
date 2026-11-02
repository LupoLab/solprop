# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The low-storage RK4IP march's resident engine on native CPU: SEP 24 7.2 and 7.3.

Phase 3 runs the march in solprop-native: one native call a step, the four register
passes in the route's own scratch, accepted in place. Its parity with ``numpy_ref`` runs
on every route the translation admits the march on, each case one schedule of at most
ten steps within a rounding bound fixed before the comparison
(:func:`low_storage_cases.parity_bound`). Then what is the engine's own: the ledger and
the estimate, which agree to the byte, the varying operator's flows, the field-free
plane held in the native memory space and its resume, the in-flight guard, and the
commit a signal waits for. No agreement here is bitwise (SEP 25): the resume meets the
uninterrupted run within the parity bound.
"""

import dataclasses
import functools
import math
import signal
import warnings
from pathlib import Path

import numpy as np
import pytest

from backends.low_storage_cases import (
    STEPS,
    parity_bound,
    replay,
    uniform,
    varying_capillary,
    with_mode,
)
from backends.low_storage_cases import assert_compiled as compiled_against_reference
from backends.low_storage_cases import assert_parity as parity_against_reference
from backends.matched import native_thread_fields
from backends.planar_cases import planar_case
from backends.test_low_storage import MARCH, cartesian_march, marched
from backends.test_radial_nonlinear import CHI3, QuadraticRate
from backends.test_radial_nonlinear import LENGTH as RADIAL_LENGTH
from backends.test_radial_nonlinear import make_problem as radial_problem
from bounds import assert_within
from general_parity import conditioning as general_conditioning
from general_parity import nonlinear_scale, problem_for, scaled_case, source_bound
from solprop._interrupts import handled
from solprop.backends.memory import (
    estimate_memory,
    low_storage_flows,
    low_storage_stage_register,
)
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.backend import _general_maps
from solprop.boundaries import absorber
from solprop.constants import C
from solprop.driver import Simulation
from solprop.exceptions import BackendError, GeneralModalBasisWarning
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.noise import modified_noise, spatial_noise
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.nonlinear.raman import DampedOscillator, RamanResponse, undamped_dephasing
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.output.memory import MemoryOutput
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialFlux, SpatialMedium
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
    problem = soliton_problem(
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
    assert_parity(problem, narrow_grid, noise=noise)


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
    problem = linear_problem(
        narrow_grid,
        save_planes=[length],
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        schedule=schedule,
        integrator=MARCH,
        **extra,
    )
    assert_parity(problem, narrow_grid, growth=1.0)


@pytest.mark.medium
@pytest.mark.parametrize("mode", ["tabulate", "recompute"])
def test_a_varying_capillary_with_raman_and_plasma_agrees(mode):
    simulation = varying_capillary()
    problem = with_mode(simulation.translate(), mode)
    assert not problem.flags.constant_linop
    assert problem.flags.has_raman and problem.flags.has_plasma
    assert_parity(
        problem, simulation.grid, modes=simulation.modes, depth=problem.grids.n_to
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
    problem = with_mode(simulation.translate(), mode)
    assert problem.linear.phi_representation == "table"
    assert_compiled(
        problem, simulation.grid, parity_bound(problem), what=f"the z table, {mode}"
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
    space = RadialGrid(3e-4, 16, window_factor=0.5)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), 80e-6
    )
    problem = radial_problem(
        grid,
        space,
        pulse.build(grid, space, medium=SpatialMedium()),
        [KerrResponse(CHI3, thg=False)],
        steps=STEPS,
    )
    problem = marched(problem, problem.stepper.schedule)
    assert problem.transform.geometry is not None
    assert_parity(problem, grid)


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
        problem = problem_for(
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
    limit = math.exp(phi) * (phi * source + parity_bound(problem, growth=1.0))
    assert_compiled(problem, grid, limit, what="general route")


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
                FreeGrid(3e-4, 9, 4e-4, 12, window_factor=0.25 if collar else 0),
                SpatialMedium(),
            ),
            np.random.default_rng(7),
            model="modified",
        )
        if noisy
        else None
    )
    problem, grid, space = cartesian_march(mode, noise=noise, collar=collar)
    if noisy:
        # A beam beside the draw, so that the Kerr source carries it.
        pulse = SpatialPulse(
            GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), 80e-6
        )
        problem = dataclasses.replace(
            problem,
            initial=dataclasses.replace(
                problem.initial,
                field0=pulse.build(grid, space, medium=SpatialMedium())[None],
            ),
        )
    assert_parity(problem, grid, noise=noise, transverse=problem.transform.n_nodes)


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
    space = LineGrid(3e-4 if collar else 4e-4, 24, window_factor=0.3 if collar else 0.0)
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
    problem = radial_problem(
        grid,
        space,
        pulse.build(grid, space, medium=SpatialMedium()),
        responses,
        steps=STEPS,
        density=density,
    )
    problem = marched(problem, problem.stepper.schedule)
    if collar:
        problem = dataclasses.replace(
            problem,
            responses=dataclasses.replace(
                problem.responses, absorber_scheme="nonlinear"
            ),
        )
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=7)
    )
    assert_parity(problem, grid, transverse=problem.transform.n_nodes, depth=depth)


PLANAR = (
    "kerr",
    "kerr-collar",
    "kerr-split",
    "collar-only",
    pytest.param("raman", marks=pytest.mark.medium),
    pytest.param("field-plasma", marks=pytest.mark.medium),
    pytest.param("envelope-plasma", marks=pytest.mark.medium),
)


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.filterwarnings(
    "ignore:the temporal absorber attenuates a packet walking off"
    ":solprop.exceptions.SolpropWarning"
)
@pytest.mark.parametrize("name", PLANAR)
def test_a_planar_beam_agrees(name):
    """The hybrid planar kind: both halves of the column volume folded at once."""
    grid, _, problem = planar_case(name, steps=STEPS, node_chunk=2)
    problem = marched(problem, problem.stepper.schedule)
    depth = grid.n_to if "plasma" in name else 0
    geometry = problem.transform.geometry
    assert geometry is not None
    n_y = geometry.shape[1]
    assert_parity(problem, grid, transverse=n_y, depth=depth)


# --- the remaining branches of each route -------------------------------------------


def test_a_linear_cartesian_beam_agrees():
    """No response and no collar: the pass is the fold alone, no transform."""
    problem, grid, _ = cartesian_march("tabulate")
    problem = dataclasses.replace(
        problem,
        # Without the Kerr coefficients too, which a compiled backend may read.
        responses=dataclasses.replace(
            problem.responses,
            response_order=problem.responses.response_order[-1:],
            kerr_coefficients=None,
        ),
    )
    assert len(problem.responses.response_order) == 1
    assert low_storage_stage_register(problem, "native")
    assert_parity(problem, grid, growth=1.0)


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
    problem = free_problem(
        [DispersiveKerrResponse(ARGON, "symmetric", thg=False)], grid, adams=False
    )
    problem = marched(problem, problem.stepper.schedule)
    assert problem.responses.dispersive_kerr is not None
    limit = 4 * STEPS * evaluation_bound(problem) + parity_bound(problem)
    assert_compiled(problem, grid, limit, what="dispersive response")


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.filterwarnings(
    "ignore:the temporal absorber attenuates a packet walking off"
    ":solprop.exceptions.SolpropWarning"
)
def test_a_planar_beam_with_neither_term_is_the_flow():
    """No response and no collar on the planar kind: the fold of a zero source."""
    grid, _, problem = planar_case("kerr", steps=STEPS)
    problem = marched(problem, problem.stepper.schedule)
    problem = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses,
            response_order=problem.responses.response_order[-1:],
            kerr_coefficients=None,
            temporal_absorber=None,
            temporal_absorber_fine=None,
        ),
    )
    geometry = problem.transform.geometry
    assert geometry is not None
    n_y = geometry.shape[1]
    assert_parity(problem, grid, transverse=n_y, growth=1.0)


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
        problem = problem_for(
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
    limit = parity_bound(problem, growth=1.0)
    if variant == "collar":
        maps = _general_maps(problem)
        assert maps is not None
        phi = nonlinear_scale(maps, state, grid)
        source = source_bound(maps, grid.n_to, general_conditioning(maps, state, grid))
        limit = math.exp(phi) * (phi * source + limit)
    assert_compiled(problem, grid, limit, what=f"general route, {variant}")


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
    problem, grid, _ = cartesian_march(mode, collar=collar)
    field = 16 * math.prod(problem.shape)
    backend = NativeBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    backend.propagate_to(float(problem.stepper.save_planes[-1]))
    ledger = backend.memory_ledger()
    volumes = {row: ledger[row] for row in MARCH_ROWS}
    assert volumes == {
        "state": field,
        "transverse_scratch": field,
        "register_accumulator": field,
        "register_stage": field if collar else 0,
        "propagators": field if mode == "tabulate" else 0,
    }
    assert low_storage_stage_register(problem, "native") == collar
    assert low_storage_flows(problem, "native") == (mode == "tabulate")
    assert sum(volumes.values()) == field * (3 + int(collar) + (mode == "tabulate"))
    for row in (*RUNGE_KUTTA_ROWS, "noise", "held_plane", "adams_history"):
        assert ledger[row] == 0, row
    estimate = estimate_memory(problem, "native")
    assert estimate.groups["adams_history"] == 0
    assert backend._low_storage is not None
    engine = backend._low_storage.engine
    assert engine.atomic and math.isnan(engine.stiffness)
    metadata = backend.metadata()
    schedule = problem.stepper.schedule
    assert schedule is not None
    assert metadata["propagator_mode"] == mode
    assert metadata["source_evaluations"] == 4 * schedule.n_steps
    backend.close()


@pytest.mark.parametrize("collar", [False, True], ids=["no collar", "in-stage collar"])
@pytest.mark.parametrize("mode", ["tabulate", "recompute"])
def test_the_estimate_is_the_ledger_to_the_byte(mode, collar):
    """The march's field-sized rows of the estimate, row by row, as numpy_ref counts.

    The estimate's base is the reference's rows less the elided register; its march
    rows -- the state, the scratch, the registers and the flows -- are the native
    ledger's own, to the byte.
    """
    from solprop.backends.memory import (  # noqa: PLC0415
        reference_base_memory,
    )

    problem, grid, _ = cartesian_march(mode, collar=collar)
    field = 16 * math.prod(problem.shape)
    reference = reference_base_memory(
        problem,
        tabulating=mode == "tabulate",
        embedded=False,
        noise=False,
        statistics_weight=0,
        spatial_absorber=0,
    )
    backend = NativeBackend(problem, grid=grid)
    ledger = backend.memory_ledger()
    backend.close()
    for row in MARCH_ROWS:
        expected = 0 if row == "register_stage" and not collar else reference[row]
        assert ledger[row] == expected, row
    estimate = estimate_memory(problem, "native")
    assert estimate.groups["base"] == sum(reference.values()) - (0 if collar else field)


def test_the_low_storage_estimate_counts_the_elided_register():
    """Q4 in the admission's figure: the march's estimate without ``C`` where elided."""
    from solprop.backends.memory import low_storage_memory  # noqa: PLC0415

    for collar in (False, True):
        problem, _, _ = cartesian_march("tabulate", collar=collar)
        field = 16 * math.prod(problem.shape)
        under_march = estimate_memory(problem, "native").total
        assert low_storage_memory(problem, "native") == under_march
        # The same problem under the Runge-Kutta march: the march's figure is its own.
        runge_kutta = dataclasses.replace(
            problem,
            stepper=dataclasses.replace(
                problem.stepper, integrator=None, propagator_mode="auto"
            ),
        )
        assert low_storage_memory(runge_kutta, "native") == under_march
        held = low_storage_stage_register(problem, "native")
        assert held == collar
        assert low_storage_memory(problem, "numpy_ref") - under_march == (
            0 if held else field
        )


def test_a_varying_operator_holds_its_pair_tabulated_and_none_recomputed():
    """A varying modal operator's ``P`` and ``Q``: two volumes, or formed per element.

    Recomputed, the flows are formed in the finish from the phase terms, so nothing
    of the state's size is held for them: no transient outside the ledger.
    """
    simulation = varying_capillary()
    for mode, held in (("tabulate", 2), ("recompute", 0)):
        problem = with_mode(simulation.translate(), mode)
        backend = NativeBackend(problem, grid=simulation.grid, modes=simulation.modes)
        field = 16 * math.prod(problem.shape)
        ledger = backend.memory_ledger()
        assert ledger["propagators"] == held * field
        assert low_storage_flows(problem, "native") == held
        # The modal route holds C: its input is a padded fine spectrum.
        assert ledger["register_stage"] == field
        assert all(ledger[row] == 0 for row in RUNGE_KUTTA_ROWS)
        backend.close()


def test_a_field_free_plane_is_held_natively_and_resumes_to_rounding():
    """SEP 22's held plane in the native memory space: one volume, then one copy.

    Nothing crosses at the field-free plane; the ledger names the held volume; a
    resume from it meets the uninterrupted run within the parity bound (the table is
    built again from the resumed run's own step, section 4.3).
    """
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    space = FreeGrid(3e-4, 9, 4e-4, 12, window_factor=0)
    noise = spatial_noise(
        SpatialFlux.resolve(grid, space, SpatialMedium()),
        np.random.default_rng(7),
        model="modified",
    )
    problem, grid, _ = cartesian_march("tabulate", noise=noise)
    field = 16 * math.prod(problem.shape)
    end = float(problem.stepper.save_planes[-1])
    schedule = problem.stepper.schedule
    assert schedule is not None
    middle = float(schedule.edges[2])
    backend = NativeBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, noise)
    assert backend.memory_ledger()["noise"] == field
    backend.propagate_to(middle)
    assert backend.held_observation() is None
    assert backend.memory_ledger()["held_plane"] == 0
    plane = backend.observe(field=False)
    assert plane.field is None
    assert backend.memory_ledger()["held_plane"] == field
    held = backend.held_observation()
    assert held is not None and held.noise_phase is not None and held.z == middle
    assert held.field is not None and held.field.shape == problem.shape
    backend.propagate_to(end)
    uninterrupted = backend.observe().require_field().copy()
    # A plane that keeps its field releases the hold; the volume stays charged.
    assert backend.held_observation() is None
    assert backend.memory_ledger()["held_plane"] == field
    backend.close()
    resumed = NativeBackend(problem, grid=grid)
    resumed.set_initial(held.field, noise, z=held.z)
    resumed.propagate_to(end)
    got = resumed.observe().require_field()
    resumed.close()
    assert_within(got, uninterrupted, parity_bound(problem), what="resume")


def test_the_soliton_resumes_from_a_held_plane_on_its_reference(
    soliton_problem, narrow_grid, scales
):
    """The modal route's held plane, and its resume against the reference's run."""
    length = scales.dispersion_length
    schedule = uniform(length, 20, save_planes=[length / 2, length])
    problem = soliton_problem(
        narrow_grid,
        2,
        save_planes=[length / 2, length],
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        schedule=schedule,
        integrator=MARCH,
    )
    backend = NativeBackend(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    backend.propagate_to(length / 2)
    backend.observe(field=False)
    held = backend.held_observation()
    backend.close()
    assert held is not None and held.field is not None
    resumed = NativeBackend(problem, grid=narrow_grid)
    resumed.set_initial(held.field, None, z=held.z)
    resumed.propagate_to(length)
    got = resumed.observe().require_field()
    resumed.close()
    expected = replay(NumpyRefBackend, problem, narrow_grid)
    assert_within(got, expected, parity_bound(problem), what="resume")


def test_a_native_exception_leaves_the_backend_refusing(monkeypatch):
    """Section 6.2's guard: a step that raises after its passes stays in flight."""
    problem, grid, _ = cartesian_march("tabulate")
    end = float(problem.stepper.save_planes[-1])
    backend = NativeBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    assert backend._low_storage is not None
    engine = backend._low_storage.engine
    original = engine.step
    calls = []

    def failing(z, dz):
        calls.append(z)
        original(z, dz)
        if len(calls) == 2:
            raise RuntimeError("abandoned")
        return 1.0

    monkeypatch.setattr(engine, "step", failing)
    with pytest.raises(RuntimeError, match="abandoned"):
        backend.propagate_to(end)
    for entry in (backend.observe, backend.march_snapshot):
        with pytest.raises(BackendError, match="in flight"):
            entry()
    with pytest.raises(BackendError, match="in flight"):
        backend.propagate_to(end)
    # The abandoned step's start was the first edge: the guard names it.
    schedule = problem.stepper.schedule
    assert schedule is not None
    assert backend._in_flight == (calls[-1], float(schedule.steps[1]))
    backend.set_initial(problem.initial.field0, None)
    backend.observe()
    backend.close()


@pytest.mark.skipif(
    not hasattr(signal, "SIGINT"), reason="Ctrl-C is delivered by a POSIX signal"
)
def test_a_signal_during_the_native_step_waits_for_its_commit(monkeypatch):
    """Section 6.7: the native step, the finish and the move commit together.

    A Ctrl-C raised while the engine steps is held to the end of the commit: the
    step lands, the backend is not in flight, and the run can be observed.
    """
    problem, grid, _ = cartesian_march("tabulate")
    end = float(problem.stepper.save_planes[-1])
    backend = NativeBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    assert backend._low_storage is not None
    engine = backend._low_storage.engine
    original = engine.step

    def interrupted(z, dz):
        squared = original(z, dz)
        # Ctrl-C in-process; os.kill would terminate the process on Windows.
        signal.raise_signal(signal.SIGINT)
        return squared

    monkeypatch.setattr(engine, "step", interrupted)
    with handled(), pytest.raises(KeyboardInterrupt):
        backend.propagate_to(end)
    assert backend._in_flight is None
    schedule = problem.stepper.schedule
    assert schedule is not None
    assert backend._z == float(schedule.edges[1])
    assert np.all(np.isfinite(backend.observe().require_field()))
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
    for name in names:
        mode, collar = name.split("-")
        problem, _, _ = cartesian_march(mode, collar=collar == "collar")
        bound = parity_bound(problem, transverse=problem.transform.n_nodes)
        assert_within(again[name], one[name], bound, what=f"repeat, {name}")
        assert_within(four[name], one[name], bound, what=f"four threads, {name}")
