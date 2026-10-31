# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The low-storage march on the compiled backends against the reference: SEP 24 7.2.

Phase 2 runs the march through ``numba_cpu``'s register kernels on every route the
translation admits it on, and phase 3 through solprop-native's resident engine, whose
parity runs here on every case when it is installed (:func:`compiled_backends`). Each
case replays one schedule of at most ten steps on the reference and each compiled
backend, and the states agree within a rounding bound fixed before the comparison (SEP
25; :func:`parity_bound`): no agreement here is bitwise. The engine's own contracts --
its ledger, the threads, the held plane -- are each backend's own module's.
"""

import dataclasses
import itertools
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from numba import get_num_threads, set_num_threads

from backends.planar_cases import planar_case
from backends.test_low_storage import (
    ARGUMENT_FACTOR,
    MARCH,
    cartesian_march,
    marched,
    populated,
)
from backends.test_radial_nonlinear import CHI3, QuadraticRate
from backends.test_radial_nonlinear import LENGTH as RADIAL_LENGTH
from backends.test_radial_nonlinear import make_problem as radial_problem
from bounds import EPS, assert_within, departure, step_budget
from general_parity import conditioning as general_conditioning
from general_parity import nonlinear_scale, problem_for, scaled_case, source_bound
from solprop.backends.bodies import FOLD_COMPLETE, FOLD_EXCHANGE, FOLD_SECOND
from solprop.backends.memory import (
    estimate_memory,
    low_storage_stage_register,
    reference_base_memory,
)
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numba_cpu import low_storage as compiled
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
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialFlux, SpatialMedium
from solprop.stepper import FixedSchedule
from solprop.translate.ionisation import build_rate_table
from solprop.translate.linop import delta_phi

STEPS = 10
"""The qualification scope's propagation (AGENTS.md): ten steps."""

PRODUCTS = 32
"""Roundings of one evaluation outside its transforms, with a margin: the input's add
and scale, the prefactor, the window and the absorber's term, the fold's four products
and adds, the flow's exponential (a few eps), and the response's own few per sample."""

FFT_CONSTANT = 8
"""Roundings per level of a transform pair, Higham's ``log2 n`` with a margin."""


def parity_bound(problem, *, growth=2.0, transverse=1, depth=0):
    """Return the relative bound two backends' replays of one schedule must meet.

    Fixed before either is run, from the case's own sizes:

    - **the steps' rounding**, ``growth N 4 (PRODUCTS + FFT_CONSTANT log2(n_to n_x n_y)
      + depth) u``: four evaluations a step, each a temporal transform pair over
      ``n_to`` samples, a transverse pair over the free kinds' ``n_x n_y`` columns, a
      response whose cumulative sums have addition depth `depth` (the plasma's prefix
      over ``n_to``), and the fold;
    - **the flows' arguments**, section 4.5's ``c eps (N |D| h + |Phi(z_N)|)`` over the
      bins holding a thousandth of the launch's peak, since the backends form
      ``exp(h D / 2)`` from differently ordered exponents (the table on both, or the
      separable body against the reference's propagator), carried by the same growth;
      on the separable kinds the exponent is a principal root's, one ulp of ``k``
      rather than of ``D``, so ``2 eps k_max L`` over the run (the planar ladder's
      ``operator_bound``, ``tests/backends/test_numba_planar.py``).

    `growth` is the stated assumption on how much the run amplifies a perturbation
    over its ten steps (:func:`bounds.propagation_growth` of the nonlinear rate over
    the run is below 2 in every case here).
    """
    schedule = problem.stepper.schedule
    levels = math.log2(problem.grids.n_to) + math.log2(max(transverse, 1))
    per_step = 4 * (PRODUCTS + FFT_CONSTANT * levels + depth)
    rounding = step_budget(per_step, schedule.n_steps, growth)
    edges = schedule.edges
    swept = sum(
        np.abs(delta_phi(problem.linear, float(a), float(b)))
        for a, b in itertools.pairwise(edges)
    )
    final = np.abs(delta_phi(problem.linear, float(edges[0]), float(edges[-1])))
    argument = EPS * ARGUMENT_FACTOR * np.broadcast_to(swept + final, problem.shape)
    bins = populated([problem.initial.field0], 1e-3)
    separable = problem.linear.separable
    roots = (
        0.0
        if separable is None
        else 2
        * EPS
        * math.sqrt(float(np.max(np.abs(separable.k_squared[separable.active]))))
        * float(schedule.z_end - schedule.z_start)
    )
    return rounding + growth * (float(np.max(argument[bins])) + roots)


def replay(backend_class, problem, grid, *, modes=None, noise=None, **options):
    """Replay the problem's schedule through its planes; return the final state."""
    backend = backend_class(problem, grid=grid, modes=modes, **options)
    try:
        backend.set_initial(problem.initial.field0, noise)
        for plane in problem.stepper.save_planes:
            backend.propagate_to(float(plane))
        field = backend.observe().require_field().copy()
        assert np.all(np.isfinite(field))
        return field
    finally:
        backend.close()


def compiled_backends():
    """Return the compiled backends the march's parity runs on.

    ``numba_cpu`` and, when solprop-native is installed, its CPU build (SEP 24 phase
    3), each against ``numpy_ref``: both within one bound of the reference holds them
    within twice it of each other.
    """
    try:
        # The optional plugin extends the base NumPy/Numba qualification pair.
        from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            NativeBackend,
        )
    except ImportError:
        return (NumbaCpuBackend,)
    return (NumbaCpuBackend, NativeBackend)


def assert_compiled(problem, grid, limit, *, modes=None, noise=None, what="march"):
    """Replay on the reference and every compiled backend; assert each within `limit`.

    Returns the largest departure measured.
    """
    expected = replay(NumpyRefBackend, problem, grid, modes=modes, noise=noise)
    worst = 0.0
    for backend_class in compiled_backends():
        got = replay(backend_class, problem, grid, modes=modes, noise=noise)
        assert_within(
            got, expected, limit, what=f"{backend_class.name} against numpy_ref, {what}"
        )
        worst = max(worst, departure(got, expected))
    return worst


def assert_parity(problem, grid, *, modes=None, noise=None, **bound):
    """Replay on every backend and assert the states within :func:`parity_bound`."""
    limit = parity_bound(problem, **bound)
    return assert_compiled(problem, grid, limit, modes=modes, noise=noise), limit


def uniform(length, steps=STEPS, **planes):
    """Return a uniform schedule of `steps` steps over `length`."""
    return FixedSchedule.uniform(0.0, length, length / steps * 1.0000001, **planes)


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


def with_mode(problem, mode):
    """Return the problem with its propagator mode set."""
    return dataclasses.replace(
        problem,
        stepper=dataclasses.replace(problem.stepper, propagator_mode=mode),
    )


def varying_capillary():
    """Return a nitrogen gradient capillary with Raman and plasma under the march.

    A varying operator: the tabulated flows are the step's ``P`` and ``Q``, and the
    recomputed ones are formed in turn in one held scratch (SEP 24 section 4.3).
    """
    length = 0.02
    return Simulation.from_capillary(
        100e-6,
        length,
        "N2",
        (1.0, 0.7),
        wavelength=800e-9,
        fwhm=10e-15,
        energy=20e-6,
        trange=150e-15,
        wavelength_limits=(200e-9, 2e-6),
        raman=True,
        plasma=True,
        noise=None,
        n_save=2,
        schedule=uniform(length),
        dz_max=length / STEPS * 1.0000001,
        dz_init=length / STEPS,
        integrator=MARCH,
        backend="numpy_ref",
        # The in-stage collar would hold a step to 0.44 mm; the split-step one
        # exercises the accepted step's own treatment instead.
        absorber_scheme="split_step",
    )


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


def test_a_varying_operator_holds_its_pair_tabulated_and_one_scratch_recomputed(
    monkeypatch,
):
    """The modal flows held, and nothing state-sized allocated in a step beside them.

    Tabulating holds ``P`` and ``Q``; recomputing forms them in turn in one scratch,
    and the estimate counts it to the byte. In a step the engine allocates no complex
    array of the state's size, as the recomputed flows were, and never forms the
    exponent array (``_delta_phi``) the flows were exponentiated from.
    """
    from solprop.backends.memory import (  # noqa: PLC0415
        low_storage_flow_scratch,
        low_storage_memory,
    )

    simulation = varying_capillary()
    for mode, held in (("tabulate", 2), ("recompute", 1)):
        problem = with_mode(simulation.translate(), mode)
        assert low_storage_flow_scratch(problem, "numba_cpu") == (mode == "recompute")
        backend = NumbaCpuBackend(problem, grid=simulation.grid, modes=simulation.modes)
        field = 16 * math.prod(problem.shape)
        assert backend.memory_ledger()["propagators"] == held * field
        reference = reference_base_memory(
            problem,
            tabulating=mode == "tabulate",
            embedded=not np.array_equal(
                problem.transform.projection, problem.transform.projection_coarse
            ),
            noise=False,
            statistics_weight=0,
            spatial_absorber=0,
        )
        # The reference allocates its recomputed flows per call and counts none.
        scratch = field if mode == "recompute" else 0
        assert reference["propagators"] + scratch == held * field
        estimate = estimate_memory(problem, "numba_cpu")
        base = estimate_memory(problem, "numpy_ref").groups["base"]
        # SEP 27's compiled current streams have their own persistent layout.
        # Compare the remaining base to the reference, and count those streams
        # from the constructed backend rather than assuming equal workspaces.
        reference_current = sum(
            size for row, size in reference.items() if row.startswith("held_current_")
        )
        compiled_current = sum(
            size
            for row, size in backend.memory_ledger().items()
            if row.startswith("held_current_") and not row.endswith("heap_growth")
        )
        assert (
            estimate.groups["base"]
            == base - reference_current + compiled_current + scratch
        )
        # The admission's figure for the march, from the march or the default one.
        runge_kutta = dataclasses.replace(
            problem, stepper=dataclasses.replace(problem.stepper, integrator=None)
        )
        for asked in (problem, runge_kutta):
            assert low_storage_memory(asked, "numba_cpu") == estimate.total
        schedule = problem.stepper.schedule
        assert schedule is not None
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(schedule.edges[1])
        made: list[tuple[int, ...]] = []
        for name in ("empty", "zeros", "empty_like", "zeros_like"):
            original = getattr(np, name)

            def recorded(*args, _original=original, _made=made, _field=field, **kwargs):
                array = _original(*args, **kwargs)
                caller = sys._getframe(1).f_code.co_filename  # the allocating module
                if Path(caller) == Path(compiled.__file__) and array.nbytes == _field:
                    _made.append(array.shape)
                return array

            monkeypatch.setattr(np, name, recorded)

        def no_exponent(bundle):
            raise AssertionError(bundle)

        monkeypatch.setattr(backend, "_delta_phi", no_exponent)
        backend.propagate_to(schedule.edges[2])
        monkeypatch.undo()
        assert made == []
        backend.close()


@pytest.mark.parametrize("mode", ["tabulate", "recompute"])
def test_a_z_table_operator_agrees(mode):
    """A dispersion declared along z takes the operator's z table, and its flows.

    The modal flows are then exponentiated from the table's held difference rather
    than from the terms of ``Phi``: on ``numba_cpu`` the host table's, on the native
    CPU build the difference of two native slots (SEP 29 section 6.6).
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


# --- threads, the kernels and the engine --------------------------------------------


def test_the_parallel_kernels_agree_with_the_serial_ones_at_1_2_and_4_threads(
    thread_budget_lifted,
):
    """Each element's arithmetic is the same; only the norm's column sum may reorder.

    Every fold mode and flow treatment of the Cartesian kernels, and the modal and
    planar finishes, against their serial variants on the same arrays: within the
    norm's summation bound, ``n_columns`` roundings of the partial sum, and four
    roundings of an element.
    """
    rng = np.random.default_rng(11)
    problem, _, _ = cartesian_march("tabulate", collar=True)
    backend = NumbaCpuBackend(problem, grid=None)
    data = backend._separable
    grids = problem.grids
    shape = (backend._n_columns, backend._n_omega)
    backend.close()

    def complex_array():
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    start = [complex_array() for _ in range(5)]
    table = np.exp(1j * rng.standard_normal(shape))
    drawn = complex_array()
    passes = [
        (stage, mode, term, tabulated)
        for stage in range(4)
        for mode, term in (
            (FOLD_COMPLETE, 1),
            (FOLD_EXCHANGE, 1),
            (FOLD_SECOND, 2),
            (FOLD_COMPLETE, 0),
        )
        for tabulated in (True, False)
    ]
    threads = get_num_threads()

    def folded(parallel):
        kernels = compiled.make_free_kernels(noise=True, parallel=parallel)
        arrays = [values.copy() for values in start]
        partials = np.zeros(shape[0])
        norms = []
        for stage, mode, term, tabulated in passes:
            kernels.fold(
                data,
                (stage, mode, term),
                1e-6,
                arrays[3],
                grids.omega_window,
                grids.active_mask,
                (arrays[0], arrays[1], arrays[2]),
                table,
                tabulated,
                (drawn, 2e-4, 0.5, False),
                grids.norm_mask,
                partials,
            )
            norms.append(float(np.sum(partials)))
        kernels.input(data, arrays[2], drawn, 3e-4, 0.25, arrays[4])
        return arrays, norms

    serial, serial_norms = folded(False)
    try:
        for count in (1, 2, min(4, threads)):
            set_num_threads(count)
            parallel, parallel_norms = folded(True)
            for got, expected in zip(parallel, serial, strict=True):
                assert_within(got, expected, 4 * EPS, what=f"{count} threads")
            np.testing.assert_allclose(
                parallel_norms, serial_norms, rtol=shape[0] * EPS, atol=0
            )
    finally:
        set_num_threads(threads)


@pytest.mark.medium
def test_whole_runs_agree_at_1_2_and_4_threads(thread_budget_lifted):
    """The kernel and FFTW pools at 1, 2 and 4 threads: the same replay, to rounding.

    The large regime is asked for so the column loops are ``prange``; the transforms
    split their work by thread count, which reorders sums only, so the runs agree
    within the parity bound.
    """
    problem, grid, _ = cartesian_march("tabulate", collar=True)
    limit = parity_bound(problem, transverse=problem.transform.n_nodes)
    runs = [
        replay(
            NumbaCpuBackend,
            problem,
            grid,
            threads=count,
            fftw_threads=count,
            transverse_fftw_threads=count,
            regime="large",
        )
        for count in (1, 2, 4)
    ]
    for got in runs[1:]:
        assert_within(got, runs[0], limit, what="thread counts")


@pytest.mark.parametrize("collar", [False, True], ids=["no collar", "in-stage collar"])
@pytest.mark.parametrize("mode", ["tabulate", "recompute"])
def test_the_engine_holds_its_registers_and_the_table_and_releases_them(mode, collar):
    """Section 7.3 on ``numba_cpu``: the ledger names the registers and the table.

    Beside the state and the transverse scratch: the accumulator, the stage register
    under an in-stage collar only (Q4: without one each finish writes the next input
    into the scratch), and the table when tabulating -- so four volumes tabulated
    without a collar and three recomputed, five and four with one -- and none of the
    Runge-Kutta march's. The estimate is the ledger to the byte for these rows.
    """
    problem, grid, _ = cartesian_march(mode, collar=collar)
    field = 16 * math.prod(problem.shape)
    backend = NumbaCpuBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    backend.propagate_to(float(problem.stepper.save_planes[-1]))
    ledger = backend.memory_ledger()
    rows = (
        "state",
        "transverse_scratch",
        "register_accumulator",
        "register_stage",
        "propagators",
    )
    volumes = {row: ledger[row] for row in rows}
    assert volumes == {
        "state": field,
        "transverse_scratch": field,
        "register_accumulator": field,
        "register_stage": field if collar else 0,
        "propagators": field if mode == "tabulate" else 0,
    }
    assert low_storage_stage_register(problem, "numba_cpu") == collar
    assert sum(volumes.values()) == field * (2 + int(collar) + (mode == "tabulate") + 1)
    for row in (
        "stages",
        "accumulator_solution",
        "accumulator_error",
        "advanced",
        "exponent",
        "recycled_stage",
        "noise",
        "held_plane",
    ):
        assert ledger[row] == 0, row
    # Every other row is a chunk's, bounded by the node chunk, not the state.
    assert all(
        size < field
        for row, size in ledger.items()
        if row not in rows and not row.endswith("heap_growth")
    )
    reference = reference_base_memory(
        problem,
        tabulating=mode == "tabulate",
        embedded=False,
        noise=False,
        statistics_weight=0,
        spatial_absorber=0,
    )
    for row in (*rows, "stages", "accumulator_solution", "accumulator_error"):
        expected = 0 if row == "register_stage" and not collar else reference[row]
        assert ledger[row] == expected, row
    estimate = estimate_memory(problem, "numba_cpu")
    assert estimate.groups["adams_history"] == 0
    # The estimate's base is the reference's count less the elided register.
    assert estimate.groups["base"] == sum(reference.values()) - (0 if collar else field)
    assert backend._low_storage is not None
    engine = backend._low_storage.engine
    assert isinstance(engine, compiled.NumbaLowStorage)
    backend.close()
    assert set(engine.held().values()) == {0}


def test_the_noise_and_a_field_free_plane_add_one_volume_each():
    """The draw is one counted volume; the held plane one more (SEP 22, Q14)."""
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    space = FreeGrid(3e-4, 9, 4e-4, 12, window_factor=0)
    noise = spatial_noise(
        SpatialFlux.resolve(grid, space, SpatialMedium()),
        np.random.default_rng(7),
        model="modified",
    )
    problem, grid, _ = cartesian_march("tabulate", noise=noise)
    field = 16 * math.prod(problem.shape)
    backend = NumbaCpuBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, noise)
    backend.propagate_to(float(problem.stepper.save_planes[-1]) / 2)
    assert backend.memory_ledger()["noise"] == field
    assert backend.memory_ledger()["held_plane"] == 0
    plane = backend.observe(field=False)
    assert plane.field is None
    assert backend.memory_ledger()["held_plane"] == field
    held = backend.held_observation()
    assert held is not None and held.noise_phase is not None
    assert held.field is not None and held.field.shape == problem.shape
    backend.close()


def test_a_held_plane_resumes_to_rounding(soliton_problem, narrow_grid, scales):
    """A field-free plane's held state resumes and meets the uninterrupted run."""
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
    backend = NumbaCpuBackend(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    backend.propagate_to(length / 2)
    backend.observe(field=False)
    held = backend.held_observation()
    assert held is not None and held.field is not None
    backend.propagate_to(length)
    uninterrupted = backend.observe().require_field().copy()
    backend.close()
    resumed = NumbaCpuBackend(problem, grid=narrow_grid)
    resumed.set_initial(held.field, None, z=held.z)
    resumed.propagate_to(length)
    got = resumed.observe().require_field()
    resumed.close()
    # The same arithmetic on the same steps from the same state, but for the table,
    # which a resume builds again from its own step (section 4.3): the parity bound.
    assert_within(got, uninterrupted, parity_bound(problem), what="resume")


def test_an_abandoned_step_leaves_the_backend_refusing(monkeypatch):
    """Section 6.2's guard holds for the resident engine too."""
    problem, grid, _ = cartesian_march("tabulate")
    backend = NumbaCpuBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    assert backend._low_storage is not None
    engine = backend._low_storage.engine
    assert isinstance(engine, compiled.NumbaLowStorage)
    original = engine._free_pass
    calls = []

    def failing(*arguments):
        calls.append(arguments[0])
        if len(calls) == 6:
            raise RuntimeError("abandoned")
        return original(*arguments)

    monkeypatch.setattr(engine, "_free_pass", failing)
    with pytest.raises(RuntimeError, match="abandoned"):
        backend.propagate_to(float(problem.stepper.save_planes[-1]))
    with pytest.raises(BackendError, match="in flight"):
        backend.observe()
    with pytest.raises(BackendError, match="in flight"):
        backend.propagate_to(float(problem.stepper.save_planes[-1]))
    backend.set_initial(problem.initial.field0, None)
    backend.observe()
    backend.close()


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
    assert low_storage_stage_register(problem, "numba_cpu")
    assert_parity(problem, grid, growth=1.0)


@pytest.mark.medium
def test_the_dispersive_response_rides_the_cartesian_march():
    """SEP 6's dispersive response, begun once per pass, on the elided route."""
    from backends.test_numba_dispersive_routes import (  # noqa: PLC0415
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
    backend = NumbaCpuBackend(problem, grid=grid)
    assert not backend._nonlinear and not backend._in_stage_absorber
    backend.close()
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


def test_the_low_storage_estimate_counts_the_elided_register():
    """Q4 in the admission's figure: the march's estimate without ``C`` where elided."""
    from solprop.backends.memory import low_storage_memory  # noqa: PLC0415

    for collar in (False, True):
        problem, _, _ = cartesian_march("tabulate", collar=collar)
        field = 16 * math.prod(problem.shape)
        under_march = estimate_memory(problem, "numba_cpu").total
        assert low_storage_memory(problem, "numba_cpu") == under_march
        # The same problem under the Runge-Kutta march: the march's figure is its own.
        runge_kutta = dataclasses.replace(
            problem,
            stepper=dataclasses.replace(
                problem.stepper, integrator=None, propagator_mode="auto"
            ),
        )
        assert low_storage_memory(runge_kutta, "numba_cpu") == under_march
        held = low_storage_stage_register(problem, "numba_cpu")
        assert held == collar
        assert low_storage_memory(problem, "numpy_ref") - under_march == (
            0 if held else field
        )


def test_a_read_only_state_is_copied_before_the_step_writes_it():
    """An observation may leave the accepted state read-only; the march copies it."""
    problem, grid, _ = cartesian_march("tabulate")
    schedule = problem.stepper.schedule
    assert schedule is not None
    target = float(schedule.steps[0])
    results = []
    for frozen in (False, True):
        backend = NumbaCpuBackend(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            state = backend._state
            assert state is not None
            if frozen:
                state.setflags(write=False)
            backend.propagate_to(target)
            stepped = backend._state
            assert stepped is not None and stepped.flags.writeable
            results.append(np.array(stepped, copy=True))
        finally:
            backend.close()
    # The copy is taken before any arithmetic, so the two steps are the same step.
    np.testing.assert_array_equal(results[1], results[0])
