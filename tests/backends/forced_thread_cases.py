# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The cases ``test_native_forced_threads.py`` runs at each thread width.

Each builder runs one problem on ``native`` in the probe's process and returns the
arrays to compare and the run's ``execution_sizing``. They keep the inputs of the
``numba_cpu`` thread tests SEP 43 retired, and they are here rather than beside the
tests so that a probe does not import a pytest module with its fixtures.
"""

import dataclasses
import itertools
import warnings
from typing import Any, cast

import numpy as np

import flow_parity
from backends.dispersive_cases import capillary as dispersive_capillary
from backends.matched import fixed_order
from backends.planar_cases import planar_case
from capability_routes import ROUTES
from sessions.runs import bare, load_native, native_trial, run
from solprop.backends.events import ObservationReady
from solprop.backends.ledger import CallLedger
from solprop.backends.session import Start
from solprop.boundaries import absorber
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid
from solprop.modes.capillary import MarcatiliMode
from solprop.noise import spatial_noise
from solprop.nonlinear.responses import KerrResponse
from solprop.profiles import PressureGradient
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialFlux, SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.stepper.multistep import Multistep
from solprop.translate.assembly import assemble
from solprop.translate.problem import Translation
from solprop.translate.run_request import RunRequest

CAPILLARY_ROW_NAMES = ("energy", "peak_power", "fwhm_t")
"""The statistics the capillary's rows carry, as the retired rows test read them."""


def _native():
    """Return the optional wheel's backend class, for the probe's process alone."""
    # Optional wheel: absent from the base environment.
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        NativeBackend,
    )

    return NativeBackend


def _sizing(backend) -> dict[str, dict[str, int]]:
    return dict(backend.metadata()["execution_sizing"])


def route_adams_problem(route: str, formulation: str):
    """Return a route under Adams at ``rtol`` 1e-8: simulation, problem and request.

    The retired test's input is ``ROUTES["cartesian"]`` with one formulation named; the
    other routes are the same builder's, for what it could not reach.
    """
    simulation = ROUTES[route](None)
    settings = simulation.settings
    simulation.settings = type(settings)(
        **{
            **{name: getattr(settings, name) for name in settings.__dataclass_fields__},
            "controller": PIController(
                rtol=1e-8, atol=cast("PIController", settings.controller).atol
            ),
            "integrator": Multistep(formulation=formulation),  # pyright: ignore[reportArgumentType]
        }
    )
    return simulation, simulation.translate(), simulation._run_request()


def ten_steps(run_request: RunRequest) -> RunRequest:
    """Return `run_request` on ten equal prescribed steps over its length."""
    end = float(run_request.stepper.save_planes[-1])
    return run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper,
            schedule=FixedSchedule.uniform(0.0, end, end / 10 * 1.0000001),
        )
    )


def adams_matched(option: str):
    """Ten fixed steps at the method's top order, ``"<route>|<formulation>"``."""
    route, formulation = option.split("|")
    simulation, problem, run_request = route_adams_problem(route, formulation)
    run_request = fixed_order(ten_steps(run_request))
    backend = _native()(
        problem, run_request, grid=simulation.grid, modes=simulation.modes
    )
    try:
        # No statistics, as the direct march evaluated none (SEP 47 phase 5).
        result = run(backend, problem, bare(run_request))
        return {"field": result.fields[-1]}, _sizing(backend)
    finally:
        backend.close()


def adams_route(option: str):
    """March a route of ``capability_routes`` adaptively, ``<route>|<formulation>``.

    The native session's Adams march is the resident one: its calls are the session's
    ledger, and its steps every attempt the run made.
    """
    route, formulation = option.split("|")
    simulation, problem, run_request = route_adams_problem(route, formulation)
    backend = _native()(
        problem, run_request, grid=simulation.grid, modes=simulation.modes
    )
    record: dict[str, object] = {}
    try:
        # No statistics, as the direct march evaluated none (SEP 47 phase 5).
        result = run(backend, problem, bare(run_request), record=record)
        calls = CallLedger.from_record(
            cast("dict[str, object]", record["response_calls"])
        )
        return {
            "field": result.fields[-1],
            "calls": np.array(calls.total),
            "dz": np.asarray(result.joined[0].dz).copy(),
        }, _sizing(backend)
    finally:
        backend.close()


def field_grid() -> FieldGrid:
    """Return the field grid of ``tests/backends/conftest.py``."""
    return FieldGrid.from_limits(0.2, 800e-9, (200e-9, 4e-6), 100e-15)


def replayed_capillary(names=None) -> Translation:
    """Return the gradient capillary rung on its fixed 0.5 mm schedule.

    The fixture ``capillary_problem`` of ``tests/backends/conftest.py`` and the retired
    ``_replayed_capillary``: a Marcatili mode with a two-point gradient, so the operator
    varies along z and 0.2 m is a hard kink, the real cubic and the in-stage temporal
    absorber. The run to the first plane is 200 Runge-Kutta steps.
    """
    grid = field_grid()
    mode = MarcatiliMode(
        125e-6, "He", PressureGradient.two_point(0.2, 2.0, 0.2), cladding=1.45
    )
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-6).build(grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem, run_request = assemble(
            grid,
            mode,
            [KerrResponse(chi3=2.2e-52, thg=True)],
            start,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[0.1, 0.2],
            dz_max=0.01,
            dz_init=1e-4,
            absorber=absorber(grid),
        )
    stepper = dataclasses.replace(
        run_request.stepper,
        schedule=FixedSchedule.uniform(0.0, 0.2, 0.0005, save_planes=[0.1, 0.2]),
    )
    fixed = run_request.with_stepper(stepper)
    if names is not None:
        fixed = fixed.for_run(stats=StatsSpec(names=names, periodic=names, period=1))
    return Translation(problem, fixed)


def capillary():
    """Replay the capillary to its first plane, with the statistics rows on."""
    problem, run_request = replayed_capillary(CAPILLARY_ROW_NAMES)
    backend = _native()(problem, run_request, grid=field_grid())
    try:
        # The launch and the first plane: the session closes there.
        result = run(backend, problem, observations=2)
        observed, attempts, rows = result.assembled()[1]
        assert isinstance(observed, ObservationReady)
        assert observed.z == 0.1
        arrays = {
            "field": result.fields[1],
            "steps": np.array(attempts.n_steps),
            "z": np.asarray(rows.z).copy(),
            "step": np.asarray(rows.step).copy(),
        }
        arrays |= {
            f"row_{name}": np.asarray(rows.values[name]).copy()
            for name in CAPILLARY_ROW_NAMES
        }
        return arrays, _sizing(backend)
    finally:
        backend.close()


def dispersive(option: str):
    """Ten fixed-order steps of the 33-node capillary, ``"<evaluator>-<raman|plain>"``.

    The retired test's input, run twice in this process (it held a repeat too).
    """
    evaluator, kind = option.split("-")
    raman = kind == "raman"
    simulation = dispersive_capillary(
        evaluator, gas="N2" if raman else "Ar", raman=raman, stepped=True
    )
    problem = simulation.translate()
    run_request = fixed_order(simulation._run_request())
    assert problem.transform.n_nodes == 33
    fields = []
    for _ in range(2):
        backend = _native()(
            problem,
            run_request,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        )
        try:
            fields.append(run(backend, problem).fields[-1])
            sizing = _sizing(backend)
        finally:
            backend.close()
    return {"field": fields[0], "again": fields[1]}, sizing  # pyright: ignore[reportPossiblyUnboundVariable]


def with_adams(run_request: RunRequest) -> RunRequest:
    """Return a separable problem's request under the default Adams method."""
    return run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper, integrator=Multistep(), propagator_mode="recompute"
        )
    )


def planar(march: str):
    """Replay ``kerr-collar`` at a fixed order, twice, ``rk`` or ``adams``."""
    grid, _, problem, run_request = planar_case("kerr-collar")
    run_request = fixed_order(
        with_adams(run_request) if march == "adams" else run_request
    )
    fields = []
    for _ in range(2):
        backend = _native()(problem, run_request, grid=grid)
        try:
            fields.append(run(backend, problem).fields[-1])
            sizing = _sizing(backend)
        finally:
            backend.close()
    return {"field": fields[0], "again": fields[1]}, sizing  # pyright: ignore[reportPossiblyUnboundVariable]


def noisy_problem(march: str, *, draw: bool = True):
    """Return a Cartesian Kerr beam beside a seeded modified-noise draw.

    ``cartesian_march``'s problem under the low-storage march, as
    ``test_native_low_storage.py``'s noisy parity builds it, with the draw of the
    retired combine test's seed; `march` ``"rk"`` takes the Runge-Kutta march on the
    schedule instead, ``"adams"`` the exponential Adams method at its top order.
    """
    from backends.low_storage_cases import cartesian_march  # noqa: PLC0415

    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    noise = spatial_noise(
        SpatialFlux.resolve(
            grid, FreeGrid(3e-4, 9, 4e-4, 12, collar_fraction=0), SpatialMedium()
        ),
        np.random.default_rng(32),
        model="modified",
    )
    problem, run_request, grid, space = cartesian_march(
        "tabulate", noise=noise if draw else None
    )
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
    if march != "march":
        method = None if march == "rk" else Multistep(formulation="exponential")
        run_request = fixed_order(
            run_request.with_stepper(
                dataclasses.replace(
                    run_request.stepper, integrator=method, propagator_mode="recompute"
                )
            )
        )
    return problem, run_request, grid, noise if draw else None


def noisy(march: str):
    """March the beam under the modified noise model, and once without its draw."""
    from backends.low_storage_cases import CARTESIAN_CHUNK  # noqa: PLC0415

    fields = {}
    for name, draw in (("field", True), ("quiet", False)):
        problem, run_request, grid, noise = noisy_problem(march, draw=draw)
        backend = _native()(problem, run_request, grid=grid, options=CARTESIAN_CHUNK)
        try:
            start = Start.entrance(problem, noise=noise)
            fields[name] = run(backend, problem, start=start).fields[-1]
            if draw:
                sizing = _sizing(backend)
        finally:
            backend.close()
    return fields, sizing  # pyright: ignore[reportPossiblyUnboundVariable]


def wide_transverse_problem(march: str = "march"):
    """Return ``cartesian_march``'s problem on a transverse plane of 24 x 32 columns.

    The retired ``test_whole_runs_agree_at_1_2_and_4_threads`` replayed this march with
    the transverse pool forced. Native sizes its transverse plan from the elements a
    transform holds, and the march's own 9 x 12 plane gives 10,368, below
    ``threaded_fft_min_samples`` (16,384): this plane holds enough to thread (#322).
    `march` ``"adams"`` takes the default method at its top order on the same schedule,
    which #322 asks for beside the low-storage march.
    """
    from backends.low_storage_cases import MARCH  # noqa: PLC0415

    length = 1e-3
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
    )
    space = FreeGrid(3e-4, 24, 4e-4, 32, collar_fraction=0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), 80e-6
    )
    schedule = FixedSchedule.uniform(0.0, length, length / 4)
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
        integrator=MARCH if march == "march" else Multistep(),
        propagator_mode="tabulate" if march == "march" else "recompute",
    )
    return problem, fixed_order(run_request), grid


def wide_transverse(march: str = "march"):
    """Replay `march` on the plane whose transverse transform can use its pool."""
    problem, run_request, grid = wide_transverse_problem(march)
    backend = _native()(problem, run_request, grid=grid)
    try:
        # No statistics, as the direct march evaluated none (SEP 47 phase 5).
        result = run(backend, problem, bare(run_request))
        return {"field": result.fields[-1]}, _sizing(backend)
    finally:
        backend.close()


ACCEPT_STEP = 5e-6
"""The trial's step, m: inside the in-stage collar's stability limit of 5.2e-6."""


def accept_problem():
    """Return the Cartesian route on the Runge-Kutta march, one fixed step.

    The collar is in the stage because a split one acts after the accept's partials are
    read, so that the state they reduce and the state a host can read are the same.
    """
    simulation = ROUTES["cartesian"](None)
    problem = simulation.translate()
    run_request = simulation._run_request()
    return (
        simulation,
        dataclasses.replace(
            problem,
            responses=dataclasses.replace(
                problem.responses, absorber_scheme="nonlinear"
            ),
        ),
        run_request.with_stepper(
            dataclasses.replace(
                run_request.stepper,
                integrator=None,
                save_planes=np.array([ACCEPT_STEP]),
                schedule=FixedSchedule(np.array([ACCEPT_STEP])),
            )
        ),
    )


def accept():
    """One trial's partials beside the states they reduce, the launch and the result.

    The trial is the backend's native state's own, loaded with the launch.
    """
    simulation, problem, run_request = accept_problem()
    backend = _native()(problem, run_request, grid=simulation.grid)
    try:
        state = load_native(backend, problem.initial.field0)
        launch = state.field_to_host().copy()
        partials = native_trial(backend, 0.0, ACCEPT_STEP)
        state.finish(ACCEPT_STEP)
        return {
            "launch": launch,
            "partials": partials,
            "advanced": state.field_to_host().copy(),
            "mask": np.asarray(problem.grids.norm_mask),
        }, _sizing(backend)
    finally:
        backend.close()


def linear_cartesian(mode: str):
    """Return ``cartesian_march``'s beam with no response: a pass is its fold."""
    from backends.low_storage_cases import cartesian_march  # noqa: PLC0415

    problem, run_request, grid, _ = cartesian_march(mode)
    return (
        dataclasses.replace(
            problem,
            responses=dataclasses.replace(
                problem.responses,
                kerr_coefficients=None,
            ),
        ),
        run_request,
        grid,
    )


def folds(mode: str):
    """March the linear beam on the resident engine, saving every step's squared norm.

    Each step is the resident low-storage engine's on the backend's native state, as
    ``test_native_synthetic_operator.py``'s ``resident_step`` takes one, followed by the
    state's finish, as the march accepts a step. The squared norm over the norm mask is
    the one the engine's threaded reduction hands the accept, so a width's norms test
    that reduction's order, and its state the folds.
    """
    # Optional wheel: absent from the base environment.
    from solprop_native.backend import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        _ResidentLowStorage,  # pyright: ignore[reportPrivateUsage]
    )

    from backends.low_storage_cases import CARTESIAN_CHUNK, MARCH  # noqa: PLC0415

    problem, run_request, grid = linear_cartesian(mode)
    edges = np.asarray(run_request.stepper.schedule.edges)  # pyright: ignore[reportOptionalMemberAccess]
    backend = _native()(problem, run_request, grid=grid, options=CARTESIAN_CHUNK)
    try:
        state = load_native(backend, problem.initial.field0)
        engine = _ResidentLowStorage(backend, MARCH)
        norms = []
        for z, end in itertools.pairwise(edges):
            dz = float(end - z)
            norms.append(engine.step(float(z), dz))
            state.finish(dz)
        field = np.asarray(state.field_to_host()).reshape(problem.shape).copy()
        return {"field": field, "norms": np.array(norms)}, _sizing(backend)
    finally:
        backend.close()


def linear_flow_problem(name: str):
    """Return a coupled-flow case with no response, under the Runge-Kutta march.

    A linear run's trial is the flow's passes alone, with no transform to carry a
    rounding of its own: the combination, the accumulation and the accept. The collar
    goes to the split, so that the stage has no absorber term either, as
    ``flow_parity``'s no-source case does. One step of the case's schedule.
    """
    linear = flow_parity.build(name)
    # The entry point refuses a capillary with no nonlinearity.
    object.__setattr__(linear, "responses", ())
    problem = linear.translate()
    run_request = linear._run_request()
    schedule = run_request.stepper.schedule
    return (
        dataclasses.replace(
            problem,
            responses=dataclasses.replace(
                problem.responses, absorber_scheme="split_step"
            ),
        ),
        run_request.with_stepper(
            dataclasses.replace(run_request.stepper, integrator=None)
        ),
        linear.grid,
        float(schedule.steps[0]),  # pyright: ignore[reportOptionalMemberAccess]
    )


def flow(name: str):
    """One linear trial of a coupled flow: the partials and the state."""
    problem, run_request, grid, dz = linear_flow_problem(name)
    backend = _native()(problem, run_request, grid=grid)
    try:
        state = load_native(backend, problem.initial.field0)
        partials = native_trial(backend, 0.0, dz)
        state.finish(dz)
        return {
            "partials": partials,
            "field": state.field_to_host().copy(),
        }, _sizing(backend)
    finally:
        backend.close()


def flow_kerr(name: str):
    """Ten steps of a ``test_native_coupled_flow.py`` case, with its source."""
    from coupled_flow_cases import CASES as FLOWS  # noqa: PLC0415

    case = FLOWS[name]()
    problem = case.problem
    # The plain backend, kept open past its session so its sizing can be read (the
    # native session runs NativeBackend itself, never a subclass).
    backend = _native()(problem, case.run_request, grid=case.grid)
    try:
        result = run(
            backend,
            problem,
            bare(case.run_request),
            Start(field=problem.initial.field0, z=float(problem.initial.z)),
            owns=False,
        )
        sizing = _sizing(backend)
    finally:
        backend.close()
    return {"field": result.fields[-1].copy()}, sizing


CASES: dict[str, Any] = {
    "accept": accept,
    "adams_matched": adams_matched,
    "adams_route": adams_route,
    "capillary": capillary,
    "dispersive": dispersive,
    "flow": flow,
    "flow_kerr": flow_kerr,
    "folds": folds,
    "noisy": noisy,
    "planar": planar,
    "wide_transverse": wide_transverse,
}
"""Each builder by the name the probe takes, ``"<name>"`` or ``"<name>:<option>"``."""
