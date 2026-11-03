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
import warnings
from typing import Any, cast

import numpy as np

import flow_parity
from backends.dispersive_cases import capillary as dispersive_capillary
from backends.matched import fixed_order
from backends.planar_cases import planar_case
from capability_routes import ROUTES
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
from solprop.translate.problem import KernelProblem, assemble

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
    """Return a route under Adams, adaptive at ``rtol`` 1e-8, and its simulation.

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
    return simulation, simulation.translate()


def ten_steps(problem: KernelProblem) -> KernelProblem:
    """Return `problem` on ten equal prescribed steps over its length."""
    end = float(problem.stepper.save_planes[-1])
    return dataclasses.replace(
        problem,
        stepper=dataclasses.replace(
            problem.stepper,
            schedule=FixedSchedule.uniform(0.0, end, end / 10 * 1.0000001),
        ),
    )


def adams_matched(option: str):
    """Ten fixed steps at the method's top order, ``"<route>|<formulation>"``."""
    route, formulation = option.split("|")
    simulation, problem = route_adams_problem(route, formulation)
    problem = fixed_order(ten_steps(problem))
    backend = _native()(problem, grid=simulation.grid, modes=simulation.modes)
    try:
        backend.set_initial(problem.initial.field0, None)
        for plane in problem.stepper.save_planes:
            backend.propagate_to(float(plane))
        observed = backend.observe()
        return {"field": observed.require_field().copy()}, _sizing(backend)
    finally:
        backend.close()


def adams_route(option: str):
    """March a route of ``capability_routes`` adaptively, ``<route>|<formulation>``."""
    route, formulation = option.split("|")
    simulation, problem = route_adams_problem(route, formulation)
    backend = _native()(problem, grid=simulation.grid, modes=simulation.modes)
    try:
        march = backend._adams
        assert march is not None
        assert type(march._engine).__name__ == "_ResidentAdams"
        backend.set_initial(problem.initial.field0, None)
        for plane in problem.stepper.save_planes:
            backend.propagate_to(float(plane))
        observed = backend.observe()
        return {
            "field": observed.require_field().copy(),
            "calls": np.array(march.ledger.total),
            "dz": np.asarray(observed.step_history.dz).copy(),
        }, _sizing(backend)
    finally:
        backend.close()


def field_grid() -> FieldGrid:
    """Return the field grid of ``tests/backends/conftest.py``."""
    return FieldGrid.from_limits(0.2, 800e-9, (200e-9, 4e-6), 100e-15)


def replayed_capillary(names=None) -> KernelProblem:
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
        problem = assemble(
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
        problem.stepper,
        schedule=FixedSchedule.uniform(0.0, 0.2, 0.0005, save_planes=[0.1, 0.2]),
    )
    fixed = dataclasses.replace(problem, stepper=stepper)
    if names is not None:
        fixed = dataclasses.replace(
            fixed, stats_spec=StatsSpec(names=names, periodic=names, period=1)
        )
    return fixed


def capillary():
    """Replay the capillary to its first plane, with the statistics rows on."""
    problem = replayed_capillary(CAPILLARY_ROW_NAMES)
    backend = _native()(problem, grid=field_grid())
    try:
        backend.set_initial(problem.initial.field0, None)
        summary = backend.propagate_to(0.1)
        observed = backend.observe()
        rows = observed.step_stats
        arrays = {
            "field": observed.require_field().copy(),
            "steps": np.array(summary.n_steps),
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
    problem = fixed_order(simulation.translate())
    assert problem.transform.n_nodes == 33
    fields = []
    for _ in range(2):
        backend = _native()(problem, grid=simulation.grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(float(problem.stepper.save_planes[-1]))
            fields.append(backend.observe().require_field().copy())
            sizing = _sizing(backend)
        finally:
            backend.close()
    return {"field": fields[0], "again": fields[1]}, sizing


def with_adams(problem: KernelProblem) -> KernelProblem:
    """Return a separable problem under the default Adams method."""
    return dataclasses.replace(
        problem,
        stepper=dataclasses.replace(
            problem.stepper, integrator=Multistep(), propagator_mode="recompute"
        ),
    )


def planar(march: str):
    """Replay ``kerr-collar`` at a fixed order, twice, ``rk`` or ``adams``."""
    grid, _, problem = planar_case("kerr-collar")
    problem = fixed_order(with_adams(problem) if march == "adams" else problem)
    fields = []
    for _ in range(2):
        backend = _native()(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(grid.length)
            fields.append(backend.observe().require_field().copy())
            sizing = _sizing(backend)
        finally:
            backend.close()
    return {"field": fields[0], "again": fields[1]}, sizing


def noisy_problem(march: str, *, draw: bool = True):
    """Return a Cartesian Kerr beam beside a seeded modified-noise draw.

    ``cartesian_march``'s problem under the low-storage march, as
    ``test_native_low_storage.py``'s noisy parity builds it, with the draw of the
    retired combine test's seed; `march` ``"rk"`` takes the Runge-Kutta march on the
    schedule instead, ``"adams"`` the exponential Adams method at its top order.
    """
    from backends.test_low_storage import cartesian_march  # noqa: PLC0415

    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    noise = spatial_noise(
        SpatialFlux.resolve(
            grid, FreeGrid(3e-4, 9, 4e-4, 12, collar_fraction=0), SpatialMedium()
        ),
        np.random.default_rng(32),
        model="modified",
    )
    problem, grid, space = cartesian_march("tabulate", noise=noise if draw else None)
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
    if march != "march":
        method = None if march == "rk" else Multistep(formulation="exponential")
        problem = dataclasses.replace(
            problem,
            stepper=dataclasses.replace(
                problem.stepper, integrator=method, propagator_mode="recompute"
            ),
        )
        problem = fixed_order(problem)
    return problem, grid, noise if draw else None


def noisy(march: str):
    """March the beam under the modified noise model, and once without its draw."""
    fields = {}
    for name, draw in (("field", True), ("quiet", False)):
        problem, grid, noise = noisy_problem(march, draw=draw)
        backend = _native()(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, noise)
            backend.propagate_to(float(problem.stepper.save_planes[-1]))
            fields[name] = backend.observe().require_field().copy()
            if draw:
                sizing = _sizing(backend)
        finally:
            backend.close()
    return fields, sizing


def wide_transverse_problem(march: str = "march"):
    """Return ``cartesian_march``'s problem on a transverse plane of 24 x 32 columns.

    The retired ``test_whole_runs_agree_at_1_2_and_4_threads`` replayed this march with
    the transverse pool forced. Native sizes its transverse plan from the elements a
    transform holds, and the march's own 9 x 12 plane gives 10,368, below
    ``threaded_fft_min_samples`` (16,384): this plane holds enough to thread (#322).
    `march` ``"adams"`` takes the default method at its top order on the same schedule,
    which #322 asks for beside the low-storage march.
    """
    from backends.test_low_storage import MARCH  # noqa: PLC0415

    length = 1e-3
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
    )
    space = FreeGrid(3e-4, 24, 4e-4, 32, collar_fraction=0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), 80e-6
    )
    schedule = FixedSchedule.uniform(0.0, length, length / 4)
    problem = assemble(
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
    return fixed_order(problem), grid


def wide_transverse(march: str = "march"):
    """Replay `march` on the plane whose transverse transform can use its pool."""
    problem, grid = wide_transverse_problem(march)
    backend = _native()(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(float(problem.stepper.save_planes[-1]))
        return {"field": backend.observe().require_field().copy()}, _sizing(backend)
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
    return simulation, dataclasses.replace(
        problem,
        responses=dataclasses.replace(problem.responses, absorber_scheme="nonlinear"),
        stepper=dataclasses.replace(
            problem.stepper,
            integrator=None,
            save_planes=np.array([ACCEPT_STEP]),
            schedule=FixedSchedule(np.array([ACCEPT_STEP])),
        ),
    )


def accept():
    """One trial's partials beside the states they reduce, the launch and the result."""
    simulation, problem = accept_problem()
    backend = _native()(problem, grid=simulation.grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        launch = backend._native_state.field_to_host().copy()
        partials = np.array(backend._prepare_trial(0.0, ACCEPT_STEP), dtype=np.float64)
        backend._finish(backend._native_state, ACCEPT_STEP)
        return {
            "launch": launch,
            "partials": partials,
            "advanced": backend._native_state.field_to_host().copy(),
            "mask": np.asarray(problem.grids.norm_mask),
        }, _sizing(backend)
    finally:
        backend.close()


def linear_cartesian(mode: str):
    """Return ``cartesian_march``'s beam with no response: a pass is its fold."""
    from backends.test_low_storage import cartesian_march  # noqa: PLC0415

    problem, grid, _ = cartesian_march(mode)
    return (
        dataclasses.replace(
            problem,
            responses=dataclasses.replace(
                problem.responses,
                response_order=problem.responses.response_order[-1:],
                kerr_coefficients=None,
            ),
        ),
        grid,
    )


def folds(mode: str):
    """March the linear beam, saving every step's squared norm from the engine."""
    problem, grid = linear_cartesian(mode)
    backend = _native()(problem, grid=grid)
    norms = []
    try:
        engine = backend._low_storage._engine
        step = engine.step

        def recorded(z, dz):
            norms.append(step(z, dz))
            return norms[-1]

        engine.step = recorded
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(float(problem.stepper.save_planes[-1]))
        return {
            "field": backend.observe().require_field().copy(),
            "norms": np.array(norms),
        }, _sizing(backend)
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
    schedule = problem.stepper.schedule
    return (
        dataclasses.replace(
            problem,
            responses=dataclasses.replace(
                problem.responses, absorber_scheme="split_step"
            ),
            stepper=dataclasses.replace(problem.stepper, integrator=None),
        ),
        linear.grid,
        float(schedule.steps[0]),
    )


def flow(name: str):
    """One linear trial of a coupled flow: the partials and the state."""
    problem, grid, dz = linear_flow_problem(name)
    backend = _native()(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        partials = np.array(backend._prepare_trial(0.0, dz), dtype=np.float64)
        backend._finish(backend._native_state, dz)
        return {
            "partials": partials,
            "field": backend._native_state.field_to_host().copy(),
        }, _sizing(backend)
    finally:
        backend.close()


def flow_kerr(name: str):
    """Ten steps of a ``test_native_coupled_flow.py`` case, with its source."""
    from driver.test_native_coupled_flow import CASES as FLOWS  # noqa: PLC0415
    from driver.test_native_coupled_flow import propagate  # noqa: PLC0415

    case = FLOWS[name]()
    backend_class = _native()
    sizing: dict[str, dict[str, int]] = {}

    class Sized(backend_class):
        """The backend, keeping what it sized for the probe to read after the close."""

        def close(self) -> None:
            sizing.update(_sizing(self))
            super().close()

    return {"field": propagate(case, Sized)}, sizing


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
