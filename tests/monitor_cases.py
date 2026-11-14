# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The boundary monitor's probes, shared by the reading tests of every backend.

Each probe is a route's tiny case (``tests/capability_routes.py``) launched with a copy
of its pulse near the window's edge, so that every region the monitor weighs holds
light, under an absorber scheme and, with a march, on a fixed schedule. The reference's
reading of it is the oracle the compiled backends are held to in
``tests/backends/test_monitor_reduction.py``.

A reading of a loaded state (:func:`watched`, :func:`reading`) is the reference
driver's alone: since SEP 47 phase 5 the native backend has no direct driver API, and
its monitor is read through its session (:func:`record_after_march`).
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import numpy as np

from capability_routes import LENGTH, ROUTES
from solprop.backends.events import Terminal
from solprop.backends.host.driver import HostDriver
from solprop.backends.protocol import Backend
from solprop.backends.registry import select
from solprop.backends.session import ExecutionOptions, Start
from solprop.backends.sessions import start_session
from solprop.driver import Simulation
from solprop.monitor import BoundaryCheck, BoundaryEnergies
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest

ROUTE_NAMES = (
    "gnlse",
    "capillary (mode-averaged)",
    "capillary (envelope)",
    "capillary (multimode)",
    "general guide (constant)",
    "radial",
    "cartesian",
    "line",
    "planar",
    "capillary (pressure gradient)",
    "general guide (moving)",
    "capillary (coherent wall)",
    "capillary (taper coupling)",
)
"""Every route: the single-column engine (the GNLSE and mode-averaged capillaries),
field and envelope modal columns, the general modal route, radial (whose columns carry
the Hankel quadrature), Cartesian and line free space, the planar hybrid, the pressure
gradient, and the coupled flows (:data:`COUPLED`)."""

COUPLED = (
    "general guide (moving)",
    "capillary (coherent wall)",
    "capillary (taper coupling)",
)
"""The coupled-flow routes: the moving general guide, the coherent wall and the taper
coupling, which the low-storage march refuses by name."""

SCHEMES = ("nonlinear", "split_step")
MARCHES = {
    "runge_kutta": RungeKutta(),
    "multistep": Multistep(),
    "low_storage": LowStorageRungeKutta(),
}
"""The marches whose finishes differ: the Runge-Kutta finish, the Adams adoption (the
native engine's resident history on its modal routes) and the low-storage register
step accepted in place (the native engine's resident step)."""

ROUTE_MARCHES = tuple(
    (route, march)
    for route in ROUTE_NAMES
    for march in sorted(MARCHES)
    if not (march == "low_storage" and route in COUPLED)
)
"""Every route under every march it admits: the low-storage march steps a diagonal
linear flow only, and refuses the :data:`COUPLED` routes at translation."""

STEPS = 4
"""Fixed steps over the probe's fifth of a millimetre: a matched schedule, so that the
backends take the same steps."""

COPY_SHARE = 0.5
"""Amplitude of the launch's copy near the edge, against the pulse's."""

COPY_AT = 0.42
"""Where the copy sits, as a fraction of the window from the pulse: inside the collar,
its tail in the high guard."""


QUIET_STEPS = 20
"""Fixed steps of the quiet probe: more than :data:`solprop.monitor.CADENCE`, so that a
quiet window's cadence leaves steps between its checks unchecked."""

QUIET_BUDGET = 100.0
"""The quiet probe's budget: the activity threshold, ``(0.1 budget)**2`` of the energy,
is above any share a region can hold, so no step is checked for an active edge."""


def probe(
    route: str,
    scheme: str,
    march: object | None = None,
    *,
    steps: int = STEPS,
    budget: float = 1.0,
) -> Simulation:
    """Return a route's tiny case under an absorber scheme; with a march, on a fixed
    schedule of `steps` steps, more where a hard kink splits one.

    The check's `budget` is relaxed (the launch puts light in the guards on purpose):
    at the default every step holds an active collar and is checked, and at
    :data:`QUIET_BUDGET` none does, so only the cadence decides.
    """
    simulation = ROUTES[route](None)
    settings: dict[str, Any] = {
        "boundary_check": BoundaryCheck(budget=budget, stop_factor=1.0),
        "absorber_scheme": scheme,
    }
    if march is not None:
        settings["integrator"] = march
        # A fixed schedule must land on the guide's hard kinks (the taper coupling's
        # corners), which split a segment into more, shorter steps. They are read off
        # a second build: a translated simulation takes no new schedule.
        settings["schedule"] = FixedSchedule.uniform(
            0.0,
            LENGTH,
            LENGTH / steps,
            save_planes=[LENGTH],
            hard_kinks=ROUTES[route](None).translate().linear.hard_kinks,
        )
    simulation.settings = dataclasses.replace(simulation.settings, **settings)
    return simulation


def with_copy(field: np.ndarray, n_t: int) -> np.ndarray:
    """Return the launch with a copy of itself delayed towards the window's edge.

    A whole number of samples, ``exp(-2 pi i k s / n_t)`` on bin ``k``: an exact
    circular shift on either spectral layout.
    """
    shift = round(COPY_AT * n_t)
    phase = np.exp(-2j * np.pi * np.arange(field.shape[-1]) * shift / n_t)
    return field * (1.0 + COPY_SHARE * phase)


def as_driver(backend: Backend) -> HostDriver[Any]:
    """Return the reference backend as its step driver.

    Raises
    ------
    TypeError
        For any other backend: since SEP 47 phase 5 the native backend has no direct
        driver API, and its monitor is read through its session
        (:func:`record_after_march`, or a session's ``monitor_record``).
    """
    if not isinstance(backend, HostDriver):
        msg = (
            f"{type(backend).__name__} has no step driver to load and read directly; "
            "read its boundary monitor through its session "
            "(monitor_cases.record_after_march or RunSession.monitor_record)"
        )
        raise TypeError(msg)
    return cast("HostDriver[Any]", backend)


def watched(
    simulation: Simulation,
    name: str,
    problem: PropagationProblem | None = None,
    options: ExecutionOptions | None = None,
    run_request: RunRequest | None = None,
) -> Backend:
    """Return the reference backend loaded with the edge-reaching launch and watched.

    `options` default to the simulation's own execution options; `problem` and
    `run_request` to its own translation.

    Raises
    ------
    TypeError
        For a backend other than the reference (:func:`as_driver`).
    """
    problem = simulation.translate() if problem is None else problem
    run_request = simulation._run_request() if run_request is None else run_request
    backend = select(
        name,
        problem,
        run_request,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
        options=simulation.settings.execution_options() if options is None else options,
    )
    try:
        driver = as_driver(backend)
    except TypeError:
        backend.close()
        raise
    driver.set_initial(with_copy(problem.initial.field0, simulation.grid.n_t), None)
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    driver.attach_monitor(monitor)
    return backend


def reading(backend: Backend) -> BoundaryEnergies:
    """Return the reference backend's reading of its current state.

    Raises
    ------
    TypeError
        For a backend other than the reference (:func:`as_driver`).
    """
    driver = as_driver(backend)
    return driver._boundary_energies(driver._require_state())


def record_after_march(
    simulation: Simulation,
    name: str,
    problem: PropagationProblem | None = None,
    run_request: RunRequest | None = None,
    *,
    interval: float | None = None,
) -> dict[str, Any]:
    """Return a backend's monitor record after the probe's fixed steps, with the split
    collar's crossing cap the last check read (``step_cap``).

    `interval`, when given, replaces the monitor's longest unchecked distance
    (:attr:`solprop.monitor.BoundaryMonitor.interval`, the guide's length over
    :data:`solprop.monitor.CHECKS`, which every probe's step exceeds): ``math.inf``
    leaves a quiet window to the cadence alone.

    The edge-reaching launch runs through the backend's own session, watched by the
    simulation's boundary monitor. The reference's session updates that monitor, whose
    record and cap are read from it; the native session keeps its monitor natively
    (SEP 47 section 6.4) and reports the record (``monitor_record``) and its watch's
    crossing cap (``monitor_cap``) itself.
    """
    problem = simulation.translate() if problem is None else problem
    run_request = simulation._run_request() if run_request is None else run_request
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    if interval is not None:
        monitor.interval = interval
    request = run_request.for_run(monitor=monitor)
    backend = select(
        name,
        problem,
        request,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
        options=simulation.settings.execution_options(),
    )
    start = Start(
        field=with_copy(problem.initial.field0, simulation.grid.n_t),
        z=float(problem.initial.z),
    )
    session = start_session(backend, problem, request, start)
    try:
        while not session.finished:
            batch = session.advance()
            session.acknowledge(batch.receipt)
            terminal = batch.terminal
            if isinstance(terminal, Terminal) and terminal.error is not None:
                raise terminal.error
        record = session.monitor_record()
        assert record is not None
        if isinstance(backend, HostDriver):
            cap = float(monitor.step_cap)
        else:
            cap = float(cast("Any", session).monitor_cap())
    finally:
        session.close()
    return {**record, "step_cap": cap}
