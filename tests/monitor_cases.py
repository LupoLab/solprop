# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The boundary monitor's probes, shared by the reading tests of every backend.

Each probe is a route's tiny case (``tests/capability_routes.py``) launched with a copy
of its pulse near the window's edge, so that every region the monitor weighs holds
light, under an absorber scheme and, with a march, on a fixed schedule. The reference's
reading of it is the oracle the compiled backends are held to in
``tests/backends/test_monitor_reduction.py``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import numpy as np

from capability_routes import LENGTH, ROUTES
from solprop.backends.driver import StepDriver
from solprop.backends.host.driver import HostDriver
from solprop.backends.protocol import Backend, ExecutionBackend
from solprop.backends.registry import select
from solprop.backends.session import ExecutionOptions
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
)
"""Every route: the single-column engine (the GNLSE and mode-averaged capillaries),
field and envelope modal columns, the general modal route, radial (whose columns carry
the Hankel quadrature), Cartesian and line free space and the planar hybrid."""

SCHEMES = ("nonlinear", "split_step")
MARCHES = {
    "runge_kutta": RungeKutta(),
    "multistep": Multistep(),
    "low_storage": LowStorageRungeKutta(),
}
"""The marches whose finishes differ: the Runge-Kutta finish, the Adams adoption (the
native engine's resident history on its modal routes) and the low-storage register
step accepted in place (the native engine's resident step)."""

STEPS = 4
"""Fixed steps over the probe's fifth of a millimetre: a matched schedule, so that the
backends take the same steps."""

COPY_SHARE = 0.5
"""Amplitude of the launch's copy near the edge, against the pulse's."""

COPY_AT = 0.42
"""Where the copy sits, as a fraction of the window from the pulse: inside the collar,
its tail in the high guard."""


def probe(
    route: str, scheme: str, march: object | None = None, *, steps: int = STEPS
) -> Simulation:
    """Return a route's tiny case under an absorber scheme; with a march, on a fixed
    schedule of `steps` steps."""
    simulation = ROUTES[route](None)
    settings: dict[str, Any] = {
        # Relaxed: the launch puts light in the guards on purpose.
        "boundary_check": BoundaryCheck(budget=1.0, stop_factor=1.0),
        "absorber_scheme": scheme,
    }
    if march is not None:
        settings["integrator"] = march
        settings["schedule"] = FixedSchedule.uniform(
            0.0, LENGTH, LENGTH / steps, save_planes=[LENGTH]
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


def as_driver(backend: ExecutionBackend) -> StepDriver[Any] | HostDriver[Any]:
    """Return an execution owner after checking that it has a legacy driver.

    The reference's own (:class:`~solprop.backends.host.driver.HostDriver`) or the
    native bridge's inherited one (SEP 47 phase 2).
    """
    assert isinstance(backend, (StepDriver, HostDriver))
    return cast("StepDriver[Any] | HostDriver[Any]", backend)


def watched(
    simulation: Simulation,
    name: str,
    problem: PropagationProblem | None = None,
    options: ExecutionOptions | None = None,
    run_request: RunRequest | None = None,
) -> Backend:
    """Return a backend loaded with the edge-reaching launch and watched.

    `options` default to the simulation's own execution options; `problem` and
    `run_request` to its own translation.
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
    driver = as_driver(backend)
    assert isinstance(driver, Backend)
    driver.set_initial(with_copy(problem.initial.field0, simulation.grid.n_t), None)
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    driver.attach_monitor(monitor)
    return driver


def reading(backend: ExecutionBackend) -> BoundaryEnergies:
    """Return a backend's reading of its current state."""
    driver = as_driver(backend)
    return driver._boundary_energies(driver._require_state())


def record_after_march(
    simulation: Simulation,
    name: str,
    problem: PropagationProblem | None = None,
    run_request: RunRequest | None = None,
) -> dict[str, Any]:
    """Return a backend's monitor record after the probe's fixed steps, with the split
    collar's crossing cap the last check read (``step_cap``)."""
    backend = watched(simulation, name, problem, run_request=run_request)
    try:
        backend.propagate_to(LENGTH)
        monitor = as_driver(backend)._monitor
        assert monitor is not None
        return {**monitor.summary(), "step_cap": monitor.step_cap}
    finally:
        backend.close()
