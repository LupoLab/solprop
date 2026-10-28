# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The boundary monitor's reading on every backend, against the reference (SEP 21).

A check reads five numbers of the state where the state lives
(``StepDriver._boundary_energies``): its energy and its four weighted energies, the
low and high guards, the collar's ``1 - W`` and its rate. Under the ``"split_step"``
scheme the backend's own finish adds what each accepted step's collar takes, and the
reading drains the sum. ``numpy_ref`` forms both on the host
(:meth:`solprop.monitor.BoundaryMonitor.energies`, ``split_removal``), the oracle every
other backend is held to here, on each route's tiny case
(``tests/capability_routes.py``) launched with a copy of its pulse near the window's
edge, so that every region holds light.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

from bounds import device_backend, ladder_bound
from capability_routes import LENGTH, ROUTES
from legacy_geometry import bare_absorber
from solprop.backends.driver import StepDriver
from solprop.backends.protocol import Backend
from solprop.backends.registry import available, select
from solprop.driver import Simulation
from solprop.exceptions import BackendError
from solprop.monitor import BoundaryCheck, BoundaryEnergies
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.problem import KernelProblem

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

COMPILED = ("numba_cpu", "native")
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


def _simulation(route: str, scheme: str, march: object | None = None) -> Simulation:
    """Return a route's tiny case under an absorber scheme; with a march, on a fixed
    schedule."""
    simulation = ROUTES[route](None)
    settings: dict[str, Any] = {
        # Relaxed: the launch puts light in the guards on purpose.
        "boundary_check": BoundaryCheck(budget=1.0, stop_factor=1.0),
        "absorber_scheme": scheme,
    }
    if march is not None:
        settings["integrator"] = march
        settings["schedule"] = FixedSchedule.uniform(
            0.0, LENGTH, LENGTH / STEPS, save_planes=[LENGTH]
        )
    simulation.settings = dataclasses.replace(simulation.settings, **settings)
    return simulation


def _with_copy(field: np.ndarray, n_t: int) -> np.ndarray:
    """Return the launch with a copy of itself delayed towards the window's edge.

    A whole number of samples, ``exp(-2 pi i k s / n_t)`` on bin ``k``: an exact
    circular shift on either spectral layout.
    """
    shift = round(COPY_AT * n_t)
    phase = np.exp(-2j * np.pi * np.arange(field.shape[-1]) * shift / n_t)
    return field * (1.0 + COPY_SHARE * phase)


def _loaded(
    simulation: Simulation, name: str, problem: KernelProblem | None = None
) -> Backend:
    """Return a backend loaded with the edge-reaching launch and watched."""
    problem = simulation.translate() if problem is None else problem
    backend = select(
        name,
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    driver = _driver(backend)
    driver.set_initial(_with_copy(problem.initial.field0, simulation.grid.n_t), None)
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    driver.attach_monitor(monitor)
    return backend


def _driver(backend: Backend) -> StepDriver[Any]:
    """Return a backend as the step driver every registered backend is."""
    assert isinstance(backend, StepDriver)
    return cast("StepDriver[Any]", backend)


def _reading(backend: Backend) -> BoundaryEnergies:
    """Return a backend's reading of its current state."""
    driver = _driver(backend)
    return driver._boundary_energies(driver._require_state())


def _rounding(simulation: Simulation) -> float:
    """Return the reading's rounding budget, relative to the state's energy.

    Each time sample carries the inverse transform's rounding, about ``eps log2
    n_t`` of the field's norm, and each sum adds ``eps`` per term in the worst order
    a threaded reduction may take: ``eps (log2 n_t + n)`` over the ``n`` samples of
    every column, all of them non-negative.
    """
    shape = simulation.translate().shape
    samples = math.prod(shape[:-1]) * simulation.grid.n_t
    return float(np.finfo(np.float64).eps) * (math.log2(simulation.grid.n_t) + samples)


def _require(backend: str) -> None:
    """Skip when a backend is not installed."""
    if backend not in available():
        pytest.skip(f"{backend} is not installed")


def _check_reading(
    backend: str,
    route: str,
    scheme: str,
    problem: Callable[[Simulation], KernelProblem] | None = None,
) -> None:
    """Compare a backend's reading of the edge-reaching launch with the reference's."""
    simulation = _simulation(route, scheme)
    translated = None if problem is None else problem(simulation)
    reference, compiled = (
        _loaded(simulation, name, translated) for name in ("numpy_ref", backend)
    )
    try:
        theirs, ours = _reading(reference), _reading(compiled)
    finally:
        reference.close()
        compiled.close()
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    largest = monitor.weights.max(axis=1)
    # Light in every region a weight reaches, so that each sum is compared on
    # something; without a collar the rate's row is zero.
    assert np.all(theirs.regions[largest > 0.0] > 1e-6 * theirs.total)
    # A device build adds its transform's declared envelope.
    bound = ladder_bound(backend, _rounding(simulation), simulation.grid.n_t, 1)
    assert ours.total == pytest.approx(theirs.total, rel=bound)
    # Each row's sum is bounded by its largest weight times the energy: the rate's
    # row is energy per metre.
    scale = bound * theirs.total * largest
    weighted = largest > 0.0
    np.testing.assert_array_less(
        np.abs(ours.regions - theirs.regions)[weighted], scale[weighted]
    )
    # A row no weight reaches sums nothing.
    assert np.all(ours.regions[~weighted] == 0.0)
    # Nothing was taken before the first step.
    assert ours.removed == theirs.removed == 0.0


def _march(simulation: Simulation, name: str) -> dict[str, Any]:
    """Return a backend's monitor record after the probe's fixed steps."""
    backend = _loaded(simulation, name)
    try:
        backend.propagate_to(LENGTH)
        monitor = _driver(backend)._monitor
        assert monitor is not None
        return monitor.summary()
    finally:
        backend.close()


def _check_split(backend: str, route: str, march: str) -> None:
    """Compare a backend's split removal over a fixed march with the reference's."""
    simulation = _simulation(route, "split_step", MARCHES[march])
    theirs = _march(simulation, "numpy_ref")
    ours = _march(simulation, backend)
    assert theirs["raw_removed_fraction"] > 1e-6
    # Each step's removal carries the reading's rounding, and the matched steps' states
    # agree to the march's own rounding, amplified over the steps; a device build adds
    # its transforms' declared envelope, eight round trips a step at most.
    bound = ladder_bound(
        backend, 1e3 * _rounding(simulation), simulation.grid.n_t, 8 * STEPS
    )
    for name in ("raw_removed_fraction", "removed_fraction", "collar_max"):
        assert ours[name] == pytest.approx(theirs[name], rel=bound), name
    assert ours["steps"] == theirs["steps"]


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("route", ROUTE_NAMES)
@pytest.mark.parametrize("backend", COMPILED)
def test_a_check_reads_the_reference_s_five_numbers(
    backend: str, route: str, scheme: str
) -> None:
    _require(backend)
    _check_reading(backend, route, scheme)


@pytest.mark.parametrize("march", sorted(MARCHES))
@pytest.mark.parametrize("route", ROUTE_NAMES)
@pytest.mark.parametrize("backend", COMPILED)
def test_the_split_s_removal_is_the_reference_s(
    backend: str, route: str, march: str
) -> None:
    # Every accepted step's removal, added up by the backend's own finish on each
    # march; the steps are fixed, so the backends take the same ones.
    _require(backend)
    _check_split(backend, route, march)


def _require_device() -> None:
    """Skip unless the installed native build runs on an accelerator."""
    _require("native")
    if device_backend("native") == "none":
        pytest.skip("the installed solprop-native build has no accelerator backend")


@pytest.mark.gpu
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("route", ROUTE_NAMES)
def test_a_device_reads_the_reference_s_five_numbers(route: str, scheme: str) -> None:
    # The reduction runs in the device's memory space; only the numbers cross.
    _require_device()
    _check_reading("native", route, scheme)


@pytest.mark.gpu
@pytest.mark.parametrize("march", sorted(MARCHES))
@pytest.mark.parametrize("route", ROUTE_NAMES)
def test_a_device_s_split_removal_is_the_reference_s(route: str, march: str) -> None:
    _require_device()
    _check_split("native", route, march)


@pytest.mark.parametrize("backend", ["numpy_ref", *COMPILED])
def test_a_reading_needs_a_monitor(backend: str) -> None:
    _require(backend)
    simulation = _simulation("gnlse", "nonlinear")
    problem = simulation.translate()
    made = select(backend, problem, grid=simulation.grid)
    try:
        driver = _driver(made)
        driver.set_initial(problem.initial.field0, None)
        with pytest.raises(BackendError, match="no boundary monitor"):
            driver._boundary_energies(driver._require_state())
    finally:
        made.close()


def test_a_monitor_for_another_grid_is_refused() -> None:
    # Its weights would be read out of bounds by the compiled sums.
    _require("numba_cpu")
    simulation = _simulation("gnlse", "nonlinear")
    problem = simulation.translate()
    other = _simulation("capillary (envelope)", "nonlinear")._boundary_monitor()
    assert other is not None
    made = select("numba_cpu", problem, grid=simulation.grid)
    try:
        driver = _driver(made)
        driver.set_initial(problem.initial.field0, None)
        with pytest.raises(BackendError, match="do not match the grid"):
            driver.attach_monitor(other)
    finally:
        made.close()


def _chunked(simulation: Simulation) -> KernelProblem:
    """Return the Cartesian case's problem with node chunks that leave a partial one."""
    problem = simulation.translate()
    return dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=50)
    )


def _bare(simulation: Simulation) -> KernelProblem:
    """Return a route's problem without its temporal collar."""
    simulation.settings = dataclasses.replace(
        simulation.settings, absorber=bare_absorber(simulation.grid)
    )
    return simulation.translate()


@pytest.mark.parametrize(
    ("route", "problem"),
    [("cartesian", _chunked), ("gnlse", _bare), ("capillary (multimode)", _bare)],
    ids=["partial-chunk", "single-column-no-collar", "modal-no-collar"],
)
@pytest.mark.parametrize("backend", COMPILED)
def test_a_check_reads_the_reference_s_numbers_at_the_edges(
    backend: str, route: str, problem: Callable[[Simulation], KernelProblem]
) -> None:
    # A last chunk shorter than the rest, and a run with no collar, for which the
    # backends plan their own coarse transform at the first attach.
    _require(backend)
    _check_reading(backend, route, "nonlinear", problem)


@pytest.mark.parametrize("backend", ["numpy_ref", *COMPILED])
def test_a_second_watched_run_reads_as_the_first(backend: str) -> None:
    # A runner loads its backend again for the next input and attaches a new monitor:
    # the views the first allocated are reused, and nothing taken carries over.
    _require(backend)
    simulation = _simulation("radial", "split_step", MARCHES["runge_kutta"])
    made = _loaded(simulation, backend)
    try:
        driver = _driver(made)
        first = _reading(made)
        made.propagate_to(LENGTH)
        problem = simulation.translate()
        driver.set_initial(
            _with_copy(problem.initial.field0, simulation.grid.n_t), None
        )
        monitor = simulation._boundary_monitor()
        assert monitor is not None
        driver.attach_monitor(monitor)
        again = _reading(made)
        ledger = made.memory_ledger()
    finally:
        made.close()
    assert again.total == first.total
    np.testing.assert_array_equal(again.regions, first.regions)
    assert again.removed == 0.0
    assert all(size >= 0 for size in ledger.values())
