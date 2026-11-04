# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native's reading of the transverse edge, against the host's on one state (SEP 44).

The state is the same array on both sides, so nothing here is propagation: a reading is
an inverse transform and a sum, and the two may differ by that transform's rounding.
"""

import dataclasses
from typing import Any, cast

import numpy as np
import pytest

from bounds import fft_bound
from capability_routes import LENGTH, SPATIAL_GRID
from capability_routes import ROUTES as CASES
from solprop.backends.driver import StepDriver
from solprop.backends.registry import available, select
from solprop.driver import Simulation
from solprop.grid import FreeGrid, LineGrid, RadialGrid
from solprop.monitor import BoundaryCheck
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, Multistep, RungeKutta
from solprop.transverse_monitor import EdgeReading

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)
NativeBackend = pytest.importorskip("solprop_native.backend").NativeBackend

ROUTES = ("radial", "cartesian", "line", "planar")


def _case(route: str) -> Simulation:
    """Return a tiny route, watched, under a check that stops nothing."""
    simulation = CASES[route](None)
    simulation.settings = dataclasses.replace(
        simulation.settings,
        boundary_check=BoundaryCheck(
            budget=1.0, stop_factor=1.0, transverse_budget=1.0
        ),
    )
    return simulation


def _noise_state(shape: tuple[int, ...], active: np.ndarray, seed: int) -> np.ndarray:
    """Return white noise on the active band, which fills every node and face."""
    rng = np.random.default_rng(seed)
    state = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    state[..., ~active] = 0.0
    return state


def _attached(simulation: Simulation, state: np.ndarray) -> Any:
    """Return a native backend loaded with `state` and watched by the edge."""
    problem = simulation.translate()
    backend = cast(
        "Any",
        select(
            "native",
            problem,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        ),
    )
    backend.set_initial(state, None)
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    assert monitor.transverse is not None
    backend._bind_monitor(monitor)
    return backend, monitor.transverse


def _read_both(
    route: str, *, full: bool
) -> tuple[EdgeReading, EdgeReading, np.ndarray]:
    """Return native's and the host's reading of one state, and the state."""
    simulation = _case(route)
    problem = simulation.translate()
    state = _noise_state(problem.shape, np.asarray(problem.grids.active_mask), 41)
    backend, edge = _attached(simulation, state)
    try:
        native = backend._edge_reading(backend._require_state(), edge, full=full)
        host = edge.reduction.reading(state[0], full=full)
    finally:
        backend.close()
    return native, host, state


@pytest.mark.parametrize("full", [True, False], ids=["launch", "read"])
@pytest.mark.parametrize("route", ROUTES)
def test_native_reads_what_the_host_reads(route: str, full: bool) -> None:
    native, host, state = _read_both(route, full=full)
    samples = state.shape[1] * state.shape[3]
    budget = 4 * fft_bound(samples)
    assert native.faces is not None and host.faces is not None
    assert native.columns is not None and host.columns is not None
    np.testing.assert_allclose(native.columns, host.columns, rtol=budget)
    # A sum of one region against its own scale: each is non-negative, so the rounding
    # is relative to it, and white noise leaves no region empty.
    assert np.all(host.faces[host.faces != 0.0] > 0.0)
    np.testing.assert_allclose(native.faces, host.faces, rtol=budget)
    if full:
        assert native.clip is not None and host.clip is not None
        assert native.totals is not None and host.totals is not None
        np.testing.assert_allclose(native.clip, host.clip, rtol=budget)
        np.testing.assert_allclose(native.totals, host.totals, rtol=budget)
    else:
        assert native.clip is None
        assert native.totals is None


@pytest.mark.parametrize("route", ROUTES)
def test_native_reads_a_state_it_holds_without_changing_it(route: str) -> None:
    simulation = _case(route)
    problem = simulation.translate()
    state = _noise_state(problem.shape, np.asarray(problem.grids.active_mask), 43)
    backend, edge = _attached(simulation, state)
    try:
        before = backend._native_state.field_to_host()
        backend._edge_reading(backend._require_state(), edge, full=True)
        backend._edge_reading(backend._require_state(), edge)
        after = backend._native_state.field_to_host()
    finally:
        backend.close()
    # The reading works on the scratch: the state's own bits don't move.
    assert before.tobytes() == after.tobytes()


@pytest.mark.parametrize(
    ("node", "lit_faces"),
    [((0, 5), {"y-"}), ((5, 0), {"x-"}), ((0, 0), {"x-", "y-"})],
    ids=["y", "x", "corner"],
)
def test_one_lit_node_reaches_only_its_own_faces(
    node: tuple[int, int], lit_faces: set[str]
) -> None:
    # A delta in space is a flat spectrum in the columns. A read with the wrong row
    # order, axis or scale fails this at once, where white noise can't tell them apart.
    # The low faces: on so coarse a grid the high ones hold no node.
    simulation = _case("cartesian")
    problem = simulation.translate()
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    assert monitor.transverse is not None
    space = monitor.transverse.space
    assert isinstance(space, FreeGrid)
    active = np.asarray(problem.grids.active_mask)
    state = np.zeros(problem.shape, dtype=np.complex128)
    y = np.arange(space.n_y)[:, None]
    x = np.arange(space.n_x)[None, :]
    phase = np.exp(-2j * np.pi * (y * node[0] / space.n_y + x * node[1] / space.n_x))
    state[0, :, 0, int(np.nonzero(active)[0][0])] = phase.reshape(-1)
    backend, edge = _attached(simulation, state)
    try:
        native = backend._edge_reading(backend._require_state(), edge, full=True)
        host = edge.reduction.reading(state[0], full=True)
    finally:
        backend.close()
    assert native.faces is not None
    assert host.faces is not None
    budget = 4 * fft_bound(space.n_y * space.n_x)
    names = [face.name for face in edge._faces]
    lit = np.array([name in lit_faces for name in names] + [False] * (4 - len(names)))
    np.testing.assert_allclose(native.faces[lit], host.faces[lit], rtol=budget)
    # Exactly one node's weight, so the scale is right as well as the agreement.
    expected = edge._weights[node[0] * space.n_x + node[1]]
    np.testing.assert_allclose(native.faces[lit], expected, rtol=budget)
    # The dark faces hold the transform's rounding, on the lit face's scale.
    assert np.max(np.abs(native.faces[~lit])) <= budget * expected


def _polarised(route: str) -> Simulation:
    """Return a tiny free-space route launching two polarisations."""
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9),
        80e-6,
        jones=(1, 0.4j),
    )
    medium = SpatialMedium("Ar", 1.0)
    settings = _case(route).settings
    if route == "radial":
        return Simulation.from_radial(
            SPATIAL_GRID,
            RadialGrid(3e-4, 16, collar_fraction=0),
            medium,
            [],
            pulse,
            settings=settings,
            noise=None,
        )
    space = (
        LineGrid(3e-4, 16, collar_fraction=0)
        if route == "line"
        else FreeGrid(3e-4, 12, 4e-4, 16, collar_fraction=0)
    )
    return Simulation.from_free(
        SPATIAL_GRID, space, medium, [], pulse, settings=settings, noise=None
    )


@pytest.mark.parametrize("route", ["radial", "cartesian", "line"])
def test_two_polarisations_read_as_the_host_reads(route: str) -> None:
    # A column is `mode * n_pol + pol`, so a mode's energy has to gather both.
    simulation = _polarised(route)
    problem = simulation.translate()
    assert problem.shape[2] == 2
    state = _noise_state(problem.shape, np.asarray(problem.grids.active_mask), 47)
    backend, edge = _attached(simulation, state)
    try:
        native = backend._edge_reading(backend._require_state(), edge, full=True)
        host = edge.reduction.reading(state[0], full=True)
    finally:
        backend.close()
    assert native.columns is not None and host.columns is not None
    assert native.faces is not None and host.faces is not None
    budget = 4 * fft_bound(state.shape[1] * state.shape[3])
    np.testing.assert_allclose(native.columns, host.columns, rtol=budget)
    np.testing.assert_allclose(native.faces, host.faces, rtol=budget)


@pytest.mark.parametrize(
    "march",
    [RungeKutta(), Multistep(), LowStorageRungeKutta()],
    ids=["runge_kutta", "adams", "low_storage"],
)
@pytest.mark.parametrize("route", ROUTES)
def test_a_read_at_every_step_leaves_the_march_where_it_was(
    route: str, march: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A read takes the transverse scratch at a committed position, which every march
    # has to write again before using. Against the same run with the host's read, which
    # touches no scratch: same steps, same monitor, one process, so no rounding apart.
    def final() -> tuple[np.ndarray, int]:
        simulation = _case(route)
        simulation.settings = dataclasses.replace(
            simulation.settings,
            integrator=march,  # pyright: ignore[reportArgumentType]
            schedule=FixedSchedule.uniform(
                0.0, LENGTH, LENGTH / 4, save_planes=[LENGTH]
            ),
        )
        problem = simulation.translate()
        backend = cast(
            "Any",
            select(
                "native",
                problem,
                grid=simulation.grid,
                modes=simulation.modes,
                profiles=simulation.profiles,
            ),
        )
        try:
            backend.set_initial(problem.initial.field0, None)
            monitor = simulation._boundary_monitor()
            assert monitor is not None
            backend.attach_monitor(monitor)
            backend.propagate_to(LENGTH)
            record = cast("dict[str, Any]", monitor.summary())
            return backend.observe(field=True).require_field(), int(
                record["transverse"]["reads"]
            )
        finally:
            backend.close()

    native, reads = final()
    # Every step is longer than the spacing.
    assert reads >= 4
    with monkeypatch.context() as patch:
        patch.setattr(NativeBackend, "_edge_reading", StepDriver._edge_reading)
        host, host_reads = final()
    assert host_reads == reads
    assert native.tobytes() == host.tobytes()


def test_a_field_free_default_run_crosses_no_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The principle of SEP 44: only numbers cross. Every state-sized read of the host
    # goes through `_observed_field` on this route.

    def refused(self: StepDriver[Any], state: Any) -> np.ndarray:
        raise AssertionError("a field-free run brought the state to the host")

    monkeypatch.setattr(NativeBackend, "_observed_field", refused)
    simulation = _case("cartesian")
    simulation.settings = dataclasses.replace(
        simulation.settings, fields="none", held_plane=False
    )
    problem = simulation.translate()
    with simulation.runner() as runner:
        output = runner.run(np.asarray(problem.initial.field0)[0], progress="none")
    outcome = cast("Any", output).outcome
    edge = outcome.execution["boundary_monitor"]["transverse"]
    assert edge["reads"] > 0
