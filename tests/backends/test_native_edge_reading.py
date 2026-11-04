# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native's reading of the transverse edge, against the host's on one state (SEP 44).

The state is the same array on both sides, so nothing here is propagation: a reading is
an inverse transform and a sum, and the two may differ by that transform's rounding.
"""

import dataclasses
import warnings
from typing import Any, cast

import numpy as np
import pytest

from bounds import fft_bound, higham_gamma
from capability_routes import LENGTH, SPATIAL_GRID
from capability_routes import ROUTES as CASES
from solprop.backends.driver import StepDriver
from solprop.backends.memory import low_storage_stage_register
from solprop.backends.registry import available, select
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.grid import FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.modes import PlanarGrid
from solprop.monitor import BoundaryCheck
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import (
    FixedSchedule,
    LowStorageRungeKutta,
    Multistep,
    PIController,
    RungeKutta,
)
from solprop.transverse_monitor import GUARD, EdgeReading, transverse_edge_bytes
from test_transverse_reads import REREVIEW, packet_run

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


@pytest.mark.parametrize("route", ["cartesian", "line"])
def test_the_march_above_runs_without_its_stage_register(route: str) -> None:
    # Then a pass writes the next input straight into the transverse scratch, which
    # a read takes. If this stopped being so, the test above would stop covering it.
    simulation = _case(route)
    simulation.settings = dataclasses.replace(
        simulation.settings,
        integrator=LowStorageRungeKutta(),
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / 4, save_planes=[LENGTH]),
    )
    assert not low_storage_stage_register(simulation.translate(), "native")


def _direct_node_energies(space: Any, state: np.ndarray) -> np.ndarray:
    """Return each node's ``sum |E|^2`` by a plain transform, not the edge's code."""
    flat = state.reshape(state.shape[0], -1)
    if isinstance(space, RadialGrid):
        hankel = space.hankel
        matrix = (
            hankel.inverse_scale
            * (hankel.j1[:, None] / hankel.j1[None, :])
            * hankel.matrix
        )
        return np.sum(np.abs(matrix @ flat) ** 2, axis=1)
    if isinstance(space, PlanarGrid):
        cube = flat.reshape(space.n_y, space.n_p, -1)
        return np.sum(np.abs(np.fft.ifft(cube, axis=0)) ** 2, axis=(1, 2))
    cube = flat.reshape(space.n_y, space.n_x, -1)
    return np.sum(np.abs(np.fft.ifftn(cube, axes=(0, 1))) ** 2, axis=2).ravel()


def _direct_faces(space: Any) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the node weights and each face's mask, from the grid and SEP 40's rule."""
    if isinstance(space, RadialGrid):
        r = np.asarray(space.r)
        return np.asarray(space.radial_weights), [r >= (1 - GUARD) * space.outer_radius]

    def axis_faces(axis: np.ndarray, flat: np.ndarray) -> list[np.ndarray]:
        outer = 0.5 * axis.size * float(axis[1] - axis[0])
        return [flat <= -(1 - GUARD) * outer, flat >= (1 - GUARD) * outer]

    if isinstance(space, PlanarGrid):
        y = np.asarray(space.y)
        return np.full(space.n_y, float(y[1] - y[0])), axis_faces(y, y)
    x = np.asarray(space.x)
    masks = axis_faces(x, np.tile(x, space.n_y))
    if not isinstance(space, LineGrid):
        y = np.asarray(space.y)
        masks += axis_faces(y, np.repeat(y, space.n_x))
    return np.asarray(space.spatial_weights), masks


@pytest.mark.parametrize("route", ROUTES)
def test_each_region_reads_what_its_definition_says(route: str) -> None:
    # Against the definition of SEP 40 and a plain transform, not the edge's tables: two
    # reducers agreeing on one wrongly built table wouldn't show the table is right.
    simulation = _case(route)
    problem = simulation.translate()
    state = _noise_state(problem.shape, np.asarray(problem.grids.active_mask), 53)
    backend, edge = _attached(simulation, state)
    try:
        native = backend._edge_reading(backend._require_state(), edge, full=True)
    finally:
        backend.close()
    assert native.faces is not None and native.totals is not None
    weights, masks = _direct_faces(edge.space)
    energies = _direct_node_energies(edge.space, state[0])
    budget = 4 * fft_bound(state.shape[1] * state.shape[3])
    expected = np.zeros(4)
    for index, mask in enumerate(masks):
        expected[index] = np.sum((weights * energies)[mask])
    np.testing.assert_allclose(native.faces, expected, rtol=budget)
    np.testing.assert_allclose(
        native.totals[0], np.sum(weights * energies), rtol=budget
    )
    # A row no region uses is held at zero, so that nothing reads it by accident.
    np.testing.assert_array_equal(native.totals[3], 0.0)


@pytest.mark.parametrize("route", ROUTES)
def test_the_grids_total_is_its_columns_by_parseval(route: str) -> None:
    # Two independent paths to one number: the nodes' quadrature on the synthesis, and
    # the columns' quadrature on the state, which doesn't go through a transform.
    simulation = _case(route)
    problem = simulation.translate()
    state = _noise_state(problem.shape, np.asarray(problem.grids.active_mask), 59)
    backend, edge = _attached(simulation, state)
    try:
        native = backend._edge_reading(backend._require_state(), edge, full=True)
    finally:
        backend.close()
    assert native.totals is not None and native.columns is not None
    spectral = np.asarray(edge.space.spectral_weights)
    budget = 4 * fft_bound(state.shape[1] * state.shape[3])
    if isinstance(edge.space, RadialGrid):
        # An FFT is unitary to rounding and a discrete Hankel transform isn't: its
        # columns are orthogonal only as far as its own matrix says, the largest
        # eigenvalue of the Gram matrix's departure from the identity.
        matrix = np.asarray(edge.reduction.synthesis())
        weights = np.asarray(edge.space.radial_weights)
        scaled = np.sqrt(weights)[:, None] * matrix / np.sqrt(spectral)[None, :]
        gram = scaled.T @ scaled - np.eye(spectral.size)
        budget += float(np.max(np.abs(np.linalg.eigvalsh(gram))))
    np.testing.assert_allclose(
        native.totals[0], np.sum(spectral * native.columns), rtol=budget
    )


def test_a_nearly_cancelling_radial_read_stays_within_its_absolute_budget() -> None:
    # A narrow beam's synthesis at the edge is a dense sum that all but cancels, so its
    # rounding is set by the terms' magnitudes, not by the small result: a bound
    # relative to the result would fail here. The budget is Higham's for a dot product.
    simulation = _case("radial")
    problem = simulation.translate()
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    assert monitor.transverse is not None
    space = monitor.transverse.space
    assert isinstance(space, RadialGrid)
    active = np.asarray(problem.grids.active_mask)
    r = np.asarray(space.r)
    narrow = np.exp(-((r / (0.1 * space.radius)) ** 2)).astype(complex)
    beam = space.hankel.forward(narrow)
    state = np.zeros(problem.shape, dtype=np.complex128)
    state[0, :, 0, :] = beam[:, None] * active[None, :]
    backend, edge = _attached(simulation, state)
    try:
        native = backend._edge_reading(backend._require_state(), edge, full=True)
    finally:
        backend.close()
    assert native.faces is not None
    host = edge.reduction.reading(state[0], full=True)
    assert host.faces is not None
    synthesis = np.asarray(edge.reduction.synthesis())
    flat = state[0].reshape(state.shape[1], -1)
    weights, masks = _direct_faces(space)
    terms = np.abs(synthesis) @ np.abs(flat)
    values = np.abs(synthesis @ flat)
    # Complex by real, then one sum of `n_r` terms, on each side of the comparison.
    error = higham_gamma(2 * space.n_r + 4)
    outer = masks[0]
    per_node = np.sum(2 * error * values * terms + (error * terms) ** 2, axis=1)
    bound = float(np.sum((weights * per_node)[outer]))
    # The share at the edge is rounding: what makes this case the hard one.
    total = float(np.sum(weights * np.sum(values**2, axis=1)))
    assert host.faces[0] < 1e-20 * total
    assert bound > 0.0
    assert abs(native.faces[0] - host.faces[0]) <= 2 * bound


def test_a_runner_reads_its_next_input_as_a_new_runner_would_after_a_stop() -> None:
    # What a stopped point leaves in the backend mustn't reach the next one: the bound
    # tables, the scratch and the clean columns are all reused.
    def run(*inputs: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        schedule = FixedSchedule.uniform(
            0.0, REREVIEW.length, REREVIEW.length / 16, save_planes=[REREVIEW.length]
        )
        simulation = packet_run(REREVIEW, schedule=schedule, backend="native")
        with simulation.runner() as runner, warnings.catch_warnings():
            # The leaving input warns on its way to the stop.
            warnings.simplefilter("ignore", BoundaryWarning)
            *earlier, last = inputs
            for value in earlier:
                with pytest.raises(BoundaryError):
                    runner.run(value, progress="none")
            output = cast("Any", runner.run(last, progress="none"))
        record = output.outcome.execution["boundary_monitor"]["transverse"]
        return np.asarray(output["field"])[-1], cast("dict[str, Any]", record)

    schedule = FixedSchedule.uniform(
        0.0, REREVIEW.length, REREVIEW.length / 16, save_planes=[REREVIEW.length]
    )
    base = packet_run(REREVIEW, schedule=schedule, backend="native")
    leaving = np.asarray(base.translate().initial.field0)[0]
    space = LineGrid(REREVIEW.half, REREVIEW.samples)
    staying_physical = dataclasses.replace(REREVIEW, velocity=0.0).physical(space)
    moving_physical = REREVIEW.physical(space)
    # The same pulse without the tilt: it stays near the axis and nothing warns.
    staying = (
        leaving
        * (np.fft.fft(staying_physical) / np.fft.fft(moving_physical))[:, None, None]
    )
    field_alone, alone = run(staying)
    field_after, after = run(leaving, staying)
    assert after["reads"] == alone["reads"] > 0
    assert after["edge_max"] == alone["edge_max"]
    assert field_after.tobytes() == field_alone.tobytes()


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
        simulation.settings, fields="none", backend="native"
    )
    problem = simulation.translate()
    field = np.asarray(problem.initial.field0)[0]
    with simulation.runner() as runner:
        output = runner.run(field, progress="none")
        assert isinstance(runner.backend, NativeBackend)
        state = cast("Any", runner.backend)._native_state
        to_host, from_host = state.crossed_bytes()
    outcome = cast("Any", output).outcome
    edge = outcome.execution["boundary_monitor"]["transverse"]
    assert edge["reads"] > 0
    # Refusing `_observed_field` can't see a direct `field_to_host`, so native's own
    # count says it too: nothing to the host, and the one load from it.
    assert to_host == 0
    assert from_host == field.nbytes


def test_the_crossing_count_sees_a_direct_copy() -> None:
    simulation = _case("cartesian")
    problem = simulation.translate()
    state = _noise_state(problem.shape, np.asarray(problem.grids.active_mask), 61)
    backend, _ = _attached(simulation, state)
    try:
        native = backend._native_state
        assert native.crossed_bytes() == (0, state.nbytes)
        native.field_to_host()
        assert native.crossed_bytes() == (state.nbytes, state.nbytes)
    finally:
        backend.close()


@pytest.mark.parametrize("route", ROUTES)
def test_the_edge_holds_memory_only_once_a_monitor_binds_it(route: str) -> None:
    # A check-off run pays nothing, and the formula an estimate adds is the row itself.
    simulation = _case(route)
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
        assert backend.memory_ledger()["transverse_edge"] == 0
        backend.set_initial(problem.initial.field0, None)
        assert backend.memory_ledger()["transverse_edge"] == 0
        monitor = simulation._boundary_monitor()
        assert monitor is not None
        assert monitor.transverse is not None
        backend._bind_monitor(monitor)
        _, _, n_pol, n_omega = problem.shape
        expected = transverse_edge_bytes(
            monitor.transverse.space, monitor.transverse.design, n_pol, n_omega
        )
        assert backend.memory_ledger()["transverse_edge"] == expected
        assert backend.edge_memory(monitor.transverse.space) == expected
    finally:
        backend.close()


def test_an_estimate_counts_the_edge_only_for_a_run_that_reads_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An estimate attaches no monitor, so it adds the row from the grid. A sentinel,
    # since a ledger's `fft_plan_heap_growth` is measured and differs between backends.
    sentinel = 2**50

    def estimated(check: BoundaryCheck | None) -> int:
        simulation = _case("cartesian")
        simulation.settings = dataclasses.replace(
            simulation.settings, boundary_check=check, backend="native"
        )
        return simulation.estimate().memory

    monkeypatch.setattr(NativeBackend, "edge_memory", lambda _self, _space: sentinel)
    assert estimated(BoundaryCheck(budget=1.0, stop_factor=1.0)) >= sentinel
    assert estimated(None) < sentinel


def _with_noise(route: str, model: str | None, *, ramp: bool) -> Simulation:
    """Return a tiny route with a noise model, field-free and on native."""
    simulation = CASES[route](model)
    simulation.settings = dataclasses.replace(
        simulation.settings,
        boundary_check=BoundaryCheck(
            budget=1.0, stop_factor=1.0, transverse_budget=1.0
        ),
        fields="none",
        backend="native",
        spatial_absorber=None if ramp else False,
    )
    return simulation


@pytest.mark.parametrize("ramp", [True, False], ids=["ramp", "no_ramp"])
@pytest.mark.parametrize("model", ["input", "modified", None], ids=str)
@pytest.mark.parametrize("route", ["radial", "cartesian", "line"])
def test_a_launch_with_noise_judges_as_the_host_does_with_no_transfer(
    route: str, model: str | None, ramp: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The clean columns are formed at the load, so judging a noisy launch needs no copy
    # of the state to subtract the noise from. Against the same launch read on the host.
    def launch() -> tuple[dict[str, Any], tuple[int, int]]:
        simulation = _with_noise(route, model, ramp=ramp)
        field = np.asarray(simulation.translate().initial.field0)[0]
        with simulation.runner() as runner, warnings.catch_warnings():
            warnings.simplefilter("ignore", BoundaryWarning)
            output = cast("Any", runner.run(field, progress="none"))
            crossed = cast("Any", runner.backend)._native_state.crossed_bytes()
        record = output.outcome.execution["boundary_monitor"]["transverse"]
        return cast("dict[str, Any]", record), crossed

    native, (to_host, _) = launch()
    assert to_host == 0
    with monkeypatch.context() as patch:
        patch.setattr(NativeBackend, "_edge_reading", StepDriver._edge_reading)
        host, _ = launch()
    assert native["launch_energy"] == pytest.approx(host["launch_energy"], rel=1e-11)
    assert native["noise_allowance"] == pytest.approx(
        host["noise_allowance"], rel=1e-11, abs=1e-30
    )
    assert native["noise_share_in_ramp"] == pytest.approx(
        host["noise_share_in_ramp"], rel=1e-11, abs=1e-30
    )


def test_a_field_grid_reads_as_the_host_does() -> None:
    # A real-signal grid's bins carry their own weights elsewhere: a reading has none on
    # either kind of grid, and a padded field grid's `n_to != n_t` changes nothing here.
    grid = FieldGrid.from_limits(1e-4, 800e-9, (500e-9, 1.6e-6), 200e-15)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 80e-6
    )
    settings = SimulationSettings(
        controller=PIController(rtol=1e-6, atol=0.0),
        save_planes=[1e-4],
        dz_max=1e-4,
        dz_init=1e-5,
        backend="native",
        boundary_check=BoundaryCheck(
            budget=1.0, stop_factor=1.0, transverse_budget=1.0
        ),
    )
    medium = SpatialMedium("Ar", 1.0)
    simulations = [
        Simulation.from_radial(
            grid,
            RadialGrid(6e-4, 32),
            medium,
            [],
            pulse,
            settings=settings,
            noise=None,
        ),
        Simulation.from_free(
            grid,
            LineGrid(3e-4, 16, collar_fraction=0),
            medium,
            [],
            pulse,
            settings=settings,
            noise=None,
        ),
    ]
    for simulation in simulations:
        problem = simulation.translate()
        state = _noise_state(problem.shape, np.asarray(problem.grids.active_mask), 79)
        backend, edge = _attached(simulation, state)
        try:
            native = backend._edge_reading(backend._require_state(), edge, full=True)
            host = edge.reduction.reading(state[0], full=True)
        finally:
            backend.close()
        budget = 4 * fft_bound(state.shape[1] * state.shape[3])
        np.testing.assert_allclose(native.faces, host.faces, rtol=budget)
        np.testing.assert_allclose(native.columns, host.columns, rtol=budget)


@pytest.mark.parametrize(
    "where", ["last_bin", "dc_column", "nyquist_column", "inactive_bin"], ids=str
)
def test_energy_in_one_extreme_bin_reads_as_the_host_does(where: str) -> None:
    # Off-by-one and stride errors hide under white noise. Unit energy in one element
    # of the state, at the band's last bin, at the transverse DC, or at its Nyquist.
    simulation = _case("cartesian")
    problem = simulation.translate()
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    assert monitor.transverse is not None
    space = monitor.transverse.space
    assert isinstance(space, FreeGrid)
    active = np.nonzero(np.asarray(problem.grids.active_mask))[0]
    state = np.zeros(problem.shape, dtype=np.complex128)
    if where == "last_bin":
        state[0, 5, 0, active[-1]] = 1.0
    elif where == "dc_column":
        state[0, 0, 0, active[0]] = 1.0
    elif where == "inactive_bin":
        # The load masks it, so a read sees only the lit bin beside it.
        state[0, 5, 0, active[0]] = 1.0
        state[0, 7, 0, np.nonzero(~np.asarray(problem.grids.active_mask))[0][0]] = 1.0
    else:
        nyquist = (space.n_y // 2) * space.n_x + space.n_x // 2
        state[0, nyquist, 0, active[0]] = 1.0
    backend, edge = _attached(simulation, state)
    try:
        native = backend._edge_reading(backend._require_state(), edge, full=True)
        held = backend.observe(field=True).require_field()
        host = edge.reduction.reading(held[0], full=True)
    finally:
        backend.close()
    budget = 4 * fft_bound(space.n_y * space.n_x)
    np.testing.assert_allclose(native.faces, host.faces, rtol=budget)
    np.testing.assert_allclose(native.totals, host.totals, rtol=budget, atol=1e-30)
    np.testing.assert_allclose(native.columns, host.columns, rtol=budget)


def test_a_planar_channel_above_the_first_reads_its_own_column() -> None:
    # A planar column is (y, p), with its own place in the scratch: the energy has to
    # come back on that column, and on the same row's node whichever channel holds it.
    simulation = _case("planar")
    problem = simulation.translate()
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    assert monitor.transverse is not None
    space = monitor.transverse.space
    assert isinstance(space, PlanarGrid)
    assert space.n_p >= 2
    active = np.nonzero(np.asarray(problem.grids.active_mask))[0]
    column = 1 * space.n_p + 1  # y row 1, channel 1
    state = np.zeros(problem.shape, dtype=np.complex128)
    state[0, column, 0, active[0]] = 1.0
    backend, edge = _attached(simulation, state)
    try:
        native = backend._edge_reading(backend._require_state(), edge, full=True)
        host = edge.reduction.reading(state[0], full=True)
    finally:
        backend.close()
    expected = np.zeros(problem.shape[1])
    expected[column] = 1.0
    np.testing.assert_array_equal(native.columns, expected)
    budget = 4 * fft_bound(space.n_y)
    np.testing.assert_allclose(native.totals, host.totals, rtol=budget, atol=1e-30)
