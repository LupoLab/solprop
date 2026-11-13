# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native's reading of a requested collar's strip and current (SEP 44, phase 3).

A plane wave has an exact answer for the current on a uniform grid, which needs neither
reducer: two reducers agreeing on one wrong line would show nothing. The strip is held
to the host's on the same state, and each direction to where a packet actually goes.
"""

import contextlib
import math
import threading
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from backends.edge_collar_cases import (
    INWARD,
    LENGTH,
    ROUTES,
    STEPS,
    collar_case,
)
from bounds import fft_bound
from sessions.runs import load_native, watched_run
from sessions.runs import run as session_run
from solprop import transverse_monitor
from solprop.backends.memory import low_storage_stage_register
from solprop.backends.node_chunk import plan_node_chunk
from solprop.backends.registry import available, select
from solprop.backends.session import Start
from solprop.driver import Simulation
from solprop.exceptions import BoundaryWarning, StopRequested
from solprop.grid import LineGrid
from solprop.log import NullReporter
from solprop.noise import NoiseRealisation
from solprop.output.hdf5 import resume
from solprop.stepper import (
    FixedSchedule,
    LowStorageRungeKutta,
    Multistep,
    RungeKutta,
)
from solprop.translate.routes import transverse_of
from solprop.transverse_monitor import TransverseEdge

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)
NativeBackend = pytest.importorskip("solprop_native.backend").NativeBackend

CURRENT_FLOOR = 1e-14
"""Well above rounding in a share of the launch, and what `INWARD`'s current clears."""


def node_chunk(simulation: Simulation) -> int:
    """Return how many nodes a pass of a simulation's march takes at once."""
    return plan_node_chunk(
        simulation.translate(), simulation.settings.execution_options()
    ).chunk


def _attached(simulation: Simulation, state: np.ndarray) -> tuple[Any, Any]:
    """Return a native backend loaded with `state`, its edge bound.

    The state is loaded into the backend's native state and the monitor bound there,
    as the native session binds its own (the backend's ``_bind_monitor``).
    """
    problem = simulation.translate()
    run_request = simulation._run_request()
    backend = cast(
        "Any",
        select(
            "native",
            problem,
            run_request,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
            options=simulation.settings.execution_options(),
        ),
    )
    load_native(backend, state)
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    assert monitor.transverse is not None
    assert monitor.transverse.every_step
    backend._bind_monitor(monitor)
    return backend, monitor.transverse


def _noise(problem: Any, seed: int) -> np.ndarray:
    """Return white noise on the active band, filling every line."""
    rng = np.random.default_rng(seed)
    shape = problem.shape
    state = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    state[..., ~np.asarray(problem.grids.active_mask)] = 0.0
    return state


@pytest.mark.parametrize("kerr", [False, True], ids=["linear", "padded"])
@pytest.mark.parametrize("route", ROUTES)
def test_native_reads_a_collars_strip_and_current_as_the_host_does(
    route: str, kerr: bool
) -> None:
    simulation = collar_case(route, kerr=kerr)
    problem = simulation.translate()
    if kerr and route != "planar":
        # A radial synthesis carries the padding's scale, and a read must too.
        assert problem.grids.n_to != problem.grids.n_t
    state = _noise(problem, 67)
    backend, edge = _attached(simulation, state)
    try:
        native = backend._edge_reading(
            backend._native_state, edge, current=True, full=True
        )
        host = edge.reduction.reading(state, current=True, full=True)
    finally:
        backend.close()
    # Rounding of one transform round trip and a sum of non-negative terms, each side.
    budget = 4 * fft_bound(state.shape[0] * state.shape[2])
    assert native.strip.shape == host.strip.shape
    assert np.all(host.strip[:, 1] > 0.0)
    np.testing.assert_allclose(native.strip, host.strip, rtol=budget)
    # The current is a slope's product, whose terms nearly cancel per row before they're
    # rectified. A noise state's sum is still of the terms' size, so the bound scales
    # with the transform's, over the slope's factor from the nodes' spacing.
    assert host.current > 0.0
    assert native.current == pytest.approx(host.current, rel=64 * budget)
    np.testing.assert_allclose(native.totals, host.totals, rtol=budget)
    # A later read uses only the collar's rate, from the rows a read can touch.
    backend, edge = _attached(simulation, state)
    try:
        read = backend._edge_reading(backend._native_state, edge)
    finally:
        backend.close()
    assert read.totals[1] > 0.0
    assert read.totals[1] == pytest.approx(
        edge.reduction.reading(state).totals[1], rel=budget
    )


def _line_current(
    bins: Callable[[np.ndarray, float], dict[int, np.ndarray]],
) -> tuple[float, float, float, Any, int]:
    """Read the line case's current from a state of transverse waves in its bins.

    `bins(x, kappa)` maps an active bin's index to its field along the line, for
    kappa forty cycles over the line. Returns the native current, kappa, the spacing
    h, the attached edge and the line's point count.
    """
    simulation = collar_case("line")
    problem = simulation.translate()
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    edge = monitor.transverse
    assert edge is not None
    space = edge.space
    assert isinstance(space, LineGrid)
    x = np.asarray(space.x)
    h = float(x[1] - x[0])
    kappa = 2 * math.pi * 40 / (space.n_x * h)
    active = np.nonzero(np.asarray(problem.grids.active_mask))[0]
    state = np.zeros(problem.shape, dtype=np.complex128)
    for index, wave in bins(x, kappa).items():
        state[:, 0, active[index]] = np.fft.fft(wave)
    backend, edge = _attached(simulation, state)
    try:
        native = backend._edge_reading(
            backend._native_state, edge, faces=False, current=True
        )
    finally:
        backend.close()
    return native.current, kappa, h, edge, space.n_x


def test_a_plane_wave_reads_its_exact_current() -> None:
    # On a uniform grid the central difference of exp(i kappa s) is exactly
    # i sin(kappa h) / h times it, so J = |E|^2 sin(kappa h) / (h k). It flows towards
    # smaller s on the face where it's inward, and the other face rectifies it away.
    current, kappa, h, edge, n_x = _line_current(
        lambda x, kappa: {3: np.exp(1j * kappa * x)}
    )
    expected = math.sin(kappa * h) / h / edge.k_min
    assert current == pytest.approx(expected, rel=64 * fft_bound(n_x))


def test_the_current_is_rectified_per_row_and_not_after_summing() -> None:
    # One bin inward and one outward at one amplitude: their fluxes cancel in a sum,
    # and a current rectified after summing would read nothing on either face.
    current, kappa, h, edge, n_x = _line_current(
        lambda x, kappa: {3: np.exp(1j * kappa * x), 5: np.exp(-1j * kappa * x)}
    )
    expected = 2 * math.sin(kappa * h) / h / edge.k_min
    assert current == pytest.approx(expected, rel=64 * fft_bound(n_x))


def test_a_standing_wave_reads_almost_no_current() -> None:
    # A real field carries no flux, so each row's product cancels to rounding: there the
    # bound is absolute, set by the terms and not by the result.
    current, _, h, edge, n_x = _line_current(
        lambda x, kappa: {3: np.cos(kappa * x).astype(np.complex128)}
    )
    # Unit field: |v| |s| / k, for |s| at most 2 / (2 h), times the transform's
    # rounding, on each of the two lines, which rectification can only keep.
    bound = 2 * fft_bound(n_x) / h / edge.k_min
    assert 0.0 <= current <= bound


def _packet(space: LineGrid, centre: float, kappa: float) -> np.ndarray:
    """Return a Gaussian packet at `centre` with transverse wavenumber `kappa`."""
    x = np.asarray(space.x)
    field = np.exp(-(((x - centre) / 60e-6) ** 2) + 1j * kappa * x)
    return np.fft.fft(field)


@pytest.mark.parametrize(
    ("kappa", "reads_inward"),
    # Well inside this grid's Nyquist of 2.5e5 rad/m, and above the strip's 16 / s.
    [(1.2e5, True), (-1.2e5, False)],
    ids=["inward", "outward"],
)
def test_the_strip_tells_a_packets_direction(kappa: float, reads_inward: bool) -> None:
    # Section 5.3: the windowed strip resolves direction above 16 / s, so a packet in it
    # moving outward reads below the window's leakage of itself.
    simulation = collar_case("line")
    problem = simulation.translate()
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    edge = monitor.transverse
    assert edge is not None
    space = edge.space
    assert isinstance(space, LineGrid)
    bounds = edge.reduction.lines[0]
    x = np.asarray(space.x)
    centre = float(np.mean(x[bounds.strip]))
    active = np.nonzero(np.asarray(problem.grids.active_mask))[0]
    state = np.zeros(problem.shape, dtype=np.complex128)
    state[:, 0, active[3]] = _packet(space, centre, kappa)
    backend, edge = _attached(simulation, state)
    try:
        native = backend._edge_reading(backend._native_state, edge, full=True)
    finally:
        backend.close()
    inward, windowed = native.strip[0]
    assert windowed > 0.0
    assert (inward / windowed > 0.1) == reads_inward
    if not reads_inward:
        assert inward / windowed < 1e-5


@pytest.mark.parametrize("route", ROUTES)
def test_a_small_batch_reads_what_one_batch_reads(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A strip runs in batches of planes to bound its buffer. They only change the order
    # of a non-negative sum, so that much rounding apart. A batch of a row or two here.
    simulation = collar_case(route)
    problem = simulation.translate()
    state = _noise(problem, 71)
    backend, edge = _attached(simulation, state)
    try:
        whole = backend._edge_reading(backend._native_state, edge, full=True)
    finally:
        backend.close()
    monkeypatch.setattr(transverse_monitor, "_STRIP_BATCH_BYTES", 2**12)
    backend, edge = _attached(simulation, state)
    try:
        small = backend._edge_reading(backend._native_state, edge, full=True)
    finally:
        backend.close()
    budget = 4 * fft_bound(state.shape[0] * state.shape[2])
    np.testing.assert_allclose(small.strip, whole.strip, rtol=budget)


@pytest.mark.parametrize("route", ROUTES)
def test_a_collars_memory_is_its_row(route: str) -> None:
    simulation = collar_case(route)
    problem = simulation.translate()
    state = _noise(problem, 73)
    backend, edge = _attached(simulation, state)
    try:
        _, n_pol, n_omega = problem.shape
        expected = transverse_monitor._transverse_edge_bytes(
            edge.space, edge.design, n_pol, n_omega
        )
        assert backend.memory_ledger()["transverse_edge"] == expected
        assert backend.edge_memory(edge.space) == expected
    finally:
        backend.close()


MARCHES = {
    "runge_kutta": (RungeKutta(), None),
    "runge_kutta_split": (RungeKutta(), "split_step"),
    "runge_kutta_in_stage": (RungeKutta(), "nonlinear"),
    "adams_exponential": (Multistep(), None),
    "adams_interaction": (Multistep(formulation="interaction"), None),
    "low_storage": (LowStorageRungeKutta(), None),
    "low_storage_in_stage": (LowStorageRungeKutta(), "nonlinear"),
}
"""Each march and temporal collar scheme a read has to leave undisturbed."""

MEDIUM_READS = {
    ("cartesian", "runge_kutta"),
    ("cartesian", "runge_kutta_split"),
    ("cartesian", "adams_exponential"),
    ("planar", "runge_kutta"),
    ("planar", "runge_kutta_split"),
    ("planar", "runge_kutta_in_stage"),
}
"""The routes and marches whose three runs take over ten seconds (CI, macOS arm64)."""


@pytest.mark.parametrize(
    ("route", "name"),
    [
        pytest.param(
            route,
            name,
            id=f"{route}-{name}",
            marks=[pytest.mark.medium] if (route, name) in MEDIUM_READS else [],
        )
        for route in ROUTES
        for name in MARCHES
    ],
)
def test_a_collars_read_leaves_the_march_where_it_was(route: str, name: str) -> None:
    march, scheme = MARCHES[name]

    # A collar reads at every commit, on the transverse scratch, which every march has
    # to write again before it's used. The native session reads it so; against the
    # same session unwatched, whose march touches no scratch for a read: same steps,
    # one process, so no rounding apart. Its reads and current against the reference's
    # run of the same schedule.
    def final(backend: str, *, watched: bool) -> tuple[bytes, dict[str, Any]]:
        # The planar kind's collar has the much shorter stability limit.
        steps = 16 * STEPS if route == "planar" or scheme == "nonlinear" else STEPS
        schedule = FixedSchedule.uniform(
            0.0, LENGTH, LENGTH / steps, save_planes=[LENGTH]
        )
        simulation = collar_case(
            route,
            **INWARD,
            integrator=march,
            schedule=schedule,
            save_planes=[LENGTH],
            controller=None,
            absorber_scheme=scheme,
            dz_max=LENGTH / steps,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BoundaryWarning)
            if watched:
                field, record, _ = watched_run(simulation, backend)
                return field.tobytes(), cast("dict[str, Any]", record["transverse"])
            problem = simulation.translate()
            made = select(
                backend,
                problem,
                simulation._run_request(),
                grid=simulation.grid,
                modes=simulation.modes,
                profiles=simulation.profiles,
            )
            try:
                return session_run(made, problem).fields[-1].tobytes(), {}
            finally:
                made.close()

    native, record = final("native", watched=True)
    unwatched, _ = final("native", watched=False)
    assert native == unwatched
    _, host_record = final("numpy_ref", watched=True)
    assert record["reads"] == host_record["reads"] >= STEPS // 2
    # The same numbers by two reducers, on two matched marches: their rounding. A
    # planar launch can't converge, so its current stays at rounding, and only its
    # absolute size is left to hold.
    if route == "planar":
        assert record["current"] == pytest.approx(host_record["current"], abs=1e-15)
    else:
        assert host_record["current"] > CURRENT_FLOOR
        assert record["current"] == pytest.approx(
            host_record["current"], rel=1e-8, abs=0.0
        )


@pytest.mark.parametrize(
    "route",
    [
        pytest.param(route, marks=pytest.mark.medium) if route == "cartesian" else route
        for route in ROUTES
    ],
)
def test_a_collar_run_brings_no_state_to_the_host(route: str) -> None:
    # Native's own count of the bytes through its state's crossings.
    simulation = collar_case(route, fields="none")
    field = np.asarray(simulation.translate().initial.field0)
    with simulation.runner() as runner, warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        output = runner.run(field, progress="none")
        assert isinstance(runner.backend, NativeBackend)
        to_host, from_host = cast("Any", runner.backend)._native_state.crossed_bytes()
    record = cast("Any", output).outcome.execution["boundary_monitor"]["transverse"]
    assert record["reads"] > 0
    assert to_host == 0
    assert from_host == field.nbytes


def test_a_lit_node_reads_the_collars_rate_from_its_design() -> None:
    # `totals[1]` is what `_collar_rate` reads at every read: `alpha w |E|^2`. The
    # design gives alpha and the grid gives w, so this needs neither reducer's tables.
    simulation = collar_case("line")
    problem = simulation.translate()
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    edge = monitor.transverse
    assert edge is not None
    space = edge.space
    assert isinstance(space, LineGrid)
    assert edge.design is not None
    rate = np.asarray(edge.design.collar_rate)
    node = int(np.argmax(rate))
    assert rate[node] > 0.0
    phase = np.exp(-2j * np.pi * np.arange(space.n_x) * node / space.n_x)
    active = np.nonzero(np.asarray(problem.grids.active_mask))[0]
    state = np.zeros(problem.shape, dtype=np.complex128)
    state[:, 0, active[3]] = phase
    backend, edge = _attached(simulation, state)
    try:
        read = backend._edge_reading(backend._native_state, edge)
    finally:
        backend.close()
    expected = rate[node] * float(np.asarray(space.spatial_weights)[node])
    assert read.totals[1] == pytest.approx(expected, rel=64 * fft_bound(space.n_x))


class _StopAtFirstPlane(NullReporter):
    """Set a run's stop flag at its first saved plane, between two edge reads."""

    def __init__(self, flag: threading.Event) -> None:
        self.flag = flag

    def plane(self, z: float, summary: object) -> None:
        del z, summary
        self.flag.set()


@pytest.mark.parametrize(
    "march",
    [RungeKutta(), Multistep(), LowStorageRungeKutta()],
    ids=["runge_kutta", "adams", "low_storage"],
)
def test_a_stopped_collar_run_resumes_to_the_whole_runs_record(
    march: object, tmp_path: Path
) -> None:
    # #512: a resumed collar run judged against zero, with its current and peaks
    # restarted. The trapezoid picks up from the resumed state's own sample, so the
    # current is the whole run's. On a fixed schedule, so both take the same steps.
    def build() -> Simulation:
        schedule = FixedSchedule.uniform(
            0.0, LENGTH, LENGTH / STEPS, save_planes=[LENGTH / 2, LENGTH]
        )
        return collar_case(
            "radial", **INWARD, integrator=march, schedule=schedule, controller=None
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        whole = cast("Any", build().run(progress="none"))
        path = tmp_path / "run.h5"
        flag = threading.Event()
        build().run(path, progress=_StopAtFirstPlane(flag), stop=flag)
        build().run(path, resume=True, progress="none")
    resumed = resume(path).outcome
    assert resumed is not None
    expected = whole.outcome.execution["boundary_monitor"]["transverse"]
    found = cast("Any", resumed).execution["boundary_monitor"]["transverse"]
    assert found["launch_energy"] == pytest.approx(expected["launch_energy"], rel=1e-12)
    # Field rounding, carried through a share of a share: nothing's restarted. Each is
    # well above rounding, so that one read as zero would fail here.
    for key in ("current", "strip_max"):
        assert expected[key] > CURRENT_FLOOR, key
        assert found[key] == pytest.approx(expected[key], rel=1e-8, abs=0.0), key
    assert expected["current_baseline"] > CURRENT_FLOOR
    assert found["current_baseline"] == pytest.approx(
        expected["current_baseline"], rel=1e-12, abs=0.0
    )


def test_a_collar_run_stopped_before_its_first_step_resumes_as_a_launch(
    tmp_path: Path,
) -> None:
    # A launch-only record: the run's first step never happened, so the next execution
    # has a launch to continue from and no integral yet.
    def build() -> Simulation:
        return collar_case("radial", integrator=RungeKutta())

    flag = threading.Event()
    flag.set()
    path = tmp_path / "run.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        with contextlib.suppress(StopRequested):
            build().run(path, progress="none", stop=flag)
        whole = cast("Any", build().run(progress="none"))
        build().run(path, resume=True, progress="none")
    resumed = resume(path).outcome
    assert resumed is not None
    expected = whole.outcome.execution["boundary_monitor"]["transverse"]
    found = cast("Any", resumed).execution["boundary_monitor"]["transverse"]
    assert found["launch_energy"] == pytest.approx(expected["launch_energy"], rel=1e-12)
    assert found["reads"] == expected["reads"]


def test_a_collars_low_storage_march_holds_its_stage_register() -> None:
    # A march without one writes the next pass's input straight into the transverse
    # scratch, which is what a read takes. `test_native_edge_reading` holds that one.
    simulation = collar_case(
        "line",
        integrator=LowStorageRungeKutta(),
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / STEPS),
        controller=None,
        dz_max=LENGTH / STEPS,
    )
    assert low_storage_stage_register(simulation.translate(), "native")


@pytest.mark.parametrize("route", ROUTES)
def test_the_smallest_node_chunk_reads_what_the_default_reads(route: str) -> None:
    # A read owes nothing to how a march chunks its nodes, on one state.
    simulation = collar_case(route)
    problem = simulation.translate()
    state = _noise(problem, 83)
    backend, edge = _attached(simulation, state)
    try:
        whole = backend._edge_reading(
            backend._native_state, edge, current=True, full=True
        )
    finally:
        backend.close()
    budget = 20 * 16 * problem.shape[1] * problem.grids.n_to
    small = collar_case(route, node_budget=budget)
    assert node_chunk(small) < node_chunk(simulation)
    backend, edge = _attached(small, state)
    try:
        read = backend._edge_reading(
            backend._native_state, edge, current=True, full=True
        )
    finally:
        backend.close()
    assert read.strip.tobytes() == whole.strip.tobytes()
    assert read.totals.tobytes() == whole.totals.tobytes()
    assert read.current == whole.current


@pytest.mark.parametrize(
    ("route", "march"),
    [
        pytest.param(
            route,
            march,
            id=f"{route}-{name}",
            marks=[pytest.mark.medium] if (route, name) == ("cartesian", "rk") else [],
        )
        for route in ROUTES
        for name, march in (("rk", RungeKutta()), ("adams", Multistep()))
    ],
)
def test_a_collar_run_s_monitor_record_matches_numpy_ref_on_a_matched_schedule(
    route: str, march: object
) -> None:
    """The edge reads, status, current, removed power and strip of the two backends.

    The run's field at its end, on the same schedule, is the row ``geometry edge
    collar ...`` (``backends/geometry_rows.py``).
    """
    # A controlled witness: one schedule, so the same reads at the same places, and a
    # converging beam reaching the collar, so that the current's truly inward and above
    # rounding, and the relative comparison means something.
    steps = 16 * STEPS if route == "planar" else STEPS

    def run(backend: str) -> dict[str, Any]:
        simulation = collar_case(
            route,
            **INWARD,
            backend=backend,
            integrator=march,
            schedule=FixedSchedule.uniform(
                0.0, LENGTH, LENGTH / steps, save_planes=[LENGTH / 2, LENGTH]
            ),
            controller=None,
            dz_max=LENGTH / steps,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BoundaryWarning)
            output = cast("Any", simulation.run(progress="none"))
        return output.outcome.execution["boundary_monitor"]["transverse"]

    reference = run("numpy_ref")
    native = run("native")
    assert native["reads"] == reference["reads"] > 0
    assert native["transverse_status"] == reference["transverse_status"]
    # A planar launch can't converge, so its current stays at rounding: what's left to
    # hold there is its absolute size.
    if route != "planar":
        assert reference["current"] > CURRENT_FLOOR
        assert reference["spatial_removed"] > CURRENT_FLOOR
    for key in ("current", "spatial_removed", "strip_max"):
        assert native[key] == pytest.approx(reference[key], rel=1e-8, abs=1e-15), key


def _unmasked(problem: Any, seed: int) -> NoiseRealisation:
    """Return a hand-built input draw with energy outside the active band."""
    rng = np.random.default_rng(seed)
    shape = problem.shape
    spectrum = 1e-9 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    return NoiseRealisation(spectrum=spectrum, seed=None, model="input")


@pytest.mark.parametrize("route", ["radial", "cartesian", "line"])
def test_a_supplied_unmasked_draw_judges_as_the_host_does_with_no_transfer(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The ramp judges the launch without its noise, which is where the load has both
    # parts. A masked draw would hide the mask's own part of that. The record
    # can't show it unless the stop's reached, so this looks at what's judged. The
    # native session's launch against the reference's, which holds its state on the
    # host and judges the columns it forms there.
    judge = TransverseEdge._judge_ramp_band
    judged: list[np.ndarray] = []

    def seen(self: Any, columns: np.ndarray, *args: Any, **kwargs: Any) -> None:
        judged.append(np.array(columns))
        judge(self, columns, *args, **kwargs)

    monkeypatch.setattr(TransverseEdge, "_judge_ramp_band", seen)
    # The native session hands its native judge the clean columns its load formed,
    # through the readings it forms from them: those, in the node reduction's units as
    # the host edge judges them (`TransverseEdge._column_power`), are its columns.
    transverse = pytest.importorskip("solprop_native._transverse")
    readings = transverse.noise_readings
    handed: list[np.ndarray] = []

    def formed(edge: Any, noise: np.ndarray, clean: np.ndarray | None) -> Any:
        assert clean is not None
        handed.append(np.array(edge._column_power(clean)))
        return readings(edge, noise, clean)

    monkeypatch.setattr(transverse, "noise_readings", formed)

    def launch(name: str) -> tuple[dict[str, Any], int | None]:
        simulation = collar_case(route, noise="input")
        problem = simulation.translate()
        run_request = simulation._run_request()
        assert transverse_of(problem.route) is not None
        backend = cast(
            "Any",
            select(
                name,
                problem,
                run_request,
                grid=simulation.grid,
                modes=simulation.modes,
                profiles=simulation.profiles,
            ),
        )
        try:
            monitor = simulation._boundary_monitor()
            assert monitor is not None
            record: dict[str, Any] = {}
            start = Start.entrance(problem, noise=_unmasked(problem, 89))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", BoundaryWarning)
                # The launch plane, then the first advance, whose launch is judged.
                session_run(
                    backend,
                    problem,
                    run_request.for_run(monitor=monitor),
                    start,
                    export=None,
                    observations=2,
                    monitor=record,
                )
            crossed = (
                backend._native_state.crossed_bytes()[0] if name == "native" else None
            )
            return cast("dict[str, Any]", record["transverse"]), crossed
        finally:
            backend.close()

    native, to_host = launch("native")
    assert to_host == 0
    host, _ = launch("numpy_ref")
    for key in ("launch_energy", "noise_allowance", "noise_share_in_ramp"):
        assert native[key] == pytest.approx(host[key], rel=1e-11, abs=1e-30), key
    assert native["noise_share_in_ramp"] > 0.0
    # The native session judges its launch natively, from the clean columns its load
    # formed, and records the judgement compared above; the reference judges the
    # columns it forms on the host, which hold light in the ramp's band. A column's
    # own rounding is that of adding the draw and taking it away again.
    assert len(judged) == 1
    assert len(handed) == 1
    assert np.max(judged[0]) > 0.0
    np.testing.assert_allclose(handed[0], judged[0], rtol=1e-10, atol=1e-30)


def test_a_resume_with_no_monitor_record_crosses_once() -> None:
    # The one launch whose clean columns the load didn't form: a resumed state with its
    # draw, whose run has no record of the monitor, so it's judged here as a launch.
    simulation = collar_case("line", noise="input")
    problem = simulation.translate()
    run_request = simulation._run_request()
    backend = cast(
        "Any",
        select(
            "native",
            problem,
            run_request,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        ),
    )
    try:
        state = np.asarray(problem.initial.field0)
        monitor = simulation._boundary_monitor()
        assert monitor is not None
        start = Start(
            field=state,
            z=float(problem.initial.z),
            noise=_unmasked(problem, 97),
            resume=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BoundaryWarning)
            # The resumed plane, then the first advance, whose launch is judged.
            session_run(
                backend,
                problem,
                run_request.for_run(monitor=monitor),
                start,
                export=None,
                observations=2,
            )
        assert backend._native_state.crossed_bytes()[0] == state.nbytes
    finally:
        backend.close()


def test_a_strip_of_too_few_nodes_reads_nothing_on_either_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A line too short to read a direction beside the real ones: it holds no buffer and
    # its pair is zero, in the same place in the reading.
    simulation = collar_case("line")
    problem = simulation.translate()
    state = _noise(problem, 101)
    lines = transverse_monitor._edge_lines

    def with_an_empty_line(*args: Any) -> tuple[Any, ...]:
        found = lines(*args)
        short = np.array([1.0, 2.0, 3.0])
        return (*found, transverse_monitor._edge_line("x", short, 1.0, (0.0, 4.0), 2.0))

    monkeypatch.setattr(transverse_monitor, "_edge_lines", with_an_empty_line)
    backend, edge = _attached(simulation, state)
    try:
        read = backend._edge_reading(backend._native_state, edge, full=True)
        held = edge.reduction.reading(state, full=True)
        _, n_pol, n_omega = problem.shape
        assert backend.memory_ledger()["transverse_edge"] == (
            transverse_monitor._transverse_edge_bytes(
                edge.space, edge.design, n_pol, n_omega
            )
        )
    finally:
        backend.close()
    assert read.strip.shape == held.strip.shape
    assert read.strip[-1].tolist() == held.strip[-1].tolist() == [0.0, 0.0]
    assert np.all(read.strip[:-1, 1] > 0.0)


def test_a_strip_too_short_for_any_inward_bin_reads_only_its_window() -> None:
    # Direction needs |q| >= 16 / s. Four nodes span three spacings, so that's beyond
    # Nyquist's pi / h, and the strip has a window and no inward range.
    line = transverse_monitor._edge_line("x", np.arange(8.0), 1.0, (2.0, 5.0), 6.0)
    assert line.strip.size == 4
    assert line.inward == (0, 0)
