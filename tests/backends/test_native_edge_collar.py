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
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from bounds import fft_bound
from solprop import transverse_monitor
from solprop.backends.driver import StepDriver
from solprop.backends.memory import low_storage_stage_register
from solprop.backends.registry import available, select
from solprop.boundaries import SpatialAbsorber
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BoundaryWarning, StopRequested
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.log import NullReporter
from solprop.modes import PlanarGrid, SlabMode
from solprop.monitor import BoundaryCheck
from solprop.noise import NoiseRealisation
from solprop.nonlinear.responses import KerrResponse
from solprop.output.hdf5 import resume
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import (
    FixedSchedule,
    LowStorageRungeKutta,
    Multistep,
    PIController,
    RungeKutta,
)
from solprop.translate.routes import transverse_of
from solprop.transverse_monitor import TransverseEdge

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)
NativeBackend = pytest.importorskip("solprop_native.backend").NativeBackend

LENGTH = 2e-3
"""The tiny cases' length, m."""

ROUTES = ("radial", "cartesian", "line", "planar")
"""Every route a requested collar runs on."""

STEPS = 8
"""Fixed steps over the tiny cases: short enough for the collar's stability limit."""


INWARD: dict[str, Any] = {
    "waist": 60e-6,
    "propz": -2.4e-2,
    "boundary_check": BoundaryCheck(budget=1.0, stop_factor=1.0, transverse_budget=1.0),
}
"""A converging beam reaching the collar, whose current is inward and above rounding.

A narrow beam's current is 1e-30 and a comparison of it passes whatever it reads.
"""

CURRENT_FLOOR = 1e-14
"""Well above rounding in a share of the launch, and what `INWARD`'s current clears."""


def _settings(*, planar: bool = False, **overrides: Any) -> SimulationSettings:
    """Return settings for a linear run with a requested collar on native."""
    keywords: dict[str, Any] = {
        "controller": PIController(rtol=1e-7, atol=0.0),
        "save_planes": [LENGTH / 2, LENGTH],
        "dz_max": LENGTH / 8,
        "dz_init": LENGTH * 1e-3,
        "backend": "native",
        "spatial_absorber": SpatialAbsorber(kappa=2.5e5 if planar else 8e4),
    }
    return SimulationSettings(**(keywords | overrides))


def _case(
    route: str,
    *,
    kerr: bool = False,
    waist: float = 40e-6,
    propz: float = 0.0,
    noise: str | None = None,
    **overrides: Any,
) -> Simulation:
    """Return a run with a requested collar outside its aperture.

    Parameters
    ----------
    route : str
        The route.
    kerr : bool, optional
        Whether the medium has a Kerr response, on a field grid, whose time is padded.
    waist : float, optional
        The beam's waist on a 300 um aperture, m; the planar case's scales with its own.
    propz : float, optional
        The launch's signed distance from the waist, m: negative is converging. A
        planar launch is at the guide's entrance, so it ignores this.
    noise : str or None, optional
        The noise model; the planar route takes none.
    **overrides : Any
        Settings to change.

    Returns
    -------
    solprop.driver.Simulation
        The run.
    """
    grid = (
        FieldGrid.from_limits(LENGTH, 800e-9, (600e-9, 1.2e-6), 150e-15)
        if kerr
        else EnvelopeGrid.from_limits(LENGTH, 800e-9, (700e-9, 900e-9), 200e-15)
    )
    temporal = GaussianPulse(wavelength=800e-9, fwhm=50e-15, energy=1e-9)
    settings = _settings(planar=route == "planar", **overrides)
    responses: list[Any] = (
        [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)] if kerr else []
    )
    medium = SpatialMedium("Ar", 1.0) if kerr else SpatialMedium(1.0)
    if route == "planar":
        modes = [SlabMode(63.5e-6, "Ar", 1.0, order=p) for p in (1, 3)]
        return Simulation.from_planar(
            EnvelopeGrid.from_limits(LENGTH, 800e-9, (650e-9, 1000e-9), 600e-15),
            PlanarGrid(modes, 1e-4, 40, collar_fraction=1.0),
            responses,
            PlanarPulse(
                GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3),
                waist / 2.0,
                waist_x=0.735 * 63.5e-6,
            ),
            settings=settings,
        )
    pulse = SpatialPulse(temporal, waist, propz=propz)
    if route == "radial":
        return Simulation.from_radial(
            grid,
            RadialGrid(300e-6, 48, collar_fraction=1.0),
            medium,
            responses,
            pulse,
            settings=settings,
            noise=cast("Any", noise),
        )
    space = (
        LineGrid(300e-6, 96, collar_fraction=1.0)
        if route == "line"
        else FreeGrid(300e-6, 64, 360e-6, 80, collar_fraction=1.0)
    )
    return Simulation.from_free(
        grid,
        space,
        medium,
        responses,
        pulse,
        settings=settings,
        noise=cast("Any", noise),
    )


def node_chunk(problem: Any) -> int:
    """Return how many nodes a pass of a problem's march takes at once."""
    return int(problem.transform.node_chunk)


def _attached(simulation: Simulation, state: np.ndarray) -> tuple[Any, Any]:
    """Return a native backend loaded with `state`, its edge bound."""
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
    simulation = _case(route, kerr=kerr)
    problem = simulation.translate()
    if kerr and route != "planar":
        # A radial synthesis carries the padding's scale, and a read must too.
        assert problem.grids.n_to != problem.grids.n_t
    state = _noise(problem, 67)
    backend, edge = _attached(simulation, state)
    try:
        native = backend._edge_reading(
            backend._require_state(), edge, current=True, full=True
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
        read = backend._edge_reading(backend._require_state(), edge)
    finally:
        backend.close()
    assert read.totals[1] > 0.0
    assert read.totals[1] == pytest.approx(
        edge.reduction.reading(state).totals[1], rel=budget
    )


def test_a_plane_wave_reads_its_exact_current() -> None:
    # On a uniform grid the central difference of exp(i kappa s) is exactly
    # i sin(kappa h) / h times it, so J = |E|^2 sin(kappa h) / (h k). It flows towards
    # smaller s on the face where it's inward, and the other face rectifies it away.
    simulation = _case("line")
    problem = simulation.translate()
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    edge = monitor.transverse
    assert edge is not None
    space = edge.space
    assert isinstance(space, LineGrid)
    x = np.asarray(space.x)
    h = float(x[1] - x[0])
    mode = 40
    kappa = 2 * math.pi * mode / (space.n_x * h)
    wave = np.exp(1j * kappa * x)
    active = np.nonzero(np.asarray(problem.grids.active_mask))[0]
    state = np.zeros(problem.shape, dtype=np.complex128)
    state[:, 0, active[3]] = np.fft.fft(wave)
    backend, edge = _attached(simulation, state)
    try:
        native = backend._edge_reading(
            backend._require_state(), edge, faces=False, current=True
        )
    finally:
        backend.close()
    expected = math.sin(kappa * h) / h / edge.k_min
    assert native.current == pytest.approx(expected, rel=64 * fft_bound(space.n_x))


def test_the_current_is_rectified_per_row_and_not_after_summing() -> None:
    # One bin inward and one outward at one amplitude: their fluxes cancel in a sum,
    # and a current rectified after summing would read nothing on either face.
    simulation = _case("line")
    problem = simulation.translate()
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    edge = monitor.transverse
    assert edge is not None
    space = edge.space
    assert isinstance(space, LineGrid)
    x = np.asarray(space.x)
    h = float(x[1] - x[0])
    mode = 40
    kappa = 2 * math.pi * mode / (space.n_x * h)
    active = np.nonzero(np.asarray(problem.grids.active_mask))[0]
    state = np.zeros(problem.shape, dtype=np.complex128)
    state[:, 0, active[3]] = np.fft.fft(np.exp(1j * kappa * x))
    state[:, 0, active[5]] = np.fft.fft(np.exp(-1j * kappa * x))
    backend, edge = _attached(simulation, state)
    try:
        native = backend._edge_reading(
            backend._require_state(), edge, faces=False, current=True
        )
    finally:
        backend.close()
    expected = 2 * math.sin(kappa * h) / h / edge.k_min
    assert native.current == pytest.approx(expected, rel=64 * fft_bound(space.n_x))


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
    simulation = _case("line")
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
        native = backend._edge_reading(backend._require_state(), edge, full=True)
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
    simulation = _case(route)
    problem = simulation.translate()
    state = _noise(problem, 71)
    backend, edge = _attached(simulation, state)
    try:
        whole = backend._edge_reading(backend._require_state(), edge, full=True)
    finally:
        backend.close()
    monkeypatch.setattr(transverse_monitor, "_STRIP_BATCH_BYTES", 2**12)
    backend, edge = _attached(simulation, state)
    try:
        small = backend._edge_reading(backend._require_state(), edge, full=True)
    finally:
        backend.close()
    budget = 4 * fft_bound(state.shape[0] * state.shape[2])
    np.testing.assert_allclose(small.strip, whole.strip, rtol=budget)


@pytest.mark.parametrize("route", ROUTES)
def test_a_collars_memory_is_its_row(route: str) -> None:
    simulation = _case(route)
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


@pytest.mark.parametrize("name", MARCHES)
@pytest.mark.parametrize("route", ROUTES)
def test_a_collars_read_leaves_the_march_where_it_was(
    route: str, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    march, scheme = MARCHES[name]

    # A collar reads at every commit, on the transverse scratch, which every march has
    # to write again before it's used. Against the same run read on the host, which
    # touches no scratch: same steps, same monitor, one process, so no rounding apart.
    def final() -> tuple[bytes, dict[str, Any]]:
        # The planar kind's collar has the much shorter stability limit.
        steps = 16 * STEPS if route == "planar" or scheme == "nonlinear" else STEPS
        schedule = FixedSchedule.uniform(
            0.0, LENGTH, LENGTH / steps, save_planes=[LENGTH]
        )
        simulation = _case(
            route,
            **INWARD,
            integrator=march,
            schedule=schedule,
            save_planes=[LENGTH],
            controller=None,
            absorber_scheme=scheme,
            dz_max=LENGTH / steps,
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", BoundaryWarning)
                backend.attach_monitor(monitor)
                backend.propagate_to(LENGTH)
            record = cast("dict[str, Any]", monitor.summary())["transverse"]
            return backend.observe(field=True).require_field().tobytes(), record
        finally:
            backend.close()

    native, record = final()
    with monkeypatch.context() as patch:
        patch.setattr(NativeBackend, "_edge_reading", StepDriver._edge_reading)
        host, host_record = final()
    assert native == host
    assert record["reads"] == host_record["reads"] >= STEPS // 2
    # The same numbers by two reducers: their rounding. A planar launch can't converge,
    # so its current stays at rounding, and only its absolute size is left to hold.
    if route == "planar":
        assert record["current"] == pytest.approx(host_record["current"], abs=1e-15)
    else:
        assert host_record["current"] > CURRENT_FLOOR
        assert record["current"] == pytest.approx(
            host_record["current"], rel=1e-8, abs=0.0
        )


@pytest.mark.parametrize("route", ROUTES)
def test_a_collar_run_brings_no_state_to_the_host(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(self: Any, state: Any) -> np.ndarray:
        raise AssertionError("a field-free collar run brought the state to the host")

    monkeypatch.setattr(NativeBackend, "_observed_field", refused)
    simulation = _case(route, fields="none")
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
    simulation = _case("line")
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
        read = backend._edge_reading(backend._require_state(), edge)
    finally:
        backend.close()
    expected = rate[node] * float(np.asarray(space.spatial_weights)[node])
    assert read.totals[1] == pytest.approx(expected, rel=64 * fft_bound(space.n_x))


def test_a_standing_wave_reads_almost_no_current() -> None:
    # A real field carries no flux, so each row's product cancels to rounding: there the
    # bound is absolute, set by the terms and not by the result.
    simulation = _case("line")
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
    state[:, 0, active[3]] = np.fft.fft(np.cos(kappa * x).astype(np.complex128))
    backend, edge = _attached(simulation, state)
    try:
        native = backend._edge_reading(
            backend._require_state(), edge, faces=False, current=True
        )
    finally:
        backend.close()
    # Unit field: |v| |s| / k, for |s| at most 2 / (2 h), times the transform's
    # rounding, on each of the two lines, which rectification can only keep.
    bound = 2 * fft_bound(space.n_x) / h / edge.k_min
    assert 0.0 <= native.current <= bound


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
        return _case(
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
        return _case("radial", integrator=RungeKutta())

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
    simulation = _case(
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
    simulation = _case(route)
    problem = simulation.translate()
    state = _noise(problem, 83)
    backend, edge = _attached(simulation, state)
    try:
        whole = backend._edge_reading(
            backend._require_state(), edge, current=True, full=True
        )
    finally:
        backend.close()
    budget = 20 * 16 * problem.shape[1] * problem.grids.n_to
    small = _case(route, node_budget=budget)
    assert node_chunk(small.translate()) < node_chunk(problem)
    backend, edge = _attached(small, state)
    try:
        read = backend._edge_reading(
            backend._require_state(), edge, current=True, full=True
        )
    finally:
        backend.close()
    assert read.strip.tobytes() == whole.strip.tobytes()
    assert read.totals.tobytes() == whole.totals.tobytes()
    assert read.current == whole.current


@pytest.mark.parametrize("march", [RungeKutta(), Multistep()], ids=["rk", "adams"])
@pytest.mark.parametrize("route", ROUTES)
def test_a_collar_run_matches_numpy_ref_on_a_matched_schedule(
    route: str, march: object
) -> None:
    # A controlled witness: one schedule, so the same reads at the same places, and a
    # converging beam reaching the collar, so that the current's truly inward and above
    # rounding, and the relative comparison means something.
    steps = 16 * STEPS if route == "planar" else STEPS

    def run(backend: str) -> tuple[np.ndarray, dict[str, Any]]:
        simulation = _case(
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
        record = output.outcome.execution["boundary_monitor"]["transverse"]
        return np.asarray(output["field"][-1]), record

    expected, reference = run("numpy_ref")
    found, native = run("native")
    assert np.linalg.norm(found - expected) <= 1e-10 * np.linalg.norm(expected)
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
    # can't show it unless the stop's reached, so this looks at what's judged.
    judge = TransverseEdge._judge_ramp_band
    judged: list[np.ndarray] = []

    def seen(self: Any, columns: np.ndarray, *args: Any, **kwargs: Any) -> None:
        judged.append(np.array(columns))
        judge(self, columns, *args, **kwargs)

    monkeypatch.setattr(TransverseEdge, "_judge_ramp_band", seen)

    def launch() -> tuple[dict[str, Any], int]:
        simulation = _case(route, noise="input")
        problem = simulation.translate()
        assert transverse_of(problem.route) is not None
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
            backend.set_initial(problem.initial.field0, _unmasked(problem, 89))
            monitor = simulation._boundary_monitor()
            assert monitor is not None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", BoundaryWarning)
                backend.attach_monitor(monitor)
            record = cast("dict[str, Any]", monitor.summary())["transverse"]
            return record, backend._native_state.crossed_bytes()[0]
        finally:
            backend.close()

    native, to_host = launch()
    assert to_host == 0
    with monkeypatch.context() as patch:
        patch.setattr(NativeBackend, "_edge_reading", StepDriver._edge_reading)
        patch.setattr(NativeBackend, "_edge_clean_columns", _host_clean_columns)
        host, _ = launch()
    for key in ("launch_energy", "noise_allowance", "noise_share_in_ramp"):
        assert native[key] == pytest.approx(host[key], rel=1e-11, abs=1e-30), key
    assert native["noise_share_in_ramp"] > 0.0
    # A column's own rounding is that of adding the draw and taking it away again.
    assert len(judged) == 2
    assert np.max(judged[0]) > 0.0
    np.testing.assert_allclose(judged[0], judged[1], rtol=1e-10, atol=1e-30)


def _host_clean_columns(self: Any, edge: Any) -> np.ndarray | None:
    """Return a launch's clean columns the way a state on the host gives them."""
    floor = self._input_floor
    if floor is None or edge.resumed:
        return None
    state = np.asarray(self._observed_field(self._require_state())[0])
    return edge.reduction.column_power(state - np.asarray(floor[0]))


def test_a_resume_with_no_monitor_record_crosses_once() -> None:
    # The one launch whose clean columns the load didn't form: a resumed state with its
    # draw, whose run has no record of the monitor, so it's judged here as a launch.
    simulation = _case("line", noise="input")
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
        state = np.asarray(problem.initial.field0)
        backend.set_initial(state, _unmasked(problem, 97), resume=True)
        monitor = simulation._boundary_monitor()
        assert monitor is not None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BoundaryWarning)
            backend.attach_monitor(monitor)
        assert backend._native_state.crossed_bytes()[0] == state.nbytes
    finally:
        backend.close()


def test_a_strip_of_too_few_nodes_reads_nothing_on_either_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A line too short to read a direction beside the real ones: it holds no buffer and
    # its pair is zero, in the same place in the reading.
    simulation = _case("line")
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
        read = backend._edge_reading(backend._require_state(), edge, full=True)
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
