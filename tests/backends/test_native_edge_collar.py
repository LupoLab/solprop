# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native's reading of a requested collar's strip and current (SEP 44, phase 3).

A plane wave has an exact answer for the current on a uniform grid, which needs neither
reducer: two reducers agreeing on one wrong line would show nothing. The strip is held
to the host's on the same state, and each direction to where a packet actually goes.
"""

import math
import warnings
from typing import Any, cast

import numpy as np
import pytest

from bounds import fft_bound
from solprop import transverse_monitor
from solprop.backends.driver import StepDriver
from solprop.backends.registry import available, select
from solprop.boundaries import SpatialAbsorber
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BoundaryWarning
from solprop.grid import EnvelopeGrid, FreeGrid, LineGrid, RadialGrid
from solprop.modes import PlanarGrid, SlabMode
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import (
    FixedSchedule,
    LowStorageRungeKutta,
    Multistep,
    PIController,
    RungeKutta,
)

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


def _case(route: str, **overrides: Any) -> Simulation:
    """Return a linear run with a requested collar outside its aperture."""
    grid = EnvelopeGrid.from_limits(LENGTH, 800e-9, (700e-9, 900e-9), 200e-15)
    temporal = GaussianPulse(wavelength=800e-9, fwhm=50e-15, energy=1e-9)
    settings = _settings(planar=route == "planar", **overrides)
    if route == "planar":
        modes = [SlabMode(63.5e-6, "Ar", 1.0, order=p) for p in (1, 3)]
        return Simulation.from_planar(
            EnvelopeGrid.from_limits(LENGTH, 800e-9, (650e-9, 1000e-9), 600e-15),
            PlanarGrid(modes, 1e-4, 40, collar_fraction=1.0),
            [],
            PlanarPulse(
                GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3),
                20e-6,
                waist_x=0.735 * 63.5e-6,
            ),
            settings=settings,
        )
    pulse = SpatialPulse(temporal, 40e-6)
    if route == "radial":
        return Simulation.from_radial(
            grid,
            RadialGrid(300e-6, 48, collar_fraction=1.0),
            SpatialMedium(1.0),
            [],
            pulse,
            settings=settings,
            noise=None,
        )
    space = (
        LineGrid(300e-6, 96, collar_fraction=1.0)
        if route == "line"
        else FreeGrid(300e-6, 64, 360e-6, 80, collar_fraction=1.0)
    )
    return Simulation.from_free(
        grid, space, SpatialMedium(1.0), [], pulse, settings=settings, noise=None
    )


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


@pytest.mark.parametrize("route", ROUTES)
def test_native_reads_a_collars_strip_and_current_as_the_host_does(route: str) -> None:
    simulation = _case(route)
    problem = simulation.translate()
    state = _noise(problem, 67)
    backend, edge = _attached(simulation, state)
    try:
        native = backend._edge_reading(
            backend._require_state(), edge, current=True, full=True
        )
        host = edge.reduction.reading(state[0], current=True, full=True)
    finally:
        backend.close()
    # Rounding of one transform round trip and a sum of non-negative terms, each side.
    budget = 4 * fft_bound(state.shape[1] * state.shape[3])
    assert native.strip.shape == host.strip.shape
    assert np.all(host.strip[:, 1] > 0.0)
    np.testing.assert_allclose(native.strip, host.strip, rtol=budget)
    # The current is a slope's product, whose terms nearly cancel per row before they're
    # rectified. A noise state's sum is still of the terms' size, so the bound scales
    # with the transform's, over the slope's factor from the nodes' spacing.
    assert host.current > 0.0
    assert native.current == pytest.approx(host.current, rel=64 * budget)
    np.testing.assert_allclose(native.totals, host.totals, rtol=budget)


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
    x = np.asarray(space.x)
    h = float(x[1] - x[0])
    mode = 40
    kappa = 2 * math.pi * mode / (space.n_x * h)
    wave = np.exp(1j * kappa * x)
    active = np.nonzero(np.asarray(problem.grids.active_mask))[0]
    state = np.zeros(problem.shape, dtype=np.complex128)
    state[0, :, 0, active[3]] = np.fft.fft(wave)
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
    x = np.asarray(space.x)
    h = float(x[1] - x[0])
    mode = 40
    kappa = 2 * math.pi * mode / (space.n_x * h)
    active = np.nonzero(np.asarray(problem.grids.active_mask))[0]
    state = np.zeros(problem.shape, dtype=np.complex128)
    state[0, :, 0, active[3]] = np.fft.fft(np.exp(1j * kappa * x))
    state[0, :, 0, active[5]] = np.fft.fft(np.exp(-1j * kappa * x))
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
    bounds = edge.reduction.lines[0]
    x = np.asarray(space.x)
    centre = float(np.mean(x[bounds.strip]))
    active = np.nonzero(np.asarray(problem.grids.active_mask))[0]
    state = np.zeros(problem.shape, dtype=np.complex128)
    state[0, :, 0, active[3]] = _packet(space, centre, kappa)
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
    monkeypatch.setattr(transverse_monitor, "STRIP_BATCH_BYTES", 2**12)
    backend, edge = _attached(simulation, state)
    try:
        small = backend._edge_reading(backend._require_state(), edge, full=True)
    finally:
        backend.close()
    budget = 4 * fft_bound(state.shape[1] * state.shape[3])
    np.testing.assert_allclose(small.strip, whole.strip, rtol=budget)


@pytest.mark.parametrize("route", ROUTES)
def test_a_collars_memory_is_its_row(route: str) -> None:
    simulation = _case(route)
    problem = simulation.translate()
    state = _noise(problem, 73)
    backend, edge = _attached(simulation, state)
    try:
        _, _, n_pol, n_omega = problem.shape
        expected = transverse_monitor.transverse_edge_bytes(
            edge.space, edge.design, n_pol, n_omega
        )
        assert backend.memory_ledger()["transverse_edge"] == expected
        assert backend.edge_memory(edge.space) == expected
    finally:
        backend.close()


@pytest.mark.parametrize(
    "march",
    [RungeKutta(), Multistep(), LowStorageRungeKutta()],
    ids=["runge_kutta", "adams", "low_storage"],
)
@pytest.mark.parametrize("route", ROUTES)
def test_a_collars_read_leaves_the_march_where_it_was(
    route: str, march: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A collar reads at every commit, on the transverse scratch, which every march has
    # to write again before it's used. Against the same run read on the host, which
    # touches no scratch: same steps, same monitor, one process, so no rounding apart.
    def final() -> tuple[bytes, dict[str, Any]]:
        # The planar kind's collar has the much shorter stability limit.
        steps = 16 * STEPS if route == "planar" else STEPS
        schedule = FixedSchedule.uniform(
            0.0, LENGTH, LENGTH / steps, save_planes=[LENGTH]
        )
        simulation = _case(
            route,
            integrator=march,
            schedule=schedule,
            save_planes=[LENGTH],
            controller=None,
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
    # The same numbers by two reducers: their rounding, on a beam far from the edge.
    assert record["current"] == pytest.approx(
        host_record["current"], rel=1e-8, abs=1e-15
    )


@pytest.mark.parametrize("route", ROUTES)
def test_a_collar_run_brings_no_state_to_the_host(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(self: Any, state: Any) -> np.ndarray:
        raise AssertionError("a field-free collar run brought the state to the host")

    monkeypatch.setattr(NativeBackend, "_observed_field", refused)
    simulation = _case(route, fields="none")
    field = np.asarray(simulation.translate().initial.field0)[0]
    with simulation.runner() as runner, warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        output = runner.run(field, progress="none")
        assert isinstance(runner.backend, NativeBackend)
        to_host, from_host = cast("Any", runner.backend)._native_state.crossed_bytes()
    record = cast("Any", output).outcome.execution["boundary_monitor"]["transverse"]
    assert record["reads"] > 0
    assert to_host == 0
    assert from_host == field.nbytes
