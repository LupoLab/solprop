# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The split temporal collar's crossing limit (``solprop.boundaries.SPLIT_CROSSING``).

The ``"split_step"`` scheme damps the collar once per accepted step. A step whose
group-delay walk-off exceeds the collar's width carries light through it and round the
periodic window undamped, and on a linear stretch the error estimate sees nothing:
``Multistep`` grew a linear GNLSE pulse's steps to 0.3 m and left 0.12 of its launch
wrapped into a window it had left, and a 1 pJ radial pulse in 60 bar of argon left
0.34. Translation holds adaptive steps to ``0.5 w / max |d tau / dz|`` and refuses a
fixed schedule past it; the in-stage scheme keeps its bounds.
"""

from __future__ import annotations

import dataclasses
import logging
import math

import numpy as np
import pytest

import solprop.translate.problem as problem_module
from monitor_cases import as_driver
from solprop.backends.registry import available, select
from solprop.boundaries import (
    SPLIT_CROSSING,
    SPLIT_RATE_BUCKETS,
    SplitCrossing,
    split_bucket_rates,
    split_crossing_step,
    split_rate_buckets,
)
from solprop.constants import C
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BackendError, BoundaryError, TranslationError
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.monitor import BoundaryCheck
from solprop.nonlinear.responses import KerrResponse
from solprop.output import MemoryOutput
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium, critical_power
from solprop.stepper.multistep import Multistep
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.linop import constant_phi, walk_off_buckets
from solprop.translate.masks import AbsorberScheme

BACKENDS = ["numpy_ref", "native"]

BETA2 = 2e-26
"""s^2/m: the GNLSE pulse walks by its carrier offset under this dispersion."""

LEFT = 1e-3
"""The most of the launch a window the light has walked out of may still hold. A
Gaussian of 200 fs walking one realised window (1423 fs) over the metre leaves about
2e-4 of itself inside; light wrapped back into it left 1e-2 to 0.3."""

GRID = EnvelopeGrid.from_limits(1.0, 1e-6, (0.8e-6, 1.25e-6), 1e-12)


def _require(backend: str) -> None:
    """Skip unless `backend` is installed."""
    if backend not in available():
        pytest.skip(f"{backend} is not installed")


def _gnlse(
    backend: str,
    integrator: Multistep | RungeKutta,
    scheme: AbsorberScheme = "split_step",
    *,
    dz_max: float | None = None,
    schedule: FixedSchedule | None = None,
) -> Simulation:
    """Return a linear pulse whose group delay walks one realised window over 1 m."""
    window = GRID.n_t * GRID.dt
    offset = window / BETA2
    return Simulation.from_gnlse(
        1e-30, 1.0, [0.0, 0.0, BETA2], wavelength=1e-6, grid=GRID,
        pulses=[GaussianPulse(
            wavelength=2 * np.pi * C / (2 * np.pi * C / 1e-6 + offset),
            fwhm=200e-15, peak_power=1.0,
        )],
        noise=None, raman=False, shock=False, backend=backend, integrator=integrator,
        absorber_scheme=scheme, boundary_check=None, dz_max=dz_max, schedule=schedule,
        save_planes=None if schedule is None else [1.0],
    )  # fmt: skip


def _radial(backend: str) -> Simulation:
    """Return a 1 pJ pulse at 1150 nm in 60 bar of argon on a 400 fs window."""
    grid = FieldGrid.from_limits(1.0, 800e-9, (500e-9, 1.3e-6), 400e-15)
    return Simulation.from_radial(
        grid,
        RadialGrid(1.2e-2, 16),
        SpatialMedium("Ar", 60.0),
        [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)],
        SpatialPulse(
            GaussianPulse(wavelength=1150e-9, fwhm=40e-15, energy=1e-12), 2e-3
        ),
        settings=SimulationSettings(
            save_planes=[1.0],
            backend=backend,
            dz_max=1.0,
            dz_init=1e-3,
            boundary_check=None,
        ),
        noise=None,
    )


def _left(simulation: Simulation) -> float:
    """Return the share of its launch energy a run ends with in its window."""
    fields = np.asarray(simulation.run(progress="none").fields("fft"))
    return float(np.sum(np.abs(fields[-1]) ** 2) / np.sum(np.abs(fields[0]) ** 2))


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_gnlse_pulse_that_walked_out_leaves_the_window(backend: str) -> None:
    # Multistep doubles a low-source run's steps; unclamped they reached 0.3 m.
    _require(backend)
    assert _left(_gnlse(backend, Multistep())) < LEFT


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_radial_pulse_that_walked_out_leaves_the_window(backend: str) -> None:
    # The spatial route's default scheme and integrator, dz_max as long as the run.
    _require(backend)
    simulation = _radial(backend)
    assert simulation.translate().responses.absorber_scheme == "split_step"
    assert _left(simulation) < LEFT


def test_the_launch_s_limit_holds_dz_max() -> None:
    simulation = _gnlse("numpy_ref", Multistep())
    problem = simulation.translate()
    crossing = problem.split_crossing
    assert crossing is not None
    width = min(GRID.t_collar.width_low, GRID.t_collar.width_high)
    assert crossing.width == width
    # The launch's light walks at about its carrier offset's beta2 (1.4e-12 s/m, the
    # window over the metre), not at the band's edge (9.4e-12 s/m): the limit is the
    # pulse's, within a bucket's 18 % and the pulse's spectral width.
    rate = SPLIT_CROSSING * width / crossing.launch_limit
    assert GRID.n_t * GRID.dt < rate < 2.0 * GRID.n_t * GRID.dt
    limit = crossing.limit(crossing.occupancy(problem.initial.field0))
    assert crossing.launch_limit == limit
    assert problem.stepper.dz_max == limit
    assert problem.stepper.dz_init <= limit


def test_the_in_stage_scheme_keeps_its_step_bounds() -> None:
    problem = _gnlse(
        "numpy_ref", Multistep(), scheme="nonlinear", dz_max=1.0
    ).translate()
    assert problem.split_crossing is None
    assert problem.stepper.dz_max == 1.0


def test_a_fixed_schedule_past_the_launch_s_limit_is_refused() -> None:
    schedule = FixedSchedule.uniform(0.0, 1.0, 1 / 8, save_planes=[1.0])
    simulation = _gnlse("numpy_ref", RungeKutta(), schedule=schedule)
    with pytest.raises(TranslationError, match=r"crossing limit 5\.\d+e-02 m"):
        simulation.translate()


def test_a_fixed_schedule_within_the_launch_s_limit_runs() -> None:
    schedule = FixedSchedule.uniform(0.0, 1.0, 1 / 100, save_planes=[1.0])
    simulation = _gnlse("numpy_ref", RungeKutta(), schedule=schedule)
    assert simulation.translate().stepper.schedule is not None


def _spreading(backend: str) -> Simulation:
    """Return a GNLSE pulse whose self-phase modulation spreads it into fast bins.

    Launched, its 500 fs spectrum walks slowly: the limit is 0.89 m. Self-phase
    modulation of 1.5 rad per 5 cm step broadens it, and by 15 cm it reaches bins
    that walk 20 times faster against a 2 ps window's collar.
    """
    grid = EnvelopeGrid.from_limits(2.0, 1e-6, (0.8e-6, 1.25e-6), 2e-12)
    return Simulation.from_gnlse(
        0.03, 2.0, [0.0, 0.0, BETA2], wavelength=1e-6, grid=grid,
        pulses=[GaussianPulse(wavelength=1e-6, fwhm=500e-15, peak_power=1e3)],
        noise=None, raman=False, shock=False, backend=backend,
        integrator=RungeKutta(), absorber_scheme="split_step",
        schedule=FixedSchedule.uniform(0.0, 2.0, 0.05, save_planes=[2.0]),
        save_planes=[2.0],
    )  # fmt: skip


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_fixed_schedule_the_light_outgrows_is_stopped(backend: str) -> None:
    # Admitted on the launch, stopped at the first check whose cap is below its
    # 5 cm steps, naming the cap.
    _require(backend)
    simulation = _spreading(backend)
    crossing = simulation.translate().split_crossing
    assert crossing is not None and crossing.launch_limit > 0.05
    with pytest.raises(
        BoundaryError,
        match=r"crossing limit fell to \d\.\d+e-02 m, below the fixed schedule's "
        r"step 5\.000e-02 m before the next check",
    ):
        simulation.run(progress="none")


def _beam(backend: str, nodes: int, trange: float, fraction: float) -> Simulation:
    """Return a 100 um beam in argon on a 1 mm radial grid, 5 cm, adaptive."""
    length = 0.05
    grid = FieldGrid.from_limits(length, 800e-9, (500e-9, 1.3e-6), trange)
    medium = SpatialMedium("Ar", 1.0)
    chi3 = gamma3("Ar", wavelength=800e-9)
    density = float(medium.density(np.array([0.0]))[0])
    index = float(np.real(medium.index(np.array([2 * math.pi * C / 800e-9]))[0]))
    power = fraction * critical_power(800e-9, chi3 * density, index=index)
    energy = power * 50e-15 * math.sqrt(math.pi / (4 * math.log(2)))
    return Simulation.from_radial(
        grid,
        RadialGrid(1e-3, nodes),
        medium,
        [KerrResponse(chi3, thg=False)],
        SpatialPulse(
            GaussianPulse(wavelength=800e-9, fwhm=50e-15, energy=energy), 100e-6
        ),
        settings=SimulationSettings(
            save_planes=[length],
            backend=backend,
            dz_max=length,
            dz_init=length * 1e-4,
            auto_init=True,
        ),
        noise=None,
    )


def _accepted(simulation: Simulation) -> int:
    """Return how many steps a run accepted."""
    run = simulation.run(progress="none")
    assert isinstance(run, MemoryOutput)
    return sum(
        int(np.count_nonzero(plane.step_history.accepted))
        for plane in run.planes()
        if plane.step_history is not None
    )


@pytest.mark.parametrize(
    ("nodes", "trange"),
    [(256, 800e-15), (256, 500e-15), (512, 800e-15), (512, 500e-15)],
    ids=["256-169fs", "256-70fs", "512-169fs", "512-70fs"],
)
@pytest.mark.parametrize("backend", ["numpy_ref", "native"])
def test_a_free_space_beam_keeps_its_steps(
    backend: str, nodes: int, trange: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The outer columns walk fastest and hold none of the beam: read over every bin,
    # they held these runs to 4.9e-4 m and 4.4 times the steps. Read where the light
    # is, the limit stays above every step the runs take.
    _require(backend)
    clamped = _accepted(_beam(backend, nodes, trange, 0.01))
    monkeypatch.setattr(problem_module, "_split_crossing", lambda *_: None)
    assert clamped == _accepted(_beam(backend, nodes, trange, 0.01))


def test_the_cap_tightens_as_the_beam_self_focuses() -> None:
    # Half a critical power: the beam's spectrum and its transverse wavenumbers
    # widen, and the cap read at the checks falls to about half the launch's.
    simulation = _beam("numpy_ref", 256, 500e-15, 0.5)
    problem = simulation.translate()
    crossing = problem.split_crossing
    assert crossing is not None
    backend = select("numpy_ref", problem, grid=simulation.grid)
    try:
        driver = as_driver(backend)
        driver.set_initial(problem.initial.field0, None)
        monitor = simulation._boundary_monitor()
        assert monitor is not None
        driver.attach_monitor(monitor)
        assert monitor.step_cap == crossing.launch_limit
        tightest = monitor.step_cap
        for z in np.linspace(0.005, 0.05, 10):
            backend.propagate_to(float(z))
            tightest = min(tightest, monitor.step_cap)
        assert tightest < 0.7 * crossing.launch_limit
    finally:
        backend.close()


def test_nothing_walking_sets_no_limit() -> None:
    assert math.isinf(split_crossing_step(0.0, 1e-13))


def test_lowering_dz_max_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="solprop.translate.problem"):
        _gnlse("numpy_ref", Multistep(), dz_max=1.0).translate()
    assert any("crossing limit holds dz_max" in one.message for one in caplog.records)


def test_a_dz_max_within_the_limit_is_kept_and_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="solprop.translate.problem"):
        problem = _gnlse("numpy_ref", Multistep(), dz_max=5e-3).translate()
    assert problem.stepper.dz_max == 5e-3
    assert not any("crossing limit" in one.message for one in caplog.records)


def test_every_batch_member_and_polarisation_is_read() -> None:
    # Two members, two modes, two polarisations: each mode's bin takes its fastest.
    omega = np.linspace(1.0e15, 3.0e15, 201)
    quadratic = -1j * 0.5 * (omega - 2.0e15) ** 2
    beta2 = np.array(
        [[[1e-26, 2e-26], [3e-26, 1e-26]], [[1e-26, 1e-26], [1e-26, 4e-26]]]
    )
    table = constant_phi(beta2[..., None] * quadratic, 1.0)
    buckets, _ = walk_off_buckets(table, omega, np.ones(201, dtype=bool))
    rates = split_bucket_rates()[buckets[:, 150]]  # 5e14 rad/s from the reference
    # Mode 0's fastest is 2e-26, mode 1's 4e-26, times the offset; a bucket's edge.
    for rate, fastest in zip(rates, (2e-26 * 5e14, 4e-26 * 5e14), strict=True):
        assert fastest <= rate < 1.2 * fastest


def test_edge_bins_keep_their_rates_and_inactive_ones_are_the_slowest() -> None:
    # The phase beta2 omega^2 / 2 walks at beta2 omega, fastest at the axis's top. The
    # bins beside an inactive one, and the axis's ends, read one-sided differences.
    omega = np.linspace(1.0e15, 3.0e15, 11)
    table = constant_phi(-1j * 0.5 * BETA2 * omega**2, 1.0)
    active = np.ones(11, dtype=bool)
    active[5] = False
    buckets, fastest = walk_off_buckets(table, omega, active)
    rates = split_bucket_rates(fastest)[buckets[0]]
    assert buckets[0, 5] == SPLIT_RATE_BUCKETS - 1
    # One-sided at the ends and beside the gap: the forward or backward difference of
    # a quadratic, beta2 times the pair's midpoint, within a bucket's edge.
    spacing = float(omega[1] - omega[0])
    for index, middle in (
        (0, omega[0] + 0.5 * spacing),
        (4, omega[4] - 0.5 * spacing),
        (6, omega[6] + 0.5 * spacing),
        (10, omega[10] - 0.5 * spacing),
    ):
        assert BETA2 * middle <= rates[index] < 1.2 * BETA2 * middle
    assert fastest == pytest.approx(BETA2 * (omega[10] - 0.5 * spacing))


def test_a_rate_past_the_fastest_edge_is_counted_at_its_own() -> None:
    # A group index 300 from the frame's walks past the buckets' top edge: the open
    # first bucket takes that rate, so its limit is no longer than its own.
    buckets = split_rate_buckets(np.array([[1e-6, 1e-12]]))
    crossing = SplitCrossing(buckets, 1e-13, fastest=1e-6)
    occupancy = crossing.occupancy(np.array([[[[1.0, 1.0]]]], dtype=complex))
    assert crossing.limit(occupancy) == pytest.approx(split_crossing_step(1e-6, 1e-13))


def test_a_varying_guide_is_read_at_its_own_nodes() -> None:
    # A tapered capillary's operator takes the z table: each interval is read at its
    # start, middle and end, the end from the interval it closes.
    from solprop.profiles import PiecewiseLinear  # noqa: PLC0415
    from solprop.translate.linop import _rate_positions  # noqa: PLC0415

    taper = PiecewiseLinear(
        np.array([0.0, 0.5, 1.0]), np.array([100e-6, 100e-6, 60e-6])
    )
    simulation = Simulation.from_capillary(
        taper, 1.0, "He", 1.5, taper_coupling=False, wavelength=800e-9,
        fwhm=10e-15, energy=1e-9, trange=250e-15,
        wavelength_limits=(200e-9, 3e-6), absorber_scheme="split_step", noise=None,
    )  # fmt: skip
    linear = simulation.translate().linear
    assert linear.phi_constant is None
    edges = np.asarray(linear.phi_breakpoints)
    positions = _rate_positions(linear)
    assert len(positions) == 3 * (len(edges) - 1)
    assert [z for z, before, _ in positions if not before][::2] == edges[:-1].tolist()
    assert [z for z, before, _ in positions if before] == edges[1:].tolist()
    assert len(_rate_positions(_radial("numpy_ref").translate().linear)) == 1


def _scheduled(backend: str, steps: np.ndarray) -> Simulation:
    """Return the spreading GNLSE pulse on a schedule of `steps`."""
    length = float(np.sum(steps))
    grid = EnvelopeGrid.from_limits(length, 1e-6, (0.8e-6, 1.25e-6), 2e-12)
    return Simulation.from_gnlse(
        0.03, length, [0.0, 0.0, BETA2], wavelength=1e-6, grid=grid,
        pulses=[GaussianPulse(wavelength=1e-6, fwhm=500e-15, peak_power=1e3)],
        noise=None, raman=False, shock=False, backend=backend,
        integrator=RungeKutta(), absorber_scheme="split_step",
        schedule=FixedSchedule(steps), save_planes=[length],
    )  # fmt: skip


def test_a_longer_step_before_the_next_check_is_refused() -> None:
    # The cap holds until the next check. Checked every step (a busy edge), the next
    # step, 1 cm, is the only one held to it; checked every eighth step, the
    # schedule's third, 5 cm, is held to it too, and refused.
    simulation = _scheduled("numpy_ref", np.array([0.01, 0.01, 0.05, 0.05]))
    problem = simulation.translate()
    backend = select("numpy_ref", problem, grid=simulation.grid)
    try:
        driver = as_driver(backend)
        driver.set_initial(problem.initial.field0, None)
        monitor = simulation._boundary_monitor()
        assert monitor is not None
        driver.attach_monitor(monitor)
        monitor._cap = 0.03
        monitor._busy = True
        assert monitor.steps_to_check() == 1
        driver._refuse_scheduled_crossing(monitor)
        monitor._busy = False
        assert monitor.steps_to_check() >= 3
        with pytest.raises(BoundaryError, match=r"schedule's step 5\.000e-02 m"):
            driver._refuse_scheduled_crossing(monitor)
    finally:
        backend.close()


def test_a_state_that_walks_past_the_schedule_is_refused_when_watched() -> None:
    # A resume: the state the schedule continues from has spread a thousandth of its
    # energy to the band's blue edge, whose limit is below its 5 cm steps.
    simulation = _scheduled("numpy_ref", np.full(4, 0.05))
    problem = simulation.translate()
    grid = simulation.grid
    edge = int(np.argmax(np.where(grid.active_mask, grid.omega, -np.inf)))
    state = np.array(problem.initial.field0)
    total = float(np.sum(np.abs(state) ** 2))
    state[..., edge - 2 : edge] = math.sqrt(5e-4 * total)
    backend = select("numpy_ref", problem, grid=grid)
    try:
        driver = as_driver(backend)
        driver.set_initial(state, None)
        # Relaxed: the spread light fills the window, guards included.
        simulation.settings = dataclasses.replace(
            simulation.settings,
            boundary_check=BoundaryCheck(budget=1.0, stop_factor=1.0),
        )
        monitor = simulation._boundary_monitor()
        assert monitor is not None
        with pytest.raises(BoundaryError, match="crossing limit fell to"):
            driver.attach_monitor(monitor)
    finally:
        backend.close()


def test_no_light_sets_no_cap() -> None:
    crossing = SplitCrossing(np.zeros((1, 4), dtype=np.uint8), 1e-13)
    assert math.isinf(crossing.limit(np.zeros(SPLIT_RATE_BUCKETS)))


def test_a_split_reading_without_the_occupancy_is_refused() -> None:
    simulation = _gnlse("numpy_ref", Multistep())
    problem = simulation.translate()
    simulation.settings = dataclasses.replace(
        simulation.settings, boundary_check=BoundaryCheck()
    )
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    monitor.crossing = problem.split_crossing
    reading = monitor.energies(problem.initial.field0)
    with pytest.raises(BackendError, match="occupancy"):
        monitor.launch(reading, 0.0)


def _taper(coupling: bool) -> Simulation:
    """Return a three-mode helium capillary taper under the split, coupled or not."""
    from solprop.profiles import PiecewiseLinear  # noqa: PLC0415

    taper = PiecewiseLinear(
        np.array([0.0, 0.05, 0.1]), np.array([100e-6, 100e-6, 90e-6])
    )
    return Simulation.from_capillary(
        taper, 0.1, "He", 1.5, taper_coupling=coupling, wavelength=800e-9,
        fwhm=10e-15, energy=1e-9, trange=250e-15,
        wavelength_limits=(400e-9, 2e-6), absorber_scheme="split_step",
        noise=None, modes=3, dz_max=2e-3,
    )  # fmt: skip


def test_a_coupled_taper_s_channels_walk_at_their_own_rates() -> None:
    # A coupled flow's operator carries the carrier's D for every channel, and each
    # channel's own propagation constant in the flow's generator: read with it, the
    # channels take the buckets the adiabatic route's own D gives them, and the
    # third mode, walking about fifteen times faster than the first, its own.
    coupled = _taper(coupling=True).translate()
    adiabatic = _taper(coupling=False).translate()
    assert coupled.linear.coupled_flow is not None
    crossing, reference = coupled.split_crossing, adiabatic.split_crossing
    assert crossing is not None and reference is not None
    np.testing.assert_array_equal(crossing.buckets, reference.buckets)
    inband = coupled.grids.inband_mask
    assert np.min(crossing.buckets[2][inband]) < np.min(crossing.buckets[0][inband])
    # Light in the third mode alone bounds the cap below light in the first alone.
    launch = np.asarray(coupled.initial.field0)
    alone = [np.zeros_like(launch) for _ in range(2)]
    alone[0][:, 0] = launch[:, 0]
    alone[1][:, 2] = launch[:, 0]
    first, third = (crossing.limit(crossing.occupancy(state)) for state in alone)
    assert third < 0.2 * first


def test_a_factored_coupled_flow_is_refused_under_the_split() -> None:
    from solprop.translate.linop import _channel_rate  # noqa: PLC0415

    linear = _taper(coupling=True).translate().linear
    flow = linear.coupled_flow
    assert flow is not None and flow.generator is not None
    n_modes, n_omega = flow.n_modes, len(flow.omega)
    factored = dataclasses.replace(
        flow,
        generator=None,
        fixed_terms=np.zeros((1, n_modes, n_modes, n_omega), dtype=complex),
        term_weights=np.zeros((*flow.generator.shape[:2], 1), dtype=complex),
        connection=None,
    )
    table = dataclasses.replace(linear, coupled_flow=factored)
    with pytest.raises(TranslationError, match="factored"):
        _channel_rate(table, 0.0, False, None)
