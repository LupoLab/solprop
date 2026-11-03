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
    split_bucket_rates,
    split_crossing_step,
)
from solprop.constants import C
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BoundaryError, TranslationError
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
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

BACKENDS = ["numpy_ref", "numba_cpu", "native"]

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
        r"next step 5\.000e-02 m",
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
@pytest.mark.parametrize("backend", ["numba_cpu", "native"])
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
    buckets = walk_off_buckets(table, omega, np.ones(201, dtype=bool))
    rates = split_bucket_rates()[buckets[:, 150]]  # 5e14 rad/s from the reference
    # Mode 0's fastest is 2e-26, mode 1's 4e-26, times the offset; a bucket's edge.
    for rate, fastest in zip(rates, (2e-26 * 5e14, 4e-26 * 5e14), strict=True):
        assert fastest <= rate < 1.2 * fastest


def test_the_axis_ends_and_inactive_bins_are_the_slowest() -> None:
    omega = np.linspace(1.0e15, 3.0e15, 11)
    table = constant_phi(-1j * BETA2 * omega**2, 1.0)
    active = np.ones(11, dtype=bool)
    active[5] = False
    buckets = walk_off_buckets(table, omega, active)[0]
    last = SPLIT_RATE_BUCKETS - 1
    assert buckets[0] == buckets[-1] == last
    # An inactive bin, and the bins whose differences reach it.
    assert np.all(buckets[4:7] == last)
    assert np.all(buckets[1:4] < last)
