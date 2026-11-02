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

import math

import numpy as np
import pytest

from solprop.backends.registry import available
from solprop.boundaries import SPLIT_CROSSING, split_crossing_step
from solprop.constants import C
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import TranslationError
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper.multistep import Multistep
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.linop import constant_phi, walk_off_rate
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


def test_the_limit_is_half_the_collar_over_the_walk_off() -> None:
    simulation = _gnlse("numpy_ref", Multistep())
    problem = simulation.translate()
    rate = walk_off_rate(problem.linear, GRID.omega, GRID.omega0, GRID.inband_mask)
    width = min(GRID.t_collar.width_low, GRID.t_collar.width_high)
    # The band's bluest bin walks fastest: beta2 times its offset from the reference,
    # to the differences' rounding of a quadratic phase.
    farthest = float(np.max(np.abs(GRID.omega[GRID.inband_mask] - GRID.omega0)))
    assert rate == pytest.approx(BETA2 * farthest, rel=1e-6)
    limit = split_crossing_step(rate, width)
    assert limit == pytest.approx(SPLIT_CROSSING * width / rate)
    assert problem.stepper.dz_max == limit
    assert problem.stepper.dz_init <= limit


def test_the_in_stage_scheme_keeps_its_step_bounds() -> None:
    problem = _gnlse(
        "numpy_ref", Multistep(), scheme="nonlinear", dz_max=1.0
    ).translate()
    assert problem.stepper.dz_max == 1.0


def test_a_fixed_schedule_past_the_limit_is_refused() -> None:
    schedule = FixedSchedule.uniform(0.0, 1.0, 1 / 16, save_planes=[1.0])
    simulation = _gnlse("numpy_ref", RungeKutta(), schedule=schedule)
    with pytest.raises(TranslationError, match=r"crossing limit 1\.1"):
        simulation.translate()


def test_a_fixed_schedule_within_the_limit_runs() -> None:
    schedule = FixedSchedule.uniform(0.0, 1.0, 1 / 100, save_planes=[1.0])
    simulation = _gnlse("numpy_ref", RungeKutta(), schedule=schedule)
    assert simulation.translate().stepper.schedule is not None


def test_nothing_walking_sets_no_limit() -> None:
    assert math.isinf(split_crossing_step(0.0, 1e-13))


def test_a_reference_at_the_axis_edge_reads_no_rate() -> None:
    # No centred difference reaches it.
    omega = np.linspace(1.0e15, 3.0e15, 11)
    table = constant_phi(-1j * BETA2 * omega**2, 1.0)
    assert walk_off_rate(table, omega, float(omega[0]), np.ones(11, dtype=bool)) == 0.0
