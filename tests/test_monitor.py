# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The boundary monitor (SEP 21, decisions D5 and D7).

A linear envelope pulse given a group-delay offset walks across the time window; the
monitor sees it reach the guard, warns, and stops the run, while a pulse that stays
clear leaves every occupancy at the floor. With the stop relaxed, the energy the
collar removed, integrated by the monitor's trapezoid over committed steps, matches the
energy the run actually lost.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Any, cast

import numpy as np
import pytest

from solprop.backends.registry import available
from solprop.constants import C
from solprop.driver import Simulation
from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.grid import EnvelopeGrid
from solprop.monitor import CADENCE, CHECKS, QUIET, BoundaryCheck, BoundaryMonitor
from solprop.output.schema import Output, PlaneStore
from solprop.pulses import GaussianPulse

LENGTH = 1.0
TRANGE = 1e-12
BETA2 = 2e-26
"""s^2/m: the walk comes from a carrier offset, and the dispersion length of the 200 fs
pulse, about 1.4 m, keeps it compact over the metre."""


def walker(
    drift: float,
    *,
    check: BoundaryCheck | None = None,
    backend: str = "numpy_ref",
    trange: float = TRANGE,
    peak_power: float = 1.0,
    noise: bool = False,
) -> Simulation:
    """Return a linear pulse whose group delay walks ``drift`` seconds over the fibre.

    The frame moves with the reference wavelength, so the pulse walks by launching it
    at a carrier offset ``Omega`` under ``beta2``: ``drift = beta2 Omega L``.
    """
    grid = EnvelopeGrid.from_limits(LENGTH, 1e-6, (0.8e-6, 1.25e-6), trange)
    omega0 = 2 * np.pi * C / 1e-6
    offset = drift / (BETA2 * LENGTH)
    simulation = Simulation.from_gnlse(
        1e-30, LENGTH, [0.0, 0.0, BETA2], lambda0=1e-6, grid=grid,
        pulses=[GaussianPulse(
            wavelength=2 * np.pi * C / (omega0 + offset), fwhm=200e-15,
            peak_power=peak_power,
        )],
        noise="input" if noise else None, seed=7 if noise else None,
        raman=False, shock=False, n_save=3, backend=backend,
    )  # fmt: skip
    import dataclasses  # noqa: PLC0415

    simulation.settings = dataclasses.replace(
        simulation.settings,
        boundary_check=BoundaryCheck() if check is None else check,
    )
    return simulation


def record_of(run: Output) -> dict[str, float]:
    """Return a finished run's boundary-monitor record."""
    assert isinstance(run, PlaneStore)
    outcome = run.outcome
    assert outcome is not None and outcome.execution is not None
    record = outcome.execution["boundary_monitor"]
    assert isinstance(record, dict)
    # What was never measured is recorded as None; NaN compares as unmeasured here.
    return {
        name: math.nan if value is None else float(value)
        for name, value in record.items()
        if name != "planes"
    }


def test_the_check_refuses_bad_thresholds() -> None:
    with pytest.raises(BoundaryError, match="guard_fraction"):
        BoundaryCheck(guard_fraction=0.5)
    with pytest.raises(BoundaryError, match="budget"):
        BoundaryCheck(budget=0.0)
    with pytest.raises(BoundaryError, match="stop_factor"):
        BoundaryCheck(stop_factor=0.5)


def test_a_pulse_that_stays_clear_leaves_the_guards_empty() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", BoundaryWarning)
        # Over the metre dispersion spreads the pulse to about 0.4 ps; a 3 ps
        # window keeps its tails clear of the collar.
        run = walker(0.0, trange=3 * TRANGE).run(progress="none")
    record = record_of(run)
    # Rounding of the pulse's far tails, nothing more.
    assert record["guard_low_max"] < 1e-12
    assert record["guard_high_max"] < 1e-12
    assert record["removed_fraction"] < 1e-12
    assert record["steps"] > 0


def test_a_walking_pulse_is_stopped_at_the_guard() -> None:
    # One window's walk-off: the pulse reaches the guard well inside the fibre.
    with (
        pytest.warns(BoundaryWarning, match="guard occupancy"),
        pytest.raises(BoundaryError, match="guards"),
    ):
        walker(1.2 * TRANGE).run(progress="none")


def test_the_split_scheme_s_removal_is_read_exactly() -> None:
    # Applied once per accepted step, the split's removal is read back from each
    # committed state; it is the run's own energy deficit, not a quadrature of it.
    relaxed = BoundaryCheck(budget=1.0, stop_factor=1.0)
    simulation = walker(0.65 * TRANGE, check=relaxed)
    simulation.settings = dataclasses.replace(
        simulation.settings, absorber_scheme="split_step"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        run = simulation.run(progress="none")
    energy = np.asarray(run["stats/energy"])[:, 0, 0]
    lost = 1.0 - energy[-1] / energy[0]
    assert lost > 1e-3
    assert record_of(run)["removed_fraction"] == pytest.approx(lost, rel=1e-3)


def test_every_saved_plane_has_a_row() -> None:
    relaxed = BoundaryCheck(budget=1.0, stop_factor=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        run = walker(0.65 * TRANGE, check=relaxed).run(progress="none")
    assert isinstance(run, PlaneStore)
    assert run.outcome is not None and run.outcome.execution is not None
    record = cast("dict[str, Any]", run.outcome.execution["boundary_monitor"])
    rows = record["planes"]
    np.testing.assert_allclose(rows["z"], np.asarray(run["z"]))
    # The guards' maxima over the planes are the run's; removal only grows.
    assert max(rows["guard_high"]) == record["guard_high_max"]
    assert np.all(np.diff(rows["removed_fraction"]) >= 0.0)
    assert rows["removed_fraction"][-1] == record["removed_fraction"]
    assert rows["removed_fraction"][0] == 0.0


def test_without_a_check_nothing_is_watched() -> None:
    simulation = walker(1.2 * TRANGE)
    import dataclasses  # noqa: PLC0415

    simulation.settings = dataclasses.replace(simulation.settings, boundary_check=None)
    run = simulation.run(progress="none")
    assert isinstance(run, PlaneStore)
    assert run.outcome is not None and run.outcome.execution is not None
    assert "boundary_monitor" not in run.outcome.execution


def test_a_launch_in_the_guard_warns() -> None:
    grid = EnvelopeGrid.from_limits(LENGTH, 1e-6, (0.8e-6, 1.25e-6), TRANGE)
    t = grid.t
    state = np.exp(-(((t - 0.98 * t.max()) / 30e-15) ** 2)).astype(complex)
    spectrum = np.fft.fft(state)[None, None, None, :]
    monitor = BoundaryMonitor.build(grid, None, BoundaryCheck())
    with pytest.warns(BoundaryWarning, match="launched field"):
        monitor.launch(spectrum, 0.0)
    launched = monitor.summary()["launch_guard_high"]
    assert isinstance(launched, float)
    assert launched > 0.5


def test_the_removed_fraction_is_the_energy_the_run_lost() -> None:
    # Stop relaxed: the pulse walks fully into the collar and is absorbed; with no
    # other loss the run's own energy deficit is the collar's removal.
    relaxed = BoundaryCheck(budget=1.0, stop_factor=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        run = walker(0.65 * TRANGE, check=relaxed).run(progress="none")
    energy = np.asarray(run["stats/energy"])[:, 0, 0]
    lost = 1.0 - energy[-1] / energy[0]
    removed = record_of(run)["removed_fraction"]
    assert lost > 1e-3
    assert removed == pytest.approx(lost, rel=0.05)


@pytest.mark.parametrize("backend", ["numba_cpu", "native"])
def test_the_backends_keep_the_same_record(backend: str) -> None:
    if backend not in available():
        pytest.skip(f"{backend} is not installed")
    relaxed = BoundaryCheck(budget=1.0, stop_factor=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        reference = walker(0.65 * TRANGE, check=relaxed).run(progress="none")
        compiled = walker(0.65 * TRANGE, check=relaxed, backend=backend).run(
            progress="none"
        )
    ours = record_of(compiled)
    theirs = record_of(reference)
    for name in ("guard_low_max", "guard_high_max", "collar_max", "removed_fraction"):
        assert ours[name] == pytest.approx(theirs[name], rel=1e-6, abs=1e-15), name


def test_a_field_absorbed_to_nothing_is_not_judged() -> None:
    # Rounding residue spread over the window would read as full guards; below a
    # millionth of a millionth of the launch the monitor no longer judges the state.
    grid = EnvelopeGrid.from_limits(LENGTH, 1e-6, (0.8e-6, 1.25e-6), TRANGE)
    pulse = np.exp(-((grid.t / 50e-15) ** 2)).astype(complex)
    monitor = BoundaryMonitor.build(grid, None, BoundaryCheck())
    monitor.launch(np.fft.fft(pulse)[None, None, None, :], 0.0)
    residue = np.full(grid.n_t, 1e-12, dtype=complex)
    monitor.step(np.fft.fft(residue)[None, None, None, :], 0.1)
    assert monitor.summary()["guard_low_max"] == 0.0


def test_the_shot_noise_floor_is_not_read_as_light_at_the_edges() -> None:
    # A weak pulse: one photon per mode puts about 1e-3 of the field's energy in the
    # guards from the launch on, which the monitor judged as edge occupancy before
    # it learned the floor (SEP 21, Q24).
    with warnings.catch_warnings():
        warnings.simplefilter("error", BoundaryWarning)
        run = walker(0.0, trange=3 * TRANGE, peak_power=1e-3, noise=True).run(
            progress="none"
        )
    record = record_of(run)
    assert record["noise_floor"] > 1e-4
    assert record["raw_guard_max"] > 1e-5
    assert record["guard_low_max"] == record["guard_high_max"] == 0.0
    assert record["removed_fraction"] < 1e-6


def test_light_above_the_floor_is_still_stopped_at_the_guard() -> None:
    # The flow is linear, so a kilowatt walks as a watt does, a million times above
    # the floor. (At a watt the light that reaches the guard, after the collar, is
    # below the floor's own share of it, which the monitor rightly does not judge.)
    with (
        pytest.warns(BoundaryWarning, match="guard occupancy"),
        pytest.raises(BoundaryError, match="guards"),
    ):
        walker(1.2 * TRANGE, peak_power=1e3, noise=True).run(progress="none")


def test_with_noise_the_removed_fraction_is_still_the_signal_lost() -> None:
    relaxed = BoundaryCheck(budget=1.0, stop_factor=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        quiet = walker(0.65 * TRANGE, check=relaxed).run(progress="none")
        noisy = walker(0.65 * TRANGE, check=relaxed, noise=True).run(progress="none")
    assert record_of(noisy)["removed_fraction"] == pytest.approx(
        record_of(quiet)["removed_fraction"], rel=0.02
    )


def test_a_launch_of_the_floor_alone_is_not_judged() -> None:
    grid = EnvelopeGrid.from_limits(LENGTH, 1e-6, (0.8e-6, 1.25e-6), TRANGE)
    rng = np.random.default_rng(3)
    draw = np.exp(2j * np.pi * rng.random(grid.n_t))[None, None, None, :]
    monitor = BoundaryMonitor.build(grid, None, BoundaryCheck())
    with warnings.catch_warnings():
        warnings.simplefilter("error", BoundaryWarning)
        monitor.launch(draw, 0.0, floor=draw)
        monitor.step(draw, 0.1)
    summary = monitor.summary()
    assert summary["guard_low_max"] == summary["guard_high_max"] == 0.0
    assert summary["noise_floor"] == pytest.approx(1.0)


def _with_check(simulation: Simulation, check: BoundaryCheck | None) -> Simulation:
    simulation.settings = dataclasses.replace(simulation.settings, boundary_check=check)
    return simulation


def _exit(run: Output) -> np.ndarray:
    return np.asarray(run.fields("fft")[-1])


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_a_quiet_collar_is_left_out_without_changing_the_run(backend: str) -> None:
    # A pulse clear of the edges: the monitor finds the collar quiet, the stages leave
    # its term out, and the field is the always-on collar's to the threshold's
    # amplitude: the omitted term acts on a collar holding less than QUIET of the
    # energy, so it moves the field by less than sqrt(QUIET) of its norm. The two
    # adaptive runs otherwise differ by rounding and step selection, about 3e-9.
    if backend not in available():
        pytest.skip(f"{backend} is not installed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        watched = walker(0.0, trange=3 * TRANGE, backend=backend).run(progress="none")
        always = _with_check(walker(0.0, trange=3 * TRANGE, backend=backend), None).run(
            progress="none"
        )
    a, b = _exit(watched), _exit(always)
    assert np.linalg.norm(a - b) <= np.sqrt(QUIET) * np.linalg.norm(b)
    record = record_of(watched)
    # Every committed step is either checked or counted as skipped (the planes'
    # checks add to the checked ones).
    accepted = int(np.asarray(watched["step_history/accepted"], dtype=bool).sum())
    assert record["steps"] + record["skipped"] >= accepted


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_light_reaching_a_quiet_collar_switches_it_back_on(backend: str) -> None:
    # A walking pulse: the collar is quiet at first and absorbs once light arrives,
    # as the always-on collar does, to the threshold's amplitude.
    if backend not in available():
        pytest.skip(f"{backend} is not installed")
    relaxed = BoundaryCheck(budget=1.0, stop_factor=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        watched = walker(0.65 * TRANGE, check=relaxed, backend=backend).run(
            progress="none"
        )
        always = _with_check(walker(0.65 * TRANGE, backend=backend), None).run(
            progress="none"
        )
    energy = [np.asarray(run["stats/energy"])[-1, 0, 0] for run in (watched, always)]
    launched = np.asarray(always["stats/energy"])[0, 0, 0]
    assert energy[0] < 0.999 * launched
    assert energy[0] == pytest.approx(energy[1], rel=1e-5)
    a, b = _exit(watched), _exit(always)
    assert np.linalg.norm(a - b) <= 1e-5 * np.linalg.norm(b)


def test_a_check_is_due_every_step_while_light_is_near_an_edge() -> None:
    grid = EnvelopeGrid.from_limits(LENGTH, 1e-6, (0.8e-6, 1.25e-6), TRANGE)
    monitor = BoundaryMonitor.build(grid, None, BoundaryCheck())
    centred = np.exp(-((grid.t / 50e-15) ** 2)).astype(complex)
    edge = np.exp(-(((grid.t - 0.97 * grid.t.max()) / 30e-15) ** 2)).astype(complex)
    spectrum = lambda samples: np.fft.fft(samples)[None, None, None, :]  # noqa: E731
    monitor.launch(spectrum(centred), 0.0)
    assert monitor.collar_quiet
    due = [monitor.due(1e-4 * k) or monitor.skip() for k in range(1, CADENCE)]
    assert not any(due)
    assert monitor.due(1e-4 * CADENCE)
    # However few the steps, a quiet window is checked every length / CHECKS.
    monitor.step(spectrum(centred), 1e-4 * CADENCE)
    assert not monitor.due(1e-4 * CADENCE + 0.5 * LENGTH / CHECKS)
    assert monitor.due(1e-4 * CADENCE + LENGTH / CHECKS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        with pytest.raises(BoundaryError):
            monitor.step(spectrum(centred + edge), 0.1)
    assert not monitor.collar_quiet
    assert monitor.due(0.1)


def test_the_check_may_change_after_translation() -> None:
    simulation = walker(0.0)
    simulation.translate()
    import dataclasses  # noqa: PLC0415

    simulation.settings = dataclasses.replace(simulation.settings, boundary_check=None)
    assert simulation.settings.boundary_check is None
