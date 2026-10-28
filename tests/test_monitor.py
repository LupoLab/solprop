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
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from solprop.backends.registry import available
from solprop.constants import C
from solprop.driver import Simulation, prop_gnlse
from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.grid import EnvelopeGrid
from solprop.log import NullReporter
from solprop.monitor import (
    CADENCE,
    CHECKS,
    QUIET_CHECKS,
    BoundaryCheck,
    BoundaryEnergies,
    BoundaryMonitor,
)
from solprop.output import MemoryOutput
from solprop.output.hdf5 import resume
from solprop.output.schema import Output, PlaneStore, RunOutcome, TerminalCheckpoint
from solprop.pulses import GaussianPulse
from solprop.stepper.multistep import Multistep
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule

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
        if value is None or isinstance(value, int | float)
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


def _split(
    backend: str, integrator: RungeKutta | Multistep, *, drift: float = 0.65 * TRANGE
) -> Simulation:
    """Return the walker under the ``"split_step"`` scheme with its stop relaxed."""
    simulation = walker(
        drift, check=BoundaryCheck(budget=1.0, stop_factor=1.0), backend=backend
    )
    simulation.settings = dataclasses.replace(
        simulation.settings, absorber_scheme="split_step", integrator=integrator
    )
    return simulation


def _lost(run: Output) -> float:
    """Return the fraction of its launch energy a single-mode run lost."""
    energy = np.asarray(run["stats/energy"])[:, 0, 0]
    return float(1.0 - energy[-1] / energy[0])


@pytest.mark.parametrize("integrator", [RungeKutta(), Multistep()], ids=repr)
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_the_split_scheme_s_removal_is_read_exactly(
    backend: str, integrator: RungeKutta | Multistep
) -> None:
    # Applied once per accepted step, the split's removal is added up by the
    # backend's own finish at every accepted step, checked or not, the native
    # engine's included; it is the run's own energy deficit, not a quadrature of
    # it, though the pulse walks deep into the collar.
    if backend not in available():
        pytest.skip(f"{backend} is not installed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        run = _split(backend, integrator).run(progress="none")
    lost = _lost(run)
    assert lost > 0.5
    assert record_of(run)["removed_fraction"] == pytest.approx(lost, rel=1e-3)


def test_a_single_precision_split_reads_the_physical_removal() -> None:
    # A single-precision state is held as a / s, s a power of two chosen from the
    # launch (2**7 for ten kilowatts): the removal is of the physical state, not of
    # the scaled one, which read 1/s**2 of it. The nonlinearity scales down with the
    # power, so the walk is the watt's, inside binary32's range.
    peak_power = 1e4
    grid = EnvelopeGrid.from_limits(LENGTH, 1e-6, (0.8e-6, 1.25e-6), TRANGE)
    omega0 = 2 * np.pi * C / 1e-6
    offset = 0.65 * TRANGE / (BETA2 * LENGTH)
    simulation = Simulation.from_gnlse(
        1e-9 / peak_power, LENGTH, [0.0, 0.0, BETA2], lambda0=1e-6, grid=grid,
        pulses=[GaussianPulse(
            wavelength=2 * np.pi * C / (omega0 + offset), fwhm=200e-15,
            peak_power=peak_power,
        )],
        noise=None, raman=False, shock=False, n_save=3, precision="single",
        integrator=RungeKutta(), rtol=1e-5,
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / 200),
    )  # fmt: skip
    simulation.settings = dataclasses.replace(
        simulation.settings,
        boundary_check=BoundaryCheck(budget=1.0, stop_factor=1.0),
        absorber_scheme="split_step",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        run = simulation.run(progress="none")
    lost = _lost(run)
    assert lost > 0.5
    # Binary32's rounding over 200 steps, far inside the 1/s**2 the bug read.
    assert record_of(run)["removed_fraction"] == pytest.approx(lost, rel=1e-4)


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_a_deep_collar_reads_no_removal_from_residue(backend: str) -> None:
    # A capillary whose ten steps take the collar's deepest samples to alpha h of
    # about 25, its pulse far from the edges. The band mask after each split leaves
    # residue deep in the collar; read back from the committed state the split's
    # removal multiplied it by exp(alpha h) into more than the run lost and a
    # warning to widen the window.
    if backend not in available():
        pytest.skip(f"{backend} is not installed")
    simulation = Simulation.from_capillary(
        125e-6, 1e-3, "Ar", 1.0, lambda0=800e-9, wavelength_limits=(300e-9, 2e-6),
        trange=400e-15, fwhm=30e-15, energy=10e-6, mode_average=True, noise=None,
        n_save=2, backend=backend, integrator=RungeKutta(),
        schedule=FixedSchedule.uniform(0.0, 1e-3, 1e-4),
    )  # fmt: skip
    simulation.settings = dataclasses.replace(
        simulation.settings, absorber_scheme="split_step"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", BoundaryWarning)
        run = simulation.run(progress="none")
    energy = np.asarray(run["stats/energy"]).ravel()
    # The run loses about 1e-4 to the capillary's wall, none of it in the collar.
    assert 1.0 - energy[-1] / energy[0] > 1e-4
    assert record_of(run)["removed_fraction"] < 1e-10


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


class _Recording(MemoryOutput):
    """A memory output that keeps every execution record it is given."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[RunOutcome] = []

    def record_outcome(
        self, outcome: RunOutcome, checkpoint: TerminalCheckpoint | None = None
    ) -> None:
        self.records.append(outcome)
        super().record_outcome(outcome, checkpoint)


def _monitor_record(outcome: RunOutcome | None) -> dict[str, Any] | None:
    """Return an outcome's boundary-monitor record, or ``None`` without one."""
    assert outcome is not None and outcome.execution is not None
    return cast("dict[str, Any] | None", outcome.execution.get("boundary_monitor"))


def test_only_the_closing_record_carries_the_plane_rows() -> None:
    # The execution record is rewritten at every saved plane; with a row per plane in
    # it each record was longer than the last, and the file grew with their square.
    output = _Recording()
    relaxed = BoundaryCheck(budget=1.0, stop_factor=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        walker(0.65 * TRANGE, check=relaxed).run(progress="none", output=output)
    running = [
        record
        for outcome in output.records
        if outcome.status == "running"
        and (record := _monitor_record(outcome)) is not None
    ]
    assert running
    assert all("planes" not in record for record in running)
    assert all("removed_fraction" in record for record in running)
    closing = _monitor_record(output.outcome)
    assert closing is not None
    assert closing["planes"]["z"] == [0.0, 0.5, 1.0]


def test_a_stopped_run_s_record_carries_the_plane_rows() -> None:
    output = MemoryOutput()
    with (
        pytest.warns(BoundaryWarning, match="guard occupancy"),
        pytest.raises(BoundaryError, match="guards"),
    ):
        walker(1.2 * TRANGE).run(progress="none", output=output)
    assert output.outcome is not None and output.outcome.status == "failed"
    record = _monitor_record(output.outcome)
    assert record is not None
    assert record["planes"]["z"][0] == 0.0


def test_the_file_grows_linearly_with_its_planes(tmp_path: Path) -> None:
    # Eight times the planes: the bytes per plane do not grow with them.
    per_plane = []
    for n_save in (50, 400):
        path = tmp_path / f"run_{n_save}.h5"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BoundaryWarning)
            prop_gnlse(
                0.11, 0.05, [0.0, 0.0, -1.183e-26], lambda0=835e-9,
                wavelength_limits=(500e-9, 2e-6), trange=1e-12, width=100e-15,
                peak_power=9.8, pulseshape="sech", n_save=n_save, seed=1,
                progress="none", output=path, backend="numpy_ref",
            )  # fmt: skip
        per_plane.append(path.stat().st_size / n_save)
    assert per_plane[1] < 1.05 * per_plane[0]


class _CrashError(RuntimeError):
    """What an interrupted run raises, standing in for a machine going away."""


class _StopAfterFirstPlane(NullReporter):
    """A progress reporter that stops the run at its first saved plane after launch."""

    def plane(self, z: float, summary: object) -> None:
        raise _CrashError(f"stopped at z = {z}")


def _file_record(path: Path) -> tuple[dict[str, Any], int]:
    """Return a run file's monitor record and how many executions it holds."""
    outcome = resume(path).outcome
    record = _monitor_record(outcome)
    assert outcome is not None and record is not None
    return record, len(outcome.previous_executions) + 1


def test_a_resumed_run_continues_the_record(tmp_path: Path) -> None:
    # Interrupted at its middle plane and resumed, the run's record reads as the
    # uninterrupted run's: the launch it was judged at, the rows of every plane, and
    # the removed fraction against the original launch energy.
    relaxed = BoundaryCheck(budget=1.0, stop_factor=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        whole = record_of(walker(0.65 * TRANGE, check=relaxed).run(progress="none"))
        path = tmp_path / "run.h5"
        with pytest.raises(_CrashError):
            walker(0.65 * TRANGE, check=relaxed).run(
                path, progress=_StopAfterFirstPlane()
            )
        walker(0.65 * TRANGE, check=relaxed).run(path, resume=True, progress="none")
    record, executions = _file_record(path)
    assert executions == 2
    assert record["resumed_at"] == [0.5]
    assert record["planes"]["z"] == [0.0, 0.5, 1.0]
    assert record["launch_energy"] == pytest.approx(whole["launch_energy"], rel=1e-12)
    assert record["launch_collar"] == pytest.approx(whole["launch_collar"], rel=1e-9)
    assert record["removed_fraction"] > 1e-2
    assert record["removed_fraction"] == pytest.approx(
        whole["removed_fraction"], rel=1e-3
    )
    assert record["collar_max"] == pytest.approx(whole["collar_max"], rel=1e-3)


def test_a_run_stopped_at_the_edge_resumes_without_a_launch_check(
    tmp_path: Path,
) -> None:
    # A resume is not a launch: the state it resumes from is judged by the first
    # step's check, never refused as a clipped input.
    path = tmp_path / "run.h5"
    with (
        pytest.warns(BoundaryWarning, match="guard occupancy"),
        pytest.raises(BoundaryError, match="at z ="),
    ):
        walker(1.2 * TRANGE).run(path, progress="none")
    stopped, _ = _file_record(path)
    with (
        warnings.catch_warnings(record=True) as caught,
        pytest.raises(BoundaryError, match="at z ="),
    ):
        warnings.simplefilter("always", BoundaryWarning)
        walker(1.2 * TRANGE).run(path, resume=True, progress="none")
    assert not [one for one in caught if "at the launch" in str(one.message)]
    record, executions = _file_record(path)
    assert executions == 2
    assert len(record["resumed_at"]) == 1
    assert record["guard_high_max"] >= stopped["guard_high_max"]
    check = dataclasses.asdict(BoundaryCheck())
    assert {name: record[name] for name in check} == check


@pytest.mark.parametrize(
    "record", [None, {"removed_fraction": 0.5}], ids=["none", "no_launch_energy"]
)
def test_a_resume_with_nothing_to_continue_starts_the_record_again(
    record: dict[str, float] | None,
) -> None:
    # No earlier monitor, or a record without its launch energy, leaves nothing to
    # continue against: the record starts at the resumed state, which is not judged
    # as a launch though half its energy sits in a guard.
    grid, spectrum = _two_pulses(0.5, 0.95)
    monitor = BoundaryMonitor.build(grid, None, BoundaryCheck())
    monitor.resume(record, plane=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", BoundaryWarning)
        monitor.launch(monitor.energies(spectrum), 0.5)
    summary = monitor.summary()
    assert summary["resumed_at"] == [0.5]
    assert summary["launch_guard_high"] is None
    assert summary["removed_fraction"] == 0.0
    planes = summary["planes"]
    assert isinstance(planes, dict)
    assert planes["z"] == [0.5]


def test_a_running_record_restores_the_totals_without_rows() -> None:
    # A record written at a plane carries no rows: the totals continue against the
    # original launch, and the rows start again at the next plane.
    grid, spectrum = _two_pulses(0.0, 0.95)
    first = BoundaryMonitor.build(grid, None, BoundaryCheck())
    first.launch(first.energies(spectrum), 0.0)
    running = {
        name: value
        for name, value in first.summary().items()
        if name not in ("planes", "resumed_at")
    }
    running |= {"removed_fraction": 0.25, "steps": 7}
    monitor = BoundaryMonitor.build(grid, None, BoundaryCheck())
    monitor.resume(running, plane=False)
    monitor.launch(monitor.energies(spectrum), 0.5)
    summary = monitor.summary()
    assert summary["launch_energy"] == first.summary()["launch_energy"]
    assert summary["removed_fraction"] == 0.25
    assert summary["steps"] == 7
    assert summary["resumed_at"] == [0.5]
    planes = summary["planes"]
    assert isinstance(planes, dict)
    assert planes["z"] == []


def test_only_the_quiet_collar_is_part_of_the_resume_key() -> None:
    simulation = walker(0.0)
    plain = simulation.metadata().configuration_hash()
    simulation.settings = dataclasses.replace(
        simulation.settings, boundary_check=BoundaryCheck(budget=0.5, stop_factor=2.0)
    )
    assert simulation.metadata().configuration_hash() == plain
    simulation.settings = dataclasses.replace(
        simulation.settings, boundary_check=BoundaryCheck(quiet_collar=True)
    )
    assert simulation.metadata().configuration_hash() != plain


def test_without_a_check_nothing_is_watched() -> None:
    simulation = walker(1.2 * TRANGE)
    import dataclasses  # noqa: PLC0415

    simulation.settings = dataclasses.replace(simulation.settings, boundary_check=None)
    run = simulation.run(progress="none")
    assert isinstance(run, PlaneStore)
    assert run.outcome is not None and run.outcome.execution is not None
    assert "boundary_monitor" not in run.outcome.execution


def _two_pulses(share: float, at: float) -> tuple[EnvelopeGrid, np.ndarray]:
    """Return a grid and a centred pulse beside a copy at `at`, as a spectrum.

    The copy holds `share` of the energy; `at` is a fraction of ``t.max()``.
    """
    grid = EnvelopeGrid.from_limits(LENGTH, 1e-6, (0.8e-6, 1.25e-6), TRANGE)
    t = grid.t

    def pulse(centre: float) -> np.ndarray:
        return np.exp(-(((t - centre) / 10e-15) ** 2)).astype(complex)

    state = math.sqrt(1.0 - share) * pulse(0.0) + math.sqrt(share) * pulse(at * t.max())
    return grid, np.fft.fft(state)[None, None, None, :]


def _launch_with(share: float, at: float) -> BoundaryMonitor:
    """Return a monitor that has launched :func:`_two_pulses`."""
    grid, spectrum = _two_pulses(share, at)
    monitor = BoundaryMonitor.build(grid, None, BoundaryCheck())
    monitor.launch(monitor.energies(spectrum), 0.0)
    return monitor


def test_a_launch_in_the_guard_warns_as_a_step_would() -> None:
    # One rule from the launch on: the square root of 1e-5 is above the 1e-3 budget
    # and below ten of them.
    with pytest.warns(BoundaryWarning, match="at the launch"):
        monitor = _launch_with(1e-5, 0.95)
    launched = monitor.summary()["launch_guard_high"]
    assert launched == pytest.approx(1e-5, rel=1e-3)


def test_a_launch_deep_in_the_guard_is_refused() -> None:
    with pytest.raises(BoundaryError, match="at the launch the field puts"):
        _launch_with(0.5, 0.95)


def test_a_launch_in_the_collar_alone_is_not_judged() -> None:
    # The collar absorbs what the launch puts there, and the end of the run reports
    # it; only the guards are judged, as at every step.
    with warnings.catch_warnings():
        warnings.simplefilter("error", BoundaryWarning)
        monitor = _launch_with(1e-2, 0.85)
    summary = monitor.summary()
    launched_guard = summary["launch_guard_high"]
    assert isinstance(launched_guard, float)
    assert launched_guard < 1e-12
    launched_collar = summary["launch_collar"]
    assert isinstance(launched_collar, float)
    assert launched_collar > 1e-3


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
    monitor.launch(monitor.energies(np.fft.fft(pulse)[None, None, None, :]), 0.0)
    residue = np.full(grid.n_t, 1e-12, dtype=complex)
    monitor.step(monitor.energies(np.fft.fft(residue)[None, None, None, :]), 0.1)
    assert monitor.summary()["guard_low_max"] == 0.0


def test_a_launch_with_no_light_is_not_judged() -> None:
    grid = EnvelopeGrid.from_limits(LENGTH, 1e-6, (0.8e-6, 1.25e-6), TRANGE)
    monitor = BoundaryMonitor.build(grid, None, BoundaryCheck())
    dark = np.zeros((1, 1, 1, grid.n_t), dtype=complex)
    monitor.launch(monitor.energies(dark), 0.0)
    monitor.step(monitor.energies(dark), 0.1)
    summary = monitor.summary()
    assert summary["guard_low_max"] == summary["guard_high_max"] == 0.0
    assert summary["removed_fraction"] == 0.0


def _split_monitor(
    *, floor: bool = False
) -> tuple[BoundaryMonitor, np.ndarray, BoundaryEnergies]:
    """Return a split-scheme monitor launched on a centred pulse, and its reading."""
    grid = EnvelopeGrid.from_limits(LENGTH, 1e-6, (0.8e-6, 1.25e-6), TRANGE)
    rate = np.full(grid.n_t, 10.0)
    check = BoundaryCheck(budget=1.0, stop_factor=1.0)
    monitor = BoundaryMonitor.build(grid, rate, check, split=True)
    pulse = np.exp(-((grid.t / 50e-15) ** 2)).astype(complex)
    spectrum = np.fft.fft(pulse)[None, None, None, :]
    rng = np.random.default_rng(5)
    draw = 1e-3 * np.exp(2j * np.pi * rng.random(grid.n_t))[None, None, None, :]
    reading = monitor.energies(spectrum)
    monitor.launch(reading, 0.0, floor=draw if floor else None)
    return monitor, spectrum, reading


def test_the_split_s_removal_is_what_the_backend_took() -> None:
    # The backend adds up every accepted step's split, checked or not; the monitor
    # reads the sum at the next check and integrates no rate.
    monitor, spectrum, reading = _split_monitor()
    taken = monitor.split_removal(spectrum, 0.01)
    assert taken == pytest.approx(reading.total * -math.expm1(-10.0 * 0.01))
    monitor.skip(0.01)
    monitor.step(dataclasses.replace(reading, removed=2 * taken), 0.02)
    # Two checks at one position, as a step landing on a saved plane gives: the
    # second reads nothing taken since the first.
    monitor.step(reading, 0.02)
    summary = monitor.summary()
    assert summary["raw_removed_fraction"] == pytest.approx(2 * taken / reading.total)
    assert summary["removed_fraction"] == summary["raw_removed_fraction"]


def test_the_split_s_removal_counts_when_nothing_is_left() -> None:
    # The step that absorbs the last of the light is not judged, but what it took
    # is the run's removal all the same.
    monitor, _, reading = _split_monitor()
    gone = BoundaryEnergies(0.0, np.zeros(4), removed=reading.total)
    monitor.step(gone, 0.1)
    assert monitor.summary()["removed_fraction"] == pytest.approx(1.0)


def test_the_floor_s_share_of_the_split_s_removal_is_set_aside() -> None:
    # Over every committed step, checked or not, the split takes of the uniform
    # floor what it takes of a uniform density; that share is not the signal's.
    monitor, _, reading = _split_monitor(floor=True)
    floor = monitor.summary()["noise_floor"]
    assert isinstance(floor, float)
    assert floor > 0.0
    steps = (0.01, 0.025, 0.03)
    for z in steps[:-1]:
        monitor.skip(z)
    monitor.step(dataclasses.replace(reading, removed=0.5 * reading.total), steps[-1])
    # A uniform density's share under a uniform rate, over the three steps.
    share = 1.0 - math.exp(-10.0 * steps[-1])
    summary = monitor.summary()
    assert summary["raw_removed_fraction"] == pytest.approx(0.5)
    assert summary["removed_fraction"] == pytest.approx(0.5 - floor * share)


def test_a_plane_is_recorded_once() -> None:
    grid = EnvelopeGrid.from_limits(LENGTH, 1e-6, (0.8e-6, 1.25e-6), TRANGE)
    monitor = BoundaryMonitor.build(grid, None, BoundaryCheck())
    pulse = np.exp(-((grid.t / 50e-15) ** 2)).astype(complex)
    monitor.launch(monitor.energies(np.fft.fft(pulse)[None, None, None, :]), 0.0)
    monitor.plane(0.0)
    monitor.plane(0.0)
    planes = monitor.summary()["planes"]
    assert isinstance(planes, dict)
    assert planes["z"] == [0.0]


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
        monitor.launch(monitor.energies(draw), 0.0, floor=draw)
        monitor.step(monitor.energies(draw), 0.1)
    summary = monitor.summary()
    assert summary["guard_low_max"] == summary["guard_high_max"] == 0.0
    assert summary["noise_floor"] == pytest.approx(1.0)


def _with_check(simulation: Simulation, check: BoundaryCheck | None) -> Simulation:
    simulation.settings = dataclasses.replace(simulation.settings, boundary_check=check)
    return simulation


def _exit(run: Output) -> np.ndarray:
    return np.asarray(run.fields("fft")[-1])


QUIET_ENVELOPE = 1e-8
"""Field difference allowed between an opted-in quiet collar and the always-on one: a
decade above the largest measured (7e-10 for a clear pulse on the reference, 4e-9 for
one walking into the collar under Multistep), which is rounding and the different step
sequences of two adaptive runs; the omitted term acts on less than QUIET of the
energy, so it moves the field by less than its amplitude, 1e-6, in any case."""

INTEGRATORS = {"runge_kutta": None, "multistep": Multistep()}


def _integrated(simulation: Simulation, integrator: Multistep | None) -> Simulation:
    if integrator is not None:
        simulation.settings = dataclasses.replace(
            simulation.settings, integrator=integrator
        )
    return simulation


def test_the_collar_stays_in_unless_the_run_opts_in() -> None:
    # The default check watches the edges and leaves the stages alone: the run is the
    # unwatched run exactly, and no check finds the collar left out.
    clear = 3 * TRANGE
    watched = walker(0.0, trange=clear).run(progress="none")
    unwatched = _with_check(walker(0.0, trange=clear), None).run(progress="none")
    np.testing.assert_array_equal(_exit(watched), _exit(unwatched))
    assert record_of(watched)["collar_left_out_checks"] == 0


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_a_quiet_collar_is_left_out_without_changing_the_run(backend: str) -> None:
    # A pulse clear of the edges, opted in: the monitor finds the collar quiet, the
    # stages leave its term out, and the field is the always-on collar's.
    if backend not in available():
        pytest.skip(f"{backend} is not installed")
    opted = BoundaryCheck(quiet_collar=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        watched = walker(0.0, trange=3 * TRANGE, check=opted, backend=backend).run(
            progress="none"
        )
        always = _with_check(walker(0.0, trange=3 * TRANGE, backend=backend), None).run(
            progress="none"
        )
    a, b = _exit(watched), _exit(always)
    assert np.linalg.norm(a - b) <= QUIET_ENVELOPE * np.linalg.norm(b)
    record = record_of(watched)
    assert record["collar_left_out_checks"] > 0
    # Every committed step is either checked or counted as skipped (the planes'
    # checks add to the checked ones).
    accepted = int(np.asarray(watched["step_history/accepted"], dtype=bool).sum())
    assert record["steps"] + record["skipped"] >= accepted


@pytest.mark.parametrize("integrator", sorted(INTEGRATORS))
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_light_reaching_a_quiet_collar_switches_it_back_on(
    backend: str, integrator: str
) -> None:
    # A pulse that starts clear and walks well into the collar: the collar rests at
    # first and absorbs once light arrives, as the always-on collar does. While it
    # rests, no step outruns the monitor's check interval, so light cannot cross into
    # the collar unabsorbed within one long step; the Adams march, its order no longer
    # held by the collar's stability limit while it rests, takes the step after it
    # returns at that limit.
    if backend not in available():
        pytest.skip(f"{backend} is not installed")
    method = INTEGRATORS[integrator]
    relaxed = BoundaryCheck(budget=1.0, stop_factor=1.0, quiet_collar=True)
    drift, clear = 1.8 * TRANGE, 3 * TRANGE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        watched = _integrated(
            walker(drift, trange=clear, check=relaxed, backend=backend), method
        ).run(progress="none")
        always = _integrated(
            _with_check(walker(drift, trange=clear, backend=backend), None), method
        ).run(progress="none")
    energy = [np.asarray(run["stats/energy"])[-1, 0, 0] for run in (watched, always)]
    launched = np.asarray(always["stats/energy"])[0, 0, 0]
    assert energy[0] < 0.5 * launched
    assert energy[0] == pytest.approx(energy[1], rel=1e-6)
    a, b = _exit(watched), _exit(always)
    assert np.linalg.norm(a - b) <= QUIET_ENVELOPE * np.linalg.norm(b)
    record = record_of(watched)
    assert record["collar_left_out_checks"] > 0
    # Nothing is counted as removed while the collar rested: the record is the
    # energy the run lost, to the trapezoid's accuracy over the checked steps.
    lost = 1.0 - energy[0] / launched
    assert record["removed_fraction"] == pytest.approx(lost, rel=0.05)


def test_under_input_noise_the_floor_keeps_the_collar_in() -> None:
    # Quiet is judged on the whole field: the shot-noise floor alone fills the collar
    # far above QUIET, so an opted-in noisy run never leaves it out, and is the
    # unwatched run exactly (judged above the floor, the term was left out while light
    # a few times the floor sat in the collar).
    relaxed = BoundaryCheck(budget=1.0, stop_factor=1.0, quiet_collar=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        watched = walker(0.65 * TRANGE, check=relaxed, noise=True).run(progress="none")
        unwatched = _with_check(walker(0.65 * TRANGE, noise=True), None).run(
            progress="none"
        )
    np.testing.assert_array_equal(_exit(watched), _exit(unwatched))
    assert record_of(watched)["collar_left_out_checks"] == 0


def test_the_split_scheme_keeps_its_collar() -> None:
    # The split applies the collar once per step outside the stages: there is no
    # in-stage term to leave out, so an opted-in split run never rests it.
    simulation = walker(0.0, trange=3 * TRANGE, check=BoundaryCheck(quiet_collar=True))
    simulation.settings = dataclasses.replace(
        simulation.settings, absorber_scheme="split_step"
    )
    run = simulation.run(progress="none")
    assert record_of(run)["collar_left_out_checks"] == 0


def test_a_check_is_due_every_step_while_light_is_near_an_edge() -> None:
    grid = EnvelopeGrid.from_limits(LENGTH, 1e-6, (0.8e-6, 1.25e-6), TRANGE)
    monitor = BoundaryMonitor.build(grid, None, BoundaryCheck(quiet_collar=True))
    centred = np.exp(-((grid.t / 50e-15) ** 2)).astype(complex)
    edge = np.exp(-(((grid.t - 0.97 * grid.t.max()) / 30e-15) ** 2)).astype(complex)
    spectrum = lambda samples: monitor.energies(  # noqa: E731
        np.fft.fft(samples)[None, None, None, :]
    )
    monitor.launch(spectrum(centred), 0.0)
    # One quiet check is not enough to leave the collar out (QUIET_CHECKS).
    assert QUIET_CHECKS == 2
    assert not monitor.collar_quiet
    due = [monitor.due(1e-4 * k) or monitor.skip(1e-4 * k) for k in range(1, CADENCE)]
    assert not any(due)
    assert monitor.due(1e-4 * CADENCE)
    # However few the steps, a quiet window is checked every length / CHECKS.
    monitor.step(spectrum(centred), 1e-4 * CADENCE)
    assert monitor.collar_quiet
    assert not monitor.due(1e-4 * CADENCE + 0.5 * LENGTH / CHECKS)
    assert monitor.due(1e-4 * CADENCE + LENGTH / CHECKS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        with pytest.raises(BoundaryError):
            monitor.step(spectrum(centred + edge), 0.1)
    assert not monitor.collar_quiet
    assert monitor.due(0.1)


def test_a_collar_returns_at_once_and_rests_again_only_after_quiet_checks() -> None:
    grid = EnvelopeGrid.from_limits(LENGTH, 1e-6, (0.8e-6, 1.25e-6), TRANGE)
    check = BoundaryCheck(budget=1.0, stop_factor=1.0, quiet_collar=True)
    monitor = BoundaryMonitor.build(grid, None, check)
    centred = np.exp(-((grid.t / 50e-15) ** 2)).astype(complex)
    # A thousandth of the amplitude at the collar: far above QUIET, below the guards'
    # warning at this relaxed budget.
    near = 1e-3 * np.exp(-(((grid.t - 0.93 * grid.t.max()) / 30e-15) ** 2))
    spectrum = lambda samples: monitor.energies(  # noqa: E731
        np.fft.fft(samples)[None, None, None, :]
    )
    monitor.launch(spectrum(centred), 0.0)
    verdicts = []
    for k, state in enumerate((centred, centred + near, centred, centred), start=1):
        monitor.step(spectrum(state), 0.01 * k)
        verdicts.append(monitor.collar_quiet)
    assert verdicts == [True, False, False, True]


def test_a_resting_collar_adds_no_removal() -> None:
    # While the stages leave the term out the collar absorbs nothing: the steps over
    # which it rested add nothing to the removed fraction, however much light the
    # collar holds when the next check sees it.
    grid = EnvelopeGrid.from_limits(LENGTH, 1e-6, (0.8e-6, 1.25e-6), TRANGE)
    rate = np.full(grid.n_t, 10.0)
    check = BoundaryCheck(budget=1.0, stop_factor=1.0)
    monitor = BoundaryMonitor.build(grid, rate, check)
    pulse = np.fft.fft(np.exp(-((grid.t / 50e-15) ** 2)).astype(complex))
    state = pulse[None, None, None, :]
    monitor.launch(monitor.energies(state), 0.0)
    monitor.note_collar(True)
    monitor.step(monitor.energies(state), 0.1)
    summary = monitor.summary()
    assert summary["removed_fraction"] == summary["raw_removed_fraction"] == 0.0
    assert summary["collar_left_out_checks"] == 1
    monitor.note_collar(False)
    monitor.step(monitor.energies(state), 0.2)
    removed = monitor.summary()["removed_fraction"]
    assert isinstance(removed, float)
    assert removed > 0.0


@pytest.mark.parametrize("backend", ["numpy_ref", "native"])
def test_a_runner_s_next_input_starts_without_the_last_monitor(backend: str) -> None:
    # A runner reuses its backend: the next input, run with no check, carries no
    # record of the last one's monitor and keeps its collar in every stage, as a
    # fresh backend's run does.
    if backend not in available():
        pytest.skip(f"{backend} is not installed")
    simulation = walker(
        0.0, check=BoundaryCheck(quiet_collar=True), trange=3 * TRANGE, backend=backend
    )
    pulse = GaussianPulse(wavelength=1e-6, fwhm=200e-15, peak_power=1.0)
    with simulation.runner() as runner:
        watched = runner.run(pulse, progress="none")
        simulation.settings = dataclasses.replace(
            simulation.settings, boundary_check=None
        )
        unwatched = runner.run(pulse, progress="none")
    assert record_of(watched)["collar_left_out_checks"] > 0
    assert isinstance(unwatched, PlaneStore)
    assert unwatched.outcome is not None and unwatched.outcome.execution is not None
    assert "boundary_monitor" not in unwatched.outcome.execution
    fresh = walker(0.0, check=None, trange=3 * TRANGE, backend=backend)
    fresh.settings = dataclasses.replace(fresh.settings, boundary_check=None)
    alone = fresh.run(progress="none")
    np.testing.assert_array_equal(_exit(unwatched), _exit(alone))


def test_the_check_may_change_after_translation() -> None:
    simulation = walker(0.0)
    simulation.translate()
    import dataclasses  # noqa: PLC0415

    simulation.settings = dataclasses.replace(simulation.settings, boundary_check=None)
    assert simulation.settings.boundary_check is None


@pytest.mark.medium
def test_a_resting_collar_frees_the_adams_order() -> None:
    # The in-stage collar's stability limit holds the exponential Adams march of a
    # short capillary at order 1. While an opted-in collar rests that limit no longer
    # applies, so the march climbs to its top order; the two runs agree within their
    # shared rtol of 1e-6 (measured 3.3e-7).
    def capillary(check: BoundaryCheck) -> Output:
        simulation = Simulation.from_capillary(
            125e-6, 0.02, "He", 1.0, lambda0=800e-9, wavelength_limits=(150e-9, 2e-6),
            trange=200e-15, fwhm=10e-15, energy=100e-6, mode_average=True, n_save=2,
            noise=None,
        )  # fmt: skip
        return _with_check(simulation, check).run(progress="none")

    held = capillary(BoundaryCheck())
    freed = capillary(BoundaryCheck(quiet_collar=True))
    orders = {
        name: cast("dict[str, Any]", run.outcome.execution)["response_calls"]["orders"]
        for name, run in (("held", held), ("freed", freed))
        if isinstance(run, PlaneStore) and run.outcome is not None
    }
    assert max(orders["held"]) <= 2
    assert max(orders["freed"]) >= 4
    a, b = _exit(freed), _exit(held)
    assert np.linalg.norm(a - b) <= 1e-5 * np.linalg.norm(b)
