# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build SEP 47's fibre windows and run its repeated, scanned and grouped workloads.

The ``sep47`` set's cases on one fibre (SEP 47 section 7.2): the N = 3 soliton of
``gnlse_soliton`` (its dispersion, nonlinearity, 100 fs sech launch and time grid,
noiseless) over its first 16 mm, about a hundredth of a soliton period, where the
default ``RungeKutta()`` takes about forty steps. Each window holds the events it
measures:

- ``stats_period1_adams`` and ``stats_period1_rk``: an energy and peak-power row at
  every accepted step (``StatsSpec(period=1)``), under ``Multistep()`` and
  ``RungeKutta()``;
- ``snapshots_idle``, ``snapshots_sparse`` and ``snapshots_every``: a run to HDF5
  with a ``SnapshotController``, never asked, asked after every eighth accepted step,
  and asked at every accepted endpoint. Today a controller needs ``step_planes=1``
  and HDF5 output, so every case saves a plane at every step; a variant without the
  step planes becomes possible once SEP 47 schedules snapshots independently;
- ``with_input_repeat``: three runs of other inputs through ``Simulation.with_input``;
- ``scan_local``: four points derived from the case with ``with_input``, on ``Local``,
  each run watched to count its steps;
- ``group_batch``: the same four points on ``Batch(4)``, replaying ten 1.6 mm steps
  with the split-step collar, which a fixed step this long needs. ``Batch`` watches no
  point's run, so its steps are left uncounted except by the structural counters.

Import this module only after the worker has configured its scientific runtimes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks.builders import SOLITON_PEAK, SOLITON_WIDTH, soliton
from benchmarks.records import Record, SuiteError
from benchmarks.registry import CASES
from solprop.driver import Simulation
from solprop.driver.executors import Batch, Local
from solprop.driver.scan import Scan, run_scan
from solprop.log import NullReporter, NullScanReporter, ProgressReporter
from solprop.output import MemoryOutput
from solprop.output.scan import read_point
from solprop.pulses import SechPulse
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, Multistep, RungeKutta

if TYPE_CHECKING:
    from solprop.output import SnapshotController

__all__ = ["SCALES", "Requests", "fibre", "launch", "repetition"]

SCALES = (1.0, 0.9, 1.1, 1.2)
"""Peak-power scales of a workload's inputs; the first is the case's own launch."""


def launch(scale: float = 1.0) -> SechPulse:
    """Return the soliton's launch with its peak power scaled by `scale`."""
    return SechPulse(
        wavelength=835e-9, width=SOLITON_WIDTH, peak_power=scale * SOLITON_PEAK
    )


def fibre(
    name: str, stop: float, itinerary: list[float], options: Record
) -> Simulation:
    """Build a session case's fibre to `stop`, saving at `itinerary`.

    ``gnlse_soliton``'s fibre, launch and grid (``benchmarks.builders.soliton``) over
    the case's length, with the case's statistics, step planes or schedule.

    Parameters
    ----------
    name : str
        The case.
    stop : float
        Where the run ends, m: the case's length or a fraction of it.
    itinerary : list of float
        The save planes, m, the last at `stop`.
    options : dict
        The backend, noise and grid variant (``benchmarks.builders.variant``).

    Returns
    -------
    solprop.driver.Simulation
        The case, untranslated.
    """
    case = CASES[name]
    if case.builder != "session":
        raise SuiteError(f"session builder not implemented for {name}")
    extra: dict[str, Any] = {}
    if name.startswith("stats_period1"):
        extra["stats"] = StatsSpec(
            names=("energy", "peak_power"),
            periodic=("energy", "peak_power"),
            period=1,
        )
        extra["integrator"] = Multistep() if name.endswith("adams") else RungeKutta()
    if case.snapshots is not None:
        extra["step_planes"] = 1
    if case.step is not None:
        extra["schedule"] = FixedSchedule.uniform(
            0.0, stop, case.step, save_planes=itinerary
        )
        # The in-stage collar is stable only up to 0.42 mm steps on this window.
        extra["absorber_scheme"] = "split_step"
    return soliton(case.length, options, save_planes=itinerary, **extra)


class _Steps(NullReporter):
    """Count one run's accepted and rejected attempts."""

    def __init__(self) -> None:
        self.accepted = self.rejected = 0

    def step(self, z: float, dz: float, accepted: bool) -> None:
        """Count an attempt."""
        del z, dz
        if accepted:
            self.accepted += 1
        else:
            self.rejected += 1


class _Watch(NullScanReporter):
    """Watch each point a ``Local`` scan runs, counting its attempts."""

    def __init__(self) -> None:
        self.runs: list[_Steps] = []

    def point_reporter(self, index: int) -> ProgressReporter | None:
        """Return a fresh step counter for the point's run."""
        del index
        self.runs.append(_Steps())
        return self.runs[-1]


class Requests(NullReporter):
    """Ask a snapshot controller for a snapshot after every `every`-th accepted step.

    A reporter is told of every attempt (``ProgressReporter.step``) and changes
    nothing in the run; ``every = 0`` never asks.
    """

    def __init__(self, controller: SnapshotController, every: int) -> None:
        self._controller = controller
        self._every = every
        self._accepted = 0

    def step(self, z: float, dz: float, accepted: bool) -> None:
        """Count an accepted step and ask for a snapshot when it is due."""
        del z, dz
        if accepted:
            self._accepted += 1
            if self._every and self._accepted % self._every == 0:
                self._controller.request()


def _finite_statistics(output: Mapping[str, Any]) -> bool | None:
    """Whether a run's last plane's statistics are finite; ``None`` with none."""
    values = [
        np.asarray(value)[-1]
        for key, value in output.items()
        if key.startswith("stats/") and np.asarray(value).size
    ]
    if not values:
        return None
    return all(bool(np.all(np.isfinite(value))) for value in values)


def repetition(
    workload: str, simulation: Simulation, scratch: Path, fields: str
) -> tuple[Record, np.ndarray | None]:
    """Run one repetition of a workload other than a single run.

    Parameters
    ----------
    workload : str
        ``"with_input"``, ``"scan"`` or ``"group"`` (``benchmarks.registry.Case``).
    simulation : solprop.driver.Simulation
        The case, as built.
    scratch : pathlib.Path
        An empty directory for the scan's file.
    fields : str
        The scan's field retention.

    Returns
    -------
    dict
        Accepted and rejected steps summed over the runs, where they are watched
        (``None`` for a group, whose ``Batch`` watches no point's run); whether the
        last plane of every run is finite (its field, or for a field-free scan its
        statistics); the position reached and the runs made.
    numpy.ndarray or None
        The last run's final field, or ``None`` for a scan.
    """
    if workload == "with_input":
        accepted = rejected = 0
        final = None
        z = math.nan
        for scale in SCALES[1:]:
            output = simulation.with_input(launch(scale)).run(progress="none")
            assert isinstance(output, MemoryOutput)
            history = np.asarray(output["step_history/accepted"], dtype=bool)
            accepted += int(history.sum())
            rejected += int((~history).sum())
            z = float(output["z"][-1])
            final = output.plane_field(output.n_saved - 1, "fft")
        record: Record = {
            "accepted": accepted,
            "rejected": rejected,
            "z": z,
            "runs": len(SCALES) - 1,
            "finite": final is not None and bool(np.all(np.isfinite(final))),
        }
        return record, final
    if workload not in ("scan", "group"):
        raise SuiteError(f"unknown workload {workload}")
    watch = _Watch()
    path = scratch / f"{workload}.h5"
    result = run_scan(
        Scan(f"sep47_{workload}", scale=list(SCALES)),
        lambda point: simulation.with_input(launch(float(point.values["scale"]))),  # pyright: ignore[reportArgumentType] - a scale variable
        output=path,
        executor=Batch(len(SCALES)) if workload == "group" else Local(),
        fields=fields,  # pyright: ignore[reportArgumentType] - a registry field retention
        progress=watch,
    )
    status = result.status.tolist()
    if status != ["done"] * len(SCALES):
        raise SuiteError(f"the {workload}'s points ended {status}")
    finite = [
        _finite_statistics(read_point(path, index)) for index in range(len(SCALES))
    ]
    watched = len(watch.runs) == len(SCALES)
    record = {
        "accepted": sum(run.accepted for run in watch.runs) if watched else None,
        "rejected": sum(run.rejected for run in watch.runs) if watched else None,
        "z": float(simulation.settings.save_planes[-1]),
        "runs": len(SCALES),
        "finite": None if None in finite else all(finite),
    }
    return record, None
