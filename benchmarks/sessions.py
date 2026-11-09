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
- ``scan_local``: four points derived from the case with ``with_input``, on ``Local``;
- ``group_batch``: the same four points on ``Batch(4)``, replaying ten 1.6 mm steps
  with the split-step collar, which a fixed step this long needs.

Import this module only after the worker has configured its scientific runtimes.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks.records import Record, SuiteError
from benchmarks.registry import CASES
from solprop.driver import Simulation
from solprop.driver.executors import Batch, Local
from solprop.driver.scan import Scan, run_scan
from solprop.grid import EnvelopeGrid
from solprop.log import NullReporter
from solprop.output import MemoryOutput
from solprop.pulses import SechPulse
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, Multistep, RungeKutta

if TYPE_CHECKING:
    from solprop.output import SnapshotController

__all__ = ["SCALES", "Requests", "fibre", "launch", "repetition"]

_BETA2, _GAMMA, _WIDTH = -11.83e-27, 0.11, 100e-15
"""gnlse_soliton's dispersion (s^2/m), nonlinearity (1/W/m) and sech width (s)."""
_PEAK = 3**2 * abs(_BETA2) / (_GAMMA * _WIDTH**2)
"""The N = 3 soliton's peak power, W."""
SCALES = (1.0, 0.9, 1.1, 1.2)
"""Peak-power scales of a workload's inputs; the first is the case's own launch."""


def launch(scale: float = 1.0) -> SechPulse:
    """Return the soliton's launch with its peak power scaled by `scale`."""
    return SechPulse(wavelength=835e-9, width=_WIDTH, peak_power=scale * _PEAK)


def fibre(
    name: str, stop: float, itinerary: list[float], options: Record
) -> Simulation:
    """Build a session case's fibre to `stop`, saving at `itinerary`.

    Parameters
    ----------
    name : str
        The case.
    stop : float
        Where the run ends, m: the case's length or a fraction of it.
    itinerary : list of float
        The save planes, m, the last at `stop`.
    options : dict
        The backend, precision and grid variant (``benchmarks.builders.variant``).

    Returns
    -------
    solprop.driver.Simulation
        The case, untranslated.
    """
    case = CASES[name]
    if case.builder != "session":
        raise SuiteError(f"session builder not implemented for {name}")
    limits, trange = (500e-9, 2e-6), 4e-12
    grid: dict[str, Any] = (
        {"wavelength_limits": limits, "trange": trange}
        if options["fft_size"] == "case"
        else {
            "grid": EnvelopeGrid.from_limits(
                case.length, 835e-9, limits, trange, fft_size="pow2"
            )
        }
    )
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
    return Simulation.from_gnlse(
        _GAMMA,
        case.length,
        [0.0, 0.0, _BETA2],
        wavelength=835e-9,
        width=_WIDTH,
        peak_power=_PEAK,
        shape="sech",
        raman=False,
        shock=False,
        noise=None,
        precision=options["precision"],
        backend=options["backend"],
        save_planes=itinerary,
        **grid,
        **extra,
    )


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
        Accepted and rejected steps where the outputs say (``None`` for a scan, whose
        file keeps no step history), the position reached and the runs made.
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
        runs = len(SCALES) - 1
        return {"accepted": accepted, "rejected": rejected, "z": z, "runs": runs}, final
    if workload not in ("scan", "group"):
        raise SuiteError(f"unknown workload {workload}")
    result = run_scan(
        Scan(f"sep47_{workload}", scale=list(SCALES)),
        lambda point: simulation.with_input(launch(float(point.values["scale"]))),  # pyright: ignore[reportArgumentType] - a scale variable
        output=scratch / f"{workload}.h5",
        executor=Batch(len(SCALES)) if workload == "group" else Local(),
        fields=fields,  # pyright: ignore[reportArgumentType] - a registry field retention
        progress="none",
    )
    status = result.status.tolist()
    if status != ["done"] * len(SCALES):
        raise SuiteError(f"the {workload}'s points ended {status}")
    z = float(simulation.settings.save_planes[-1])
    return {"accepted": None, "rejected": None, "z": z, "runs": len(SCALES)}, None
