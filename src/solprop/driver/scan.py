# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Describe indexed parameter scans and dispatch independent simulations."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from solprop._interrupts import StopFlag
from solprop.driver._scan_model import Scan, ScanPoint
from solprop.driver._scan_run import execute_scan
from solprop.driver.executors import Executor
from solprop.driver.simulation import Simulation
from solprop.log import ProgressChoice, ScanReporter
from solprop.output.schema import RecipeRecord
from solprop.processing.scans import Reduction, ScanOutput

__all__ = ["Scan", "ScanPoint", "run_scan"]


def run_scan(  # noqa: PLR0913 - the public scan controls
    scan: Scan,
    build: Callable[[ScanPoint], Simulation],
    *,
    output: Path | str,
    executor: Executor | None = None,
    reductions: Sequence[Reduction] = (),
    fields: Literal["none", "planes", "all"] = "none",
    field_planes: Sequence[int] = (),
    stats: Literal["all", "none"] | Sequence[str] = "all",
    retry_failed: bool = False,
    stale_age: float = 600.0,
    progress: ProgressChoice | ScanReporter = "auto",
    stop: StopFlag | None = None,
    recipe: RecipeRecord | None = None,
) -> ScanOutput:
    """Run selected points and return a reader for their collected file.

    Parameters
    ----------
    scan : Scan
        Complete scan or selection of global indices.
    build : callable
        Build a Simulation from a ScanPoint. Noise the builder leaves unseeded is
        drawn from ``point.seed``, or, on a scan without seeds, from one derived
        from the scan's name and the point's global index; noise it seeds itself
        must use ``point.seed``. Pool builders must be pickleable module-level
        callables. Cache a base simulation lazily inside this callable for one
        translation per worker.
    output : pathlib.Path or str
        Collected HDF5 path. Matching completed points are skipped on rerun.
    executor : Executor or None, optional
        Execution policy. None uses the CLI policy, or Local outside the CLI.
    reductions : sequence of Reduction, optional
        Worker-side calculations with declared shape, dtype, units and provenance.
    fields : {'none', 'planes', 'all'}, optional
        Retain no fields, selected planes, or every saved plane.
    field_planes : sequence of int, optional
        Saved-plane indices for ``fields='planes'``; negative indices are allowed.
    stats : {'all', 'none'} or sequence of str, optional
        Retained statistics; all includes periodic rows. None is discouraged:
        statistics are the diagnostics needed to assess a scan.
    retry_failed : bool, optional
        Retry previously failed points; false leaves them recorded as failures.
    stale_age : float, optional
        Positive lease age in seconds before abandoned running points are recovered.
    progress : {'auto', 'bar', 'log', 'none'} or solprop.log.ScanReporter, optional
        One outer progress display; individual runs never draw their own bars. A
        :class:`~solprop.log.ScanReporter` replaces the display and is told when the
        scan and each point start and finish; a
        :class:`~solprop.log.WatchingScanReporter` also receives each point's run on
        ``Local`` and ``Pool`` (SEP 41, S4).
    stop : object or None, optional
        The scan's stop flag: any object with an ``is_set()`` method, such as a
        :class:`threading.Event` (SEP 41, S17). Once it is set no further point is
        dispatched; ``Local`` and ``Pool`` stop the points in flight within one step,
        a ``Pool`` through a message on each busy worker's pipe; ``Batch`` stops
        between groups. A stopped point is not committed, so a rerun computes it
        again, and the scan returns normally, its file holding the points finished
        before the stop; it logs at ``INFO`` that it stopped, and a scan reporter is
        told ``"stopped"`` for each stopped point. ``Queue`` and ``Slurm`` refuse it.
    recipe : solprop.output.schema.RecipeRecord or None, optional
        The recipe the scan was built from, which the scan file records once, so that
        it reopens with its setup (SEP 41, S2); :meth:`solprop.recipe.Recipe.scan`
        passes it among its keywords. ``None`` records none.

    Returns
    -------
    solprop.processing.scans.ScanOutput
        Reader with global indices, masked incomplete points and exact statistics.

    Raises
    ------
    DriverError
        If controls are invalid, the executor cannot run the builder, or `stop` is
        given to a ``Queue`` or ``Slurm`` executor.
    OutputError
        If the file belongs to a different scan or stored configuration.
    """
    return execute_scan(
        scan,
        build,
        output=output,
        executor=executor,
        reductions=reductions,
        fields=fields,
        field_planes=field_planes,
        stats=stats,
        retry_failed=retry_failed,
        stale_age=stale_age,
        progress=progress,
        stop=stop,
        recipe=recipe,
    )
