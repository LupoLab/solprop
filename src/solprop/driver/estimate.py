# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a run will cost, before it runs (SEP 41, S11).

:func:`estimate` reports a translated simulation's grid, the persistent memory its
constructed backend holds, the size of its output, the cost of one right-hand-side
evaluation on the chosen backend timed after an untimed warm-up, and its integrator's
calls per step. It does not predict the number of adaptive steps, which depends on the
physics.
"""

import math
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from solprop.backends.protocol import Backend
from solprop.backends.registry import select
from solprop.backends.session import Start
from solprop.driver.run import field_retention
from solprop.exceptions import DriverError
from solprop.grid import FreeGrid, RadialGrid
from solprop.modes.slab import PlanarGrid
from solprop.stats import CATALOGUE, StatsSpec
from solprop.stepper import LowStorageRungeKutta, Multistep
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest

if TYPE_CHECKING:
    from solprop.driver.simulation import Simulation

__all__ = ["RunEstimate", "estimate", "field_bytes", "output_bytes", "plane_bytes"]

_REPEATS: Final = 3
"""Timed trial steps, after the warm-up; the median is reported."""

_COUNTED_PER_PLANE: Final = 3
"""Eight-byte entries a memory store keeps per plane beside the arrays: ``z`` and
the two per-plane counts."""


@dataclass(frozen=True)
class RunEstimate:
    """A run's cost, as far as it is known before the run.

    Attributes
    ----------
    shape : tuple of int
        The state's shape: modes or transverse columns, polarisations and
        frequencies.
    n_planes : int
        The planes the run saves, the entrance included.
    memory : int
        The persistent memory the constructed backend holds, its ledger's total, bytes.
    output_bytes : int
        The output's size for the requested planes and retained fields, bytes, as an
        in-memory store holds it (an HDF5 file adds little).
    backend : str
        The backend the run would use.
    threads : object
        The backend's thread count, as it reports it.
    setup : float
        Seconds to build the backend and take the untimed first step: compilation and
        transform planning.
    rhs : float
        Seconds per right-hand-side evaluation, the median of timed trial steps
        divided by the tableau's stages.
    calls_per_step : int
        Right-hand-side evaluations per accepted step of the integrator: six for the
        Runge-Kutta march (its first stage is the last's), one for the Adams method,
        four for the low-storage march.
    total_calls : int or None
        Evaluations of a replayed fixed schedule, its steps times the calls per step;
        ``None`` for an adaptive run, whose step count the physics decides.
    """

    shape: tuple[int, ...]
    n_planes: int
    memory: int
    output_bytes: int
    backend: str
    threads: object
    setup: float
    rhs: float
    calls_per_step: int
    total_calls: int | None

    @property
    def step(self) -> float:
        """Return the evaluations' share of an accepted step's time.

        Returns
        -------
        float
            Seconds per step.
        """
        return self.rhs * self.calls_per_step


def _timed_step(
    backend: Backend, problem: PropagationProblem, request: RunRequest
) -> tuple[Callable[[], object], int]:
    """Load a run's entrance and return the step to time, and its evaluations.

    The step is the backend's timing probe (SEP 47 section 6.6): one trial from the
    entrance that commits nothing, or for the low-storage march its own step, which
    advances the discarded registers in place.

    - Returns: a callable taking one step, and the right-hand-side evaluations in it.
    """
    integrator = request.integration.method
    evaluations = (
        _calls_per_step(integrator)
        if isinstance(integrator, LowStorageRungeKutta)
        else len(request.integration.tableau.nodes)
    )
    return backend.timing_probe(Start.entrance(problem)).trial, evaluations


def _calls_per_step(integrator: object) -> int:
    """Return an integrator's right-hand-side evaluations per accepted step."""
    if isinstance(integrator, Multistep):
        return 1
    if isinstance(integrator, LowStorageRungeKutta):
        return 4
    return 6


def estimate(
    simulation: "Simulation",
    *,
    repeats: int = _REPEATS,
) -> RunEstimate:
    """Estimate what a simulation's run would cost, without running it.

    Parameters
    ----------
    simulation : solprop.driver.simulation.Simulation
        The simulation; it is translated if it is not already.
    repeats : int, optional
        Timed trial steps after the warm-up.

    Returns
    -------
    RunEstimate
        The estimate.
    """
    problem = simulation.translate()
    request = simulation._run_request()
    settings = simulation.settings
    n_planes = request.observation.save_planes.size + 1
    size = output_bytes(simulation)
    integrator = request.integration.method
    calls = _calls_per_step(integrator)
    replay = request.replay
    total = None if replay is None else int(replay.schedule.steps.size) * calls
    started = time.perf_counter()
    backend = select(
        settings.backend,
        problem,
        request,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
        options=settings.execution_options(),
    )
    try:
        memory = sum(backend.memory_ledger().values())
        # Bound only when a monitor attaches, which an estimate doesn't do (SEP 44).
        if (
            settings.boundary_check is not None
            and simulation.medium is not None
            and isinstance(simulation.spatial_grid, RadialGrid | FreeGrid | PlanarGrid)
        ):
            memory += backend.edge_memory(simulation.spatial_grid)
        step, evaluations = _timed_step(backend, problem, request)
        step()
        setup = time.perf_counter() - started
        timings = []
        for _ in range(max(1, repeats)):
            begun = time.perf_counter()
            step()
            timings.append(time.perf_counter() - begun)
        rhs = statistics.median(timings) / evaluations
        reported = backend.metadata()
        # Each backend names its own: native "thread_count"; numpy_ref runs on one
        # thread and names none.
        threads = next(
            (
                reported[key]
                for key in ("thread_count", "threads")
                if reported.get(key) is not None
            ),
            1,
        )
    finally:
        backend.close()
    return RunEstimate(
        tuple(problem.shape),
        n_planes,
        memory,
        size,
        settings.backend,
        threads,
        setup,
        rhs,
        calls,
        total,
    )


def output_bytes(simulation: "Simulation") -> int:
    """Return the size of a run's output, without constructing a backend.

    Every saved plane's :func:`plane_bytes`, less the field of each plane the settings
    do not retain, as an in-memory store holds it (an HDF5 file adds little).

    Parameters
    ----------
    simulation : solprop.driver.simulation.Simulation
        The simulation; it is translated if it is not already.

    Returns
    -------
    int
        Bytes.
    """
    problem = simulation.translate()
    request = simulation._run_request()
    settings = simulation.settings
    retention = field_retention(request, settings.fields, settings.field_planes)
    n_planes = request.observation.save_planes.size + 1
    kept = sum(1 for index in range(n_planes) if retention.keeps(index))
    plane = plane_bytes(problem, request.observation.stats)
    return n_planes * plane - (n_planes - kept) * field_bytes(problem)


def plane_bytes(problem: PropagationProblem, stats: StatsSpec) -> int:
    """Return the bytes one saved plane occupies in an in-memory store.

    The field (complex128), the noise phase (float64) when the run carries a noise
    model, every statistic at its catalogued shape (float64; a statistic catalogued
    with a smaller shape for spatial beams is counted at the modal one, an upper bound
    of a few entries), and the position and the two per-plane counts.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The translated problem.
    stats : solprop.stats.StatsSpec
        The statistics its run collects.

    Returns
    -------
    int
        Bytes per plane.

    Examples
    --------
    >>> from solprop.driver import Simulation
    >>> from solprop.driver.estimate import plane_bytes
    >>> simulation = Simulation.from_gnlse(
    ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
    ...     width=100e-15, peak_power=9.8, shape="sech",
    ... )
    >>> problem = simulation.translate()
    >>> stats = simulation._run_request().observation.stats
    >>> plane_bytes(problem, stats) >= 16 * problem.shape[-1]
    True
    """
    n_modes, n_pol, n_omega = problem.shape
    entries = n_modes * n_pol * n_omega
    total = 16 * entries + 8 * _COUNTED_PER_PLANE
    if problem.initial.noise_model != "none":
        total += 8 * entries
    sizes = {
        "n_modes": n_modes,
        "n_pol": n_pol,
        "n_omega": n_omega,
        "n_windows": stats.n_windows,
        "n_apertures": stats.n_apertures,
        "n_regions": stats.n_regions,
        "n_species": problem.responses.n_species,
        "n_nodes": problem.transform.n_nodes,
    }
    for name in stats.names:
        total += 8 * _catalogued_entries(CATALOGUE[name].shape, sizes)
    return total


def field_bytes(problem: PropagationProblem) -> int:
    """Return the bytes of one plane's field, and its noise phase, in a memory store.

    The part of :func:`plane_bytes` a plane that keeps no field does not hold (SEP 22).

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The translated problem.

    Returns
    -------
    int
        Sixteen bytes per state entry, and eight more with a noise model.

    Examples
    --------
    >>> from solprop.driver import Simulation
    >>> from solprop.driver.estimate import field_bytes
    >>> simulation = Simulation.from_gnlse(
    ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
    ...     width=100e-15, peak_power=9.8, shape="sech", noise=None,
    ... )
    >>> field_bytes(simulation.translate()) == 16 * simulation.translate().shape[-1]
    True
    """
    entries = math.prod(problem.shape)
    return (16 + (8 if problem.initial.noise_model != "none" else 0)) * entries


def _catalogued_entries(shape: str, sizes: Mapping[str, int]) -> int:
    """Return how many entries a statistic of a catalogued shape holds.

    - ``shape``: the catalogue's shape, such as ``'(n_windows, n_modes)'``, ``'(2,)'``
      or ``'()'``; an alternative after ``';'`` is ignored.
    - ``sizes``: the length of every axis name a catalogued shape uses.
    - Returns: the product of the axis lengths, one for a scalar.
    """
    axes = shape.partition(";")[0].strip().strip("()")
    count = 1
    for axis in (one.strip() for one in axes.split(",")):
        if not axis:
            continue
        if axis.isdigit():
            count *= int(axis)
        elif axis in sizes:
            count *= sizes[axis]
        else:
            msg = f"no length is known for the statistics axis {axis!r} in {shape!r}"
            raise DriverError(msg)
    return count
