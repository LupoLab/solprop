# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a run will cost, before it runs (SEP 41, S11).

:func:`estimate` reports a translated simulation's grid, its persistent memory
admitted against its budget **before** anything is allocated, the size of its output,
the cost of one right-hand-side evaluation on the chosen backend timed after an untimed
warm-up, and its integrator's calls per step. It does not predict the number of
adaptive steps, which depends on the physics.
"""

import statistics
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from solprop.backends.registry import select
from solprop.driver.memory import admit_memory, field_bytes, plane_bytes
from solprop.driver.run import field_retention
from solprop.exceptions import MemoryBudgetError
from solprop.stepper import LowStorageRungeKutta, Multistep
from solprop.translate.problem import KernelProblem

if TYPE_CHECKING:
    from solprop.driver.simulation import Simulation

__all__ = ["RunEstimate", "estimate"]

_REPEATS: Final = 3
"""Timed trial steps, after the warm-up; the median is reported."""


@dataclass(frozen=True)
class RunEstimate:
    """A run's cost, as far as it is known before the run.

    Attributes
    ----------
    shape : tuple of int
        The state's shape: batch, modes or transverse columns, polarisations and
        frequencies.
    n_planes : int
        The planes the run saves, the entrance included.
    memory : int
        The persistent memory the run would hold, bytes.
    admitted : bool
        Whether it is admitted against the memory budget.
    refusal : str or None
        Why it is not, when it is not; nothing was allocated.
    output_bytes : int
        The output's size for the requested planes and retained fields, bytes, as an
        in-memory store holds it (an HDF5 file adds little).
    backend : str
        The backend the run would use.
    threads : object
        The backend's thread count, as it reports it, or ``None`` when the run is
        not admitted and no backend was built.
    setup : float or None
        Seconds to build the backend and take the untimed first step: compilation and
        transform planning; ``None`` when the run is not admitted.
    rhs : float or None
        Seconds per right-hand-side evaluation, the median of timed trial steps
        divided by the tableau's stages; ``None`` when the run is not admitted.
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
    admitted: bool
    refusal: str | None
    output_bytes: int
    backend: str
    threads: object
    setup: float | None
    rhs: float | None
    calls_per_step: int
    total_calls: int | None

    @property
    def step(self) -> float | None:
        """Seconds per accepted step, the evaluations' share of it; ``None`` untimed."""
        return None if self.rhs is None else self.rhs * self.calls_per_step


def _calls_per_step(integrator: object) -> int:
    """Return an integrator's right-hand-side evaluations per accepted step.

    Parameters
    ----------
    integrator : object
        The settings' integrator.

    Returns
    -------
    int
        One for the Adams method, four for the low-storage march, six for DOPRI5.
    """
    if isinstance(integrator, Multistep):
        return 1
    if isinstance(integrator, LowStorageRungeKutta):
        return 4
    return 6


def estimate(
    simulation: "Simulation",  # noqa: UP037 - checked at run time
    *,
    memory_budget: int | str | None = None,
    repeats: int = _REPEATS,
) -> RunEstimate:
    """Estimate what a simulation's run would cost, without running it.

    Parameters
    ----------
    simulation : solprop.driver.simulation.Simulation
        The simulation; it is translated if it is not already.
    memory_budget : int or str or None, optional
        The budget its memory is admitted against, as
        :meth:`~solprop.driver.simulation.Simulation.run` takes it.
    repeats : int, optional
        Timed trial steps after the warm-up.

    Returns
    -------
    RunEstimate
        The estimate.
    """
    problem = simulation.translate()
    settings = simulation.settings
    retention = field_retention(problem, settings.fields, settings.field_planes)
    n_planes = len(problem.stepper.save_planes) + 1
    kept = sum(1 for index in range(n_planes) if retention.keeps(index))
    output_bytes = n_planes * plane_bytes(problem) - (n_planes - kept) * field_bytes(
        problem
    )
    integrator = problem.stepper.integrator
    calls = _calls_per_step(integrator)
    schedule = problem.stepper.schedule
    total = None if schedule is None else int(schedule.steps.size) * calls
    try:
        admission = admit_memory(
            problem, settings.backend, memory_budget, retention=retention
        )
    except MemoryBudgetError as error:
        return RunEstimate(
            tuple(problem.shape),
            n_planes,
            _requested(problem, settings.backend),
            False,
            str(error),
            output_bytes,
            settings.backend,
            None,
            None,
            None,
            calls,
            total,
        )
    started = time.perf_counter()
    backend = select(
        settings.backend,
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    try:
        backend.set_initial(problem.initial.field0, None, z=problem.initial.z)
        z0 = float(problem.initial.z)
        dz = float(problem.stepper.dz_init)
        trial = getattr(backend, "_trial_step")  # noqa: B009 - every StepDriver has it
        trial(z0, dz)
        setup = time.perf_counter() - started
        timings = []
        for _ in range(max(1, repeats)):
            begun = time.perf_counter()
            trial(z0, dz)
            timings.append(time.perf_counter() - begun)
        stages = len(problem.stepper.tableau.nodes)
        rhs = statistics.median(timings) / stages
        reported = backend.metadata()
        # Each backend names its own: native "thread_count", numba_cpu
        # "numba_threads"; numpy_ref runs on one thread and names none.
        threads = next(
            (
                reported[key]
                for key in ("thread_count", "numba_threads", "threads")
                if reported.get(key) is not None
            ),
            1,
        )
    finally:
        backend.close()
    return RunEstimate(
        tuple(problem.shape),
        n_planes,
        admission.estimate.total,
        True,
        None,
        output_bytes,
        settings.backend,
        threads,
        setup,
        rhs,
        calls,
        total,
    )


def _requested(problem: KernelProblem, backend: str) -> int:
    """Return the persistent memory a refused run asked for.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    backend : str
        The backend.

    Returns
    -------
    int
        The estimate's total, bytes.
    """
    admission = admit_memory(problem, backend, "none")
    return admission.estimate.total
