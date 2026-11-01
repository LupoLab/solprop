# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Admit a run against a declared memory budget before its backend allocates.

The estimate of :func:`solprop.backends.memory.estimate_memory` is compared with a
budget **before** :func:`solprop.backends.registry.select` constructs the backend. Over
budget, the run is refused by name -- the estimate, the budget, the largest groups and
what to change -- or warned on, when the declaration asks for a warning. Nothing is
approximated to fit: the remedy for a run that does not fit is more memory, never an
approximated stored field.

**The declaration.** In order of precedence:

1. the ``memory_budget`` keyword of :meth:`solprop.driver.simulation.Simulation.run`,
   for one run in a script or a notebook;
2. the environment variable :data:`MEMORY_BUDGET_VARIABLE`
   (``SOLPROP_MEMORY_BUDGET``), which reaches every entry point, the keyword
   interfaces and the scan workers, without a change of code: the natural place for
   a job script's allocation;
3. otherwise the **machine's memory with 20 % headroom**: four fifths of the smallest
   of the physical memory, a cgroup limit and a Slurm allocation, the same limit and
   headroom the scan executors admit their workers against -- **warned on, not
   refused**, so that a user who wants to try a run that may not fit can.

A declaration is a byte count, as an integer or a string with an optional unit
(``"16 GiB"``, ``"12GB"``, ``"8e9"``), optionally prefixed ``"warn:"`` to warn instead
of refusing; ``"none"`` declares no check. A declared budget refuses; the undeclared
default only warns. The default checks against the machine because a persistent
footprint beyond it pages or is killed after the run has started,
possibly hours in; physical memory rather than free memory keeps the decision the same
on a machine whatever else is running. The headroom leaves room for what the estimate
does not count (the planning heap, per-stage transients) and for the rest of the
process.

**The output.** A run kept in memory holds every plane it saves, so :func:`admit_memory`
adds the store at its planned capacity -- the field, the noise phase and the statistics
of the entrance and every save plane -- as the group ``output``, when it is given the
run's handler (:meth:`solprop.output.schema.PlaneStore.held_bytes`). A run that keeps
its field on some planes only (SEP 22) is charged the statistics of every plane and the
field of those alone, so a statistics-only run is charged its statistics. A file output
writes each plane out and holds none of them, so it adds nothing. The step history and
the between-planes rows grow with the steps taken and are not counted.
"""

import logging
import math
import os
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

from solprop.backends.memory import (
    MemoryEstimate,
    estimate_guide_memory,
    estimate_memory,
    low_storage_memory,
)
from solprop.backends.registry import available_device_memory
from solprop.exceptions import DriverError, MemoryBudgetError, MemoryBudgetWarning
from solprop.grid import TimeGrid
from solprop.modes.guide import ModalGuide
from solprop.modes.material import JsonValue
from solprop.output.schema import FieldRetention, Output, PlaneStore
from solprop.stats import CATALOGUE
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.translate.problem import KernelProblem

__all__ = [
    "DEFAULT_HEADROOM",
    "MEMORY_BUDGET_VARIABLE",
    "MemoryAdmission",
    "MemoryLimit",
    "admit_guide_memory",
    "admit_memory",
    "field_bytes",
    "format_bytes",
    "parse_memory_budget",
    "plane_bytes",
    "resolve_memory_budget",
]

logger = logging.getLogger(__name__)

MEMORY_BUDGET_VARIABLE = "SOLPROP_MEMORY_BUDGET"
"""The environment variable that declares a memory budget for every run."""

DEFAULT_HEADROOM = 0.2
"""The fraction of the machine's memory the default budget leaves unclaimed."""

_UNITS = {
    "": 1,
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "tb": 10**12,
    "kib": 2**10,
    "mib": 2**20,
    "gib": 2**30,
    "tib": 2**40,
}
_BINARY = ("KiB", "MiB", "GiB", "TiB")
_BUDGET = re.compile(r"^\s*([0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)\s*([a-z]*)\s*$")

_ADVICE = {
    "general_reference": "fewer channels, a coarser node rule or fewer frequency bins "
    "(a narrower window or wavelength range); the reference is held at every bin",
    "general_maps": "fewer channels, a coarser node rule or fewer frequency bins",
    "general_chunk": "fewer time samples (the node chunk is already bounded)",
    "general_delayed": "fewer time samples (the delayed response's histories scale "
    "with them)",
    "coupled_flow": "fewer channels or active bins, or fewer sections or Chebyshev "
    "terms along z",
    "base": "fewer time samples, modes or nodes, or propagator_mode='recompute'",
    "adams_history": "integrator=RungeKutta(), which keeps no Adams history",
    "output": "an HDF5 file as the output instead of memory, or fewer save planes",
}

_COUNTED_PER_PLANE = 3
"""Eight-byte entries a memory store keeps per plane beside the arrays: ``z`` and
the two per-plane counts."""


@dataclass(frozen=True)
class MemoryLimit:
    """A declared memory budget and what exceeding it does.

    Attributes
    ----------
    limit : int or None
        The budget, bytes, positive; ``None`` declares no check.
    action : {'refuse', 'warn'}
        Whether a run over the budget is refused or warned on.
    source : str
        Where the declaration came from: ``"keyword"``, :data:`MEMORY_BUDGET_VARIABLE`
        or ``"machine"``.

    Examples
    --------
    >>> from solprop.driver.memory import MemoryLimit
    >>> MemoryLimit(2**30, "warn", "keyword").limit
    1073741824
    """

    limit: int | None
    action: Literal["refuse", "warn"] = "refuse"
    source: str = "keyword"

    def __post_init__(self) -> None:
        """Refuse a budget that is not a positive byte count."""
        if self.limit is not None and self.limit <= 0:
            msg = f"a memory budget must be a positive byte count, got {self.limit}"
            raise DriverError(msg)


@dataclass(frozen=True)
class MemoryAdmission:
    """A run's estimate and the budget it was admitted against.

    Attributes
    ----------
    estimate : solprop.backends.memory.MemoryEstimate
        The persistent bytes the backend will allocate, by group.
    budget : MemoryLimit
        The budget the estimate was checked against.
    """

    estimate: MemoryEstimate
    budget: MemoryLimit

    def record(self) -> dict[str, JsonValue]:
        """Return the admission as the run's backend record carries it.

        Returns
        -------
        dict
            ``estimate`` (bytes), the non-empty ``groups`` (bytes), ``budget`` (bytes,
            or ``None``) and ``budget_source``.
        """
        return {
            "estimate": self.estimate.total,
            "groups": {
                name: size for name, size in self.estimate.groups.items() if size
            },
            "budget": self.budget.limit,
            "budget_source": self.budget.source,
        }


def parse_memory_budget(value: int | str, *, source: str = "keyword") -> MemoryLimit:
    """Read a memory budget declaration.

    Parameters
    ----------
    value : int or str
        A positive byte count; or a string: a number with an optional unit (``B``,
        ``kB``, ``MB``, ``GB``, ``TB`` in powers of 1000, ``KiB``, ``MiB``, ``GiB``,
        ``TiB`` in powers of 1024; case-insensitive), optionally prefixed ``warn:``;
        or ``none``, which declares no check.
    source : str, optional
        Where the declaration came from, for the record and the messages.

    Returns
    -------
    MemoryLimit
        The declared budget.

    Raises
    ------
    DriverError
        If the declaration cannot be read or is not positive.

    Examples
    --------
    >>> from solprop.driver.memory import parse_memory_budget
    >>> parse_memory_budget("warn: 16 GiB")
    MemoryLimit(limit=17179869184, action='warn', source='keyword')
    >>> parse_memory_budget("none").limit is None
    True
    """
    if isinstance(value, int):
        return MemoryLimit(value, "refuse", source)
    text = value.strip().lower()
    action: Literal["refuse", "warn"] = "refuse"
    if text.startswith("warn:"):
        action, text = "warn", text.removeprefix("warn:")
    if text.strip() == "none":
        return MemoryLimit(None, action, source)
    match = _BUDGET.match(text)
    if match is None or match.group(2) not in _UNITS:
        msg = (
            f"cannot read the memory budget {value!r} ({source}): give a byte count "
            "with an optional unit, such as '16 GiB' or '12e9', optionally prefixed "
            "'warn:', or 'none'"
        )
        raise DriverError(msg)
    return MemoryLimit(
        math.floor(float(match.group(1)) * _UNITS[match.group(2)]), action, source
    )


def resolve_memory_budget(memory_budget: int | str | None = None) -> MemoryLimit:
    """Return the budget a run is admitted against.

    Parameters
    ----------
    memory_budget : int or str or None, optional
        A declaration for this run (:func:`parse_memory_budget`); ``None`` takes the
        environment variable :data:`MEMORY_BUDGET_VARIABLE`, and without it the
        machine's memory less :data:`DEFAULT_HEADROOM`.

    Returns
    -------
    MemoryLimit
        The budget and where it came from.

    Examples
    --------
    >>> from solprop.driver.memory import resolve_memory_budget
    >>> resolve_memory_budget(2**30)
    MemoryLimit(limit=1073741824, action='refuse', source='keyword')
    """
    if memory_budget is not None:
        return parse_memory_budget(memory_budget)
    declared = os.environ.get(MEMORY_BUDGET_VARIABLE)
    if declared:
        return parse_memory_budget(declared, source=MEMORY_BUDGET_VARIABLE)
    # Imported here, not at module level: the executors import the scan output, whose
    # own import of the driver package reaches this module first (a cycle).
    from solprop.driver.executors import pool  # noqa: PLC0415

    machine = pool._memory_limit()
    return MemoryLimit(
        math.floor(machine * (1.0 - DEFAULT_HEADROOM)), "warn", "machine"
    )


def admit_memory(
    problem: KernelProblem,
    backend: str,
    memory_budget: int | str | None = None,
    *,
    output: Output | None = None,
    retention: FieldRetention | None = None,
) -> MemoryAdmission:
    """Estimate a run's persistent memory and admit it against its budget.

    Called before the backend is constructed, so nothing the backend would allocate
    exists yet. The general route's reference is already resident, in the translated
    problem, and is counted with the rest. Given the run's output, what the output
    keeps in memory is added as the group ``output``: an in-memory store at its
    planned capacity, nothing for a file.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    backend : str
        The registry name the run will select, or ``"auto"``.
    memory_budget : int or str or None, optional
        This run's declaration (:func:`resolve_memory_budget`).
    output : solprop.output.schema.Output or None, optional
        The run's handler. A :class:`~solprop.output.schema.PlaneStore` is charged
        what :meth:`~solprop.output.schema.PlaneStore.held_bytes` says it keeps for
        the entrance and every save plane; ``None``, or another handler, is charged
        nothing.
    retention : solprop.output.schema.FieldRetention or None, optional
        Which planes keep their field, resolved against the run's planes; ``None``,
        every plane. The output is charged the field on the kept planes only.

    Returns
    -------
    MemoryAdmission
        The estimate and the budget, for the run's record.

    Raises
    ------
    MemoryBudgetError
        If the estimate exceeds a budget declared to refuse; the message names the
        estimate, the budget, the largest groups and what to change. Also, whatever
        the declaration but ``"none"`` or ``"warn:"``, if the Adams method's history
        is what does not fit -- the run without it would -- in the machine's memory
        or, on an accelerator, in four fifths of its free memory; the message names
        ``RungeKutta()``, which keeps no history.

    Warns
    -----
    MemoryBudgetWarning
        If the estimate exceeds a budget declared ``warn:``, or the undeclared
        machine budget for a reason other than the Adams history.

    Examples
    --------
    >>> from solprop.driver import Simulation
    >>> from solprop.driver.memory import admit_memory
    >>> simulation = Simulation.from_gnlse(
    ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
    ...     width=100e-15, peak_power=9.8, shape="sech",
    ... )
    >>> admission = admit_memory(simulation.translate(), "numpy_ref", "1 GiB")
    >>> admission.estimate.total < admission.budget.limit
    True
    >>> admit_memory(simulation.translate(), "numpy_ref", "1 kB")  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    solprop.exceptions.MemoryBudgetError: ...
    """
    own = estimate_memory(problem, backend)
    estimate = _with_held_plane(
        _with_output(own, problem, output, retention), problem, retention
    )
    _admit_history(
        estimate,
        resolve_memory_budget(memory_budget),
        march=estimate.total - own.total + low_storage_memory(problem, backend),
    )
    return _admitted(estimate, memory_budget, "the run's persistent memory")


def _with_output(
    estimate: MemoryEstimate,
    problem: KernelProblem,
    output: Output | None,
    retention: FieldRetention | None,
) -> MemoryEstimate:
    """Return the estimate with what the run's output keeps in memory added to it.

    Parameters
    ----------
    estimate : solprop.backends.memory.MemoryEstimate
        The backend's estimate.
    problem : solprop.translate.problem.KernelProblem
        The translated problem, which fixes the planes and their shapes.
    output : solprop.output.schema.Output or None
        The run's handler.
    retention : solprop.output.schema.FieldRetention or None
        Which planes keep their field, resolved; ``None``, every plane.

    Returns
    -------
    solprop.backends.memory.MemoryEstimate
        The estimate with an ``output`` group when the output keeps planes in memory;
        otherwise the estimate unchanged.
    """
    if not isinstance(output, PlaneStore):
        return estimate
    n_planes = len(problem.stepper.save_planes) + 1
    held = output.held_bytes(
        n_planes,
        plane_bytes(problem),
        kept=None
        if retention is None or retention.fields == "all"
        else retention.kept(n_planes),
        field_bytes=field_bytes(problem),
    )
    if not held:
        return estimate
    return replace(
        estimate,
        groups={**estimate.groups, "output": held},
        not_counted=(
            *estimate.not_counted,
            "the in-memory output's step history and between-planes rows, which grow "
            "with the steps taken",
        ),
    )


def _with_held_plane(
    estimate: MemoryEstimate, problem: KernelProblem, retention: FieldRetention | None
) -> MemoryEstimate:
    """Return the estimate with the low-storage march's held plane added to it.

    A low-storage run that keeps fewer than every field holds the state of its last
    field-free plane, one volume, so a step abandoned in flight can still be
    checkpointed (SEP 22, Q14).

    Parameters
    ----------
    estimate : solprop.backends.memory.MemoryEstimate
        The estimate so far.
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    retention : solprop.output.schema.FieldRetention or None
        Which planes keep their field; ``None``, every plane.

    Returns
    -------
    solprop.backends.memory.MemoryEstimate
        The estimate with a ``held_plane`` group where one is held.
    """
    if (
        retention is None
        or retention.fields == "all"
        or not isinstance(problem.stepper.integrator, LowStorageRungeKutta)
    ):
        return estimate
    return replace(
        estimate,
        groups={**estimate.groups, "held_plane": 16 * math.prod(problem.shape)},
    )


def plane_bytes(problem: KernelProblem) -> int:
    """Return the bytes one saved plane occupies in an in-memory store.

    The field (complex128), the noise phase (float64) when the run carries a noise
    model, every statistic at its catalogued shape (float64; a statistic catalogued
    with a smaller shape for spatial beams is counted at the modal one, an upper bound
    of a few entries), and the position and the two per-plane counts.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.

    Returns
    -------
    int
        Bytes per plane.

    Examples
    --------
    >>> from solprop.driver import Simulation
    >>> from solprop.driver.memory import plane_bytes
    >>> simulation = Simulation.from_gnlse(
    ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
    ...     width=100e-15, peak_power=9.8, shape="sech",
    ... )
    >>> problem = simulation.translate()
    >>> plane_bytes(problem) >= 16 * problem.shape[-1]
    True
    """
    _, n_modes, n_pol, n_omega = problem.shape
    entries = n_modes * n_pol * n_omega
    total = 16 * entries + 8 * _COUNTED_PER_PLANE
    if problem.initial.noise_model != "none":
        total += 8 * entries
    sizes = {
        "n_modes": n_modes,
        "n_pol": n_pol,
        "n_omega": n_omega,
        "n_windows": problem.stats_spec.n_windows,
        "n_apertures": problem.stats_spec.n_apertures,
        "n_regions": problem.stats_spec.n_regions,
        "n_species": problem.responses.n_species,
        "n_nodes": problem.transform.n_nodes,
    }
    for name in problem.stats_spec.names:
        total += 8 * _catalogued_entries(CATALOGUE[name].shape, sizes)
    return total


def field_bytes(problem: KernelProblem) -> int:
    """Return the bytes of one plane's field, and its noise phase, in a memory store.

    The part of :func:`plane_bytes` a plane that keeps no field does not hold (SEP 22).

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.

    Returns
    -------
    int
        Sixteen bytes per state entry, and eight more with a noise model.

    Examples
    --------
    >>> from solprop.driver import Simulation
    >>> from solprop.driver.memory import field_bytes
    >>> simulation = Simulation.from_gnlse(
    ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
    ...     width=100e-15, peak_power=9.8, shape="sech", noise=None,
    ... )
    >>> field_bytes(simulation.translate()) == 16 * simulation.translate().shape[-1]
    True
    """
    entries = math.prod(problem.shape[1:])
    return (16 + (8 if problem.initial.noise_model != "none" else 0)) * entries


def _catalogued_entries(shape: str, sizes: Mapping[str, int]) -> int:
    """Return how many entries a statistic of a catalogued shape holds.

    Parameters
    ----------
    shape : str
        The catalogue's shape, such as ``'(n_windows, n_modes)'``, ``'(2,)'`` or
        ``'()'``; an alternative after ``';'`` is ignored.
    sizes : collections.abc.Mapping
        The length of every axis name a catalogued shape uses.

    Returns
    -------
    int
        The product of the axis lengths, one for a scalar.

    Raises
    ------
    DriverError
        If the shape names an axis `sizes` does not give.
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


def _admit_history(
    estimate: MemoryEstimate, budget: MemoryLimit, *, march: int | None = None
) -> None:
    """Refuse a run that fits but for its Adams history, naming the marches that fit.

    The Adams history holds from seven to about sixty copies of the state, by backend
    and formulation, and the Runge-Kutta march none, so a problem near the memory's
    size may fit only with ``RungeKutta()``; the refusal also names the low-storage
    march (SEP 24) where its estimate fits too and the backend executes it.
    That run is refused before it allocates even where the machine default would only
    warn: under the default integrator it would page or fail to allocate. The limits
    are the machine's (an undeclared budget) and the accelerator's free memory less
    :data:`DEFAULT_HEADROOM`; a declared budget is checked by :func:`_admitted`, a
    ``"warn:"`` one only warns, and ``"none"`` declares no check.

    Parameters
    ----------
    estimate : solprop.backends.memory.MemoryEstimate
        The run's estimate.
    budget : MemoryLimit
        The budget the declaration resolved to.
    march : int or None, optional
        The same run's estimate under the low-storage march
        (:func:`solprop.backends.memory.low_storage_memory`, with whatever the
        estimate adds beside the backend's), bytes; ``None`` names no such march.

    Raises
    ------
    MemoryBudgetError
        Naming the estimate, the history, the limit and ``RungeKutta()``, and
        ``LowStorageRungeKutta()`` where its estimate fits.
    """
    history = int(estimate.groups.get("adams_history", 0))
    declared_warning = budget.action == "warn" and budget.source != "machine"
    if not history or budget.limit is None or declared_warning:
        return
    limits = []
    if budget.source == "machine":
        limits.append((budget.limit, "the machine's memory less its headroom"))
    free = available_device_memory(estimate.backend)
    if free is not None:
        limits.append(
            (
                math.floor(free * (1.0 - DEFAULT_HEADROOM)),
                f"the accelerator's free memory less its headroom ({format_bytes(free)}"
                " free)",
            )
        )
    for limit, where in limits:
        if estimate.total - history <= limit < estimate.total:
            # The simulation module imports this one.
            from solprop.driver.simulation import LOW_STORAGE_BACKENDS  # noqa: PLC0415

            low_storage = (
                ""
                if march is None
                or march > limit
                or estimate.backend not in LOW_STORAGE_BACKENDS
                else (
                    "; or replay a fixed schedule with "
                    "integrator=LowStorageRungeKutta(), which holds three copies of "
                    f"the state beside its scratch, estimated at {format_bytes(march)}"
                )
            )
            msg = (
                f"the run's persistent memory on {estimate.backend} is estimated at "
                f"{format_bytes(estimate.total)}, over {where}, "
                f"{format_bytes(limit)}; {format_bytes(history)} of it is the Adams "
                "method's history. Step it with integrator=RungeKutta(), which keeps "
                f"no history and needs fewer copies of the state{low_storage}; or "
                f"declare a budget with memory_budget= or {MEMORY_BUDGET_VARIABLE} "
                "('warn:...' or 'none' to try it anyway)"
            )
            raise MemoryBudgetError(msg)


def admit_guide_memory(
    guide: ModalGuide,
    grid: TimeGrid,
    *,
    z_max: float,
    backend: str,
    memory_budget: int | str | None = None,
) -> MemoryAdmission:
    """Admit a general modal guide's route against its budget before it is built.

    The route's dense E/H reference and its maps are allocated when a general modal
    simulation is constructed
    (:meth:`solprop.driver.simulation.Simulation.from_waveguide`), before
    :func:`admit_memory` can see a translated problem, and on a large guide they are
    nearly all of the run's memory. This admits the three groups
    :func:`solprop.backends.memory.estimate_guide_memory` takes from the guide's
    declared shape, under the same budget and the same refuse or warn policy, before any
    of it exists. The whole run is admitted again by :func:`admit_memory` once it is
    translated.

    Parameters
    ----------
    guide : solprop.modes.guide.ModalGuide
        A guide built by :meth:`solprop.modes.guide.ModalGuide.from_provider`.
    grid : solprop.grid.TimeGrid
        The run's field grid.
    z_max : float
        Where the run ends, m, positive.
    backend : str
        The registry name the run will select, or ``"auto"``.
    memory_budget : int or str or None, optional
        The declaration (:func:`resolve_memory_budget`).

    Returns
    -------
    MemoryAdmission
        The guide's estimate and the budget.

    Raises
    ------
    MemoryBudgetError
        If the estimate exceeds a budget declared to refuse; nothing of the route has
        been built.

    Warns
    -----
    MemoryBudgetWarning
        If the estimate exceeds a budget declared ``warn:``, or the machine's default.

    Examples
    --------
    >>> from solprop.driver.memory import admit_guide_memory
    >>> from solprop.grid import FieldGrid
    >>> from solprop.processing._examples import example_general_run
    >>> guide = example_general_run()[2]
    >>> grid = FieldGrid.from_limits(1e-5, 1.03e-6, (1.0e-6, 1.06e-6), 400e-15)
    >>> admit_guide_memory(  # doctest: +ELLIPSIS
    ...     guide, grid, z_max=1e-5, backend="numpy_ref", memory_budget="1 kB"
    ... )
    Traceback (most recent call last):
    ...
    solprop.exceptions.MemoryBudgetError: the general modal guide's reference...
    """
    return _admitted(
        estimate_guide_memory(guide, grid, z_max=z_max, backend=backend),
        memory_budget,
        "the general modal guide's reference and maps, before they are built,",
    )


def _admitted(
    estimate: MemoryEstimate, memory_budget: int | str | None, subject: str
) -> MemoryAdmission:
    """Check an estimate against the budget a declaration resolves to.

    Parameters
    ----------
    estimate : solprop.backends.memory.MemoryEstimate
        What is admitted.
    memory_budget : int or str or None
        The declaration (:func:`resolve_memory_budget`).
    subject : str
        What the estimate is of, as the message names it.

    Returns
    -------
    MemoryAdmission
        The estimate and the budget.

    Raises
    ------
    MemoryBudgetError
        Over a budget declared to refuse.

    Warns
    -----
    MemoryBudgetWarning
        Over a budget declared to warn.
    """
    budget = resolve_memory_budget(memory_budget)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "memory estimate of %s on %s: %s against a budget of %s (%s)",
            subject,
            estimate.backend,
            format_bytes(estimate.total),
            "none" if budget.limit is None else format_bytes(budget.limit),
            budget.source,
        )
    if budget.limit is None or estimate.total <= budget.limit:
        return MemoryAdmission(estimate, budget)
    message = _over_budget(estimate, budget, subject)
    if budget.action == "warn":
        warnings.warn(message, MemoryBudgetWarning, stacklevel=3)
        return MemoryAdmission(estimate, budget)
    raise MemoryBudgetError(message)


def format_bytes(size: int) -> str:
    """Return a byte count in binary units, to three significant figures.

    Parameters
    ----------
    size : int
        Bytes, non-negative.

    Returns
    -------
    str
        The count in B, KiB, MiB, GiB or TiB.

    Examples
    --------
    >>> from solprop.driver.memory import format_bytes
    >>> format_bytes(3 * 2**30 // 2), format_bytes(512)
    ('1.50 GiB', '512 B')
    """
    if size < 1024:
        return f"{size} B"
    exponent = 1
    while exponent < len(_BINARY) and size >= 1024 ** (exponent + 1):
        exponent += 1
    value, unit = size / 1024**exponent, _BINARY[exponent - 1]
    return f"{value:.3g} {unit}" if value >= 100 else f"{value:.2f} {unit}"


def _over_budget(estimate: MemoryEstimate, budget: MemoryLimit, subject: str) -> str:
    """Return the message naming an estimate over its budget and what to change.

    Parameters
    ----------
    estimate : solprop.backends.memory.MemoryEstimate
        The estimate over the budget.
    budget : MemoryLimit
        The budget, with a limit.
    subject : str
        What the estimate is of.

    Returns
    -------
    str
        The estimate, the budget, the largest groups and what to change.
    """
    assert budget.limit is not None
    largest = estimate.largest()
    groups = ", ".join(f"{name} {format_bytes(size)}" for name, size in largest)
    advice = "; ".join(f"{name}: {_ADVICE[name]}" for name, _ in largest)
    resident = (
        f" ({format_bytes(estimate.resident)} of it, the general route's reference, "
        "is already held by the translated problem)"
        if estimate.resident
        else ""
    )
    return (
        f"{subject} on {estimate.backend} is estimated at "
        f"{format_bytes(estimate.total)}{resident}, over the budget of "
        f"{format_bytes(budget.limit)} ({budget.source}). Largest groups: {groups}. "
        f"To fit, {advice}; or declare a larger budget with memory_budget= or "
        f"{MEMORY_BUDGET_VARIABLE} (for example '64 GiB', 'warn:64 GiB' or 'none')."
    )
