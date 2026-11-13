# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The parity ladder: a catalogue row on native against ``numpy_ref``, rung by rung.

The data model and the rungs; the rows and the design are in
``tests/capability_routes.py``. Every rung constructs both backends on the row's built
problem and goes through the native binding (the native state's own trial, march and
projections), never a Python re-implementation. A comparison is normwise relative
(``||ours - theirs|| / ||theirs||``) against a finite, non-zero reference, strictly
inside a bound the row declares from ``tests/bounds.py`` before the run.
"""

import dataclasses
import inspect
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from bounds import departure
from sessions.runs import native_source
from sessions.runs import run as session_run
from sessions.runs import summary as run_summary
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.passes import apply_hard_mask
from solprop.backends.session import ExecutionOptions
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest

FIELD_RUNGS = ("source", "projected", "one_step", "steps")
"""The field rungs, cheapest first; a row's statistics follow them."""

DETECTION = 100.0
"""How far above a rung's bound switching the row's term off must move the reference.

A backend that dropped the term then misses by a hundred bounds, and one that got the
term wrong by one per cent of itself fails the rung."""


@dataclasses.dataclass(frozen=True)
class Built:
    """A row's translated problem, its run's request, its grid and execution options."""

    problem: PropagationProblem
    run_request: RunRequest
    grid: Any
    options: ExecutionOptions | None = None


Bound = Callable[[Built], float]
"""A rung's bound from the built row, fixed before either backend runs."""


@dataclasses.dataclass(frozen=True)
class Row:
    """One catalogue row: a problem factory, the axes it covers and its rungs' bounds.

    ``rungs`` maps each field rung of :data:`FIELD_RUNGS` the row runs to its bound;
    ``statistics`` maps a statistic, read at the end of the row's schedule and compared
    absolutely, to its bound. ``switch_off`` returns the problem with the row's term
    off: the guard that the term moves every field rung's reference by more than
    :data:`DETECTION` bounds. ``floor`` builds a row whose own native disagreement at
    the same rung is added to the bound (a term beside the row's that is known to
    disagree by more than the row's own budget). ``masked_launch`` evaluates the
    source at the launch hard-masked to the active band on both sides, as the native
    state loads it; ``source_z`` is where the source is evaluated, m. ``metadata`` is
    what the native backend's ``metadata()`` must report (a callable value is a
    predicate on it); ``expect`` maps a dotted path into the built row
    (``"problem.transform.n_nodes"``; a method is called, ``""`` is the row itself) to
    what it must be; ``quiet`` holds the warning filters the row's run needs.
    ``unguarded`` names a rung the term does not move far enough to guard, with why the
    rung is kept; it still runs at its bound, and the row's other rungs carry its
    guard.
    """

    id: str
    axes: Mapping[str, str]
    build: Callable[[], Built]
    rungs: Mapping[str, Bound]
    switch_off: Callable[[PropagationProblem], PropagationProblem]
    statistics: Mapping[str, Bound] = dataclasses.field(default_factory=dict)
    floor: Callable[[], Built] | None = None
    masked_launch: bool = False
    source_z: float = 0.0
    metadata: Mapping[str, object] = dataclasses.field(default_factory=dict)
    expect: Mapping[str, object] = dataclasses.field(default_factory=dict)
    medium: bool = False
    quiet: tuple[str, ...] = ()
    unguarded: Mapping[str, str] = dataclasses.field(default_factory=dict)


# --- the rungs ----------------------------------------------------------------------


def _backend(backend_class: type, built: Built, request: RunRequest | None = None):
    """Construct a backend on the built row, its own request by default."""
    return backend_class(
        built.problem,
        built.run_request if request is None else request,
        grid=built.grid,
        options=built.options,
    )


def source(
    backend_class: type, built: Built, *, masked: bool = False, z: float = 0.0
) -> np.ndarray:
    """Return one evaluation of the whole stage source at the launch, lab frame, at `z`.

    Native's is its state's zero-length trial (:func:`sessions.runs.native_source`),
    the reference's its lab-frame sample; `masked` hard-masks the launch first.
    """
    problem = built.problem
    launch = problem.initial.field0
    if masked:
        launch = apply_hard_mask(launch[None], problem.grids.active_mask)[0]
    backend = _backend(backend_class, built)
    try:
        if backend.name == "native":
            return native_source(backend, launch, z)
        backend.set_initial(launch, None)
        return np.array(backend._adams_lab_source(launch[None], z))
    finally:
        backend.close()


def projected(backend_class: type, built: Built) -> list[np.ndarray | None]:
    """Return one evaluation's projected polarisation spectra, both rules, unnormalised.

    Native keeps its last evaluation's spectra, the embedded rule finished when read;
    the reference keeps its projections in time, whose transforms these are.
    """
    problem = built.problem
    backend = _backend(backend_class, built)
    try:
        if backend.name == "native":
            native_source(backend, problem.initial.field0)
            native = backend._native_state
            return [
                np.array(native.projected_to_host(False)),
                np.array(native.projected_to_host(True)) if embedded(problem) else None,
            ]
        forward = np.fft.rfft if problem.grid_kind == "field" else np.fft.fft
        backend.set_initial(problem.initial.field0, None)
        backend._adams_lab_source(problem.initial.field0[None], 0.0)
        coarse = backend._last_coarse
        return [
            forward(np.asarray(backend._last_projected), axis=-1),
            None if coarse is None else forward(np.asarray(coarse), axis=-1),
        ]
    finally:
        backend.close()


def embedded(problem: PropagationProblem) -> bool:
    """Return whether the quadrature has an embedded rule of its own, as packed."""
    transform = problem.transform
    return transform.n_nodes > 1 and not np.array_equal(
        transform.projection, transform.projection_coarse
    )


def first_step(request: RunRequest) -> RunRequest:
    """Return the request cut to the first step of its schedule."""
    schedule = request.stepper.schedule
    assert schedule is not None
    length = float(schedule.edges[1])
    return request.with_stepper(
        dataclasses.replace(
            request.stepper,
            schedule=FixedSchedule.uniform(0.0, length, length),
            save_planes=(length,),
        )
    )


def _march(
    backend_class: type, built: Built, request: RunRequest, names: tuple[str, ...] = ()
):
    """Run the session to the request's last plane, observing the statistics `names`.

    Returns the last field, the run's summary and the last observation's statistics.
    """
    backend = _backend(backend_class, built, request)
    try:
        result = session_run(
            backend, built.problem, request.for_run(stats=StatsSpec(names=names))
        )
    finally:
        backend.close()
    last = result.observations[-1]
    assert last.z == float(request.stepper.save_planes[-1])
    statistics = {name: float(last.stats[name]) for name in names}
    return result.fields[-1], run_summary(result), statistics


def one_step(backend_class: type, built: Built) -> np.ndarray:
    """Return the field after the schedule's first step, which must be one step."""
    field, summary, _ = _march(backend_class, built, first_step(built.run_request))
    assert summary.n_steps == 1
    return field


def steps(
    backend_class: type, built: Built, names: tuple[str, ...] = ()
) -> tuple[np.ndarray, dict[str, float]]:
    """Return the field after the whole schedule, every step taken, and statistics.

    The statistics `names` are read at the schedule's end, from the same run.
    """
    schedule = built.run_request.stepper.schedule
    assert schedule is not None
    assert float(built.run_request.stepper.save_planes[-1]) == float(schedule.edges[-1])
    field, summary, statistics = _march(backend_class, built, built.run_request, names)
    assert summary.n_steps == len(schedule.steps)
    return field, statistics


RUNGS: dict[str, Callable[[type, Built, Row], Any]] = {
    "source": lambda backend_class, built, row: source(
        backend_class, built, masked=row.masked_launch, z=row.source_z
    ),
    "projected": lambda backend_class, built, row: projected(backend_class, built),
    "one_step": lambda backend_class, built, row: one_step(backend_class, built),
    "steps": lambda backend_class, built, row: steps(
        backend_class, built, tuple(row.statistics)
    )[0],
}
"""Each field rung's measurement, by name."""


# --- the comparison -------------------------------------------------------------------


def _pairs(ours, theirs) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the arrays a rung compares, the same ones present on both sides."""
    if not isinstance(theirs, list):
        return [(np.asarray(ours), np.asarray(theirs))]
    assert [one is None for one in ours] == [one is None for one in theirs]
    return [
        (np.asarray(a), np.asarray(b))
        for a, b in zip(ours, theirs, strict=True)
        if b is not None
    ]


def disagreement(ours, theirs) -> float:
    """Return the worst normwise relative departure over a rung's arrays.

    Every array finite, of one size on both sides (a backend may hold a column's
    polarisation axis flattened), and the reference's non-zero, so that a backend
    returning zeros cannot pass.
    """
    worst = 0.0
    for got, expected in _pairs(ours, theirs):
        a, b = np.ravel(got), np.ravel(expected)
        assert a.size == b.size, (a.size, b.size)
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
        assert np.linalg.norm(b) > 0
        worst = max(worst, departure(a, b))
    return worst


def check(native_class: type, row: Row) -> list[tuple[str, float, float]]:
    """Run every rung of a row on native and the reference; assert each in its bound.

    Also asserts the row's expectations, its native metadata and the guard: with the
    row's term switched off -- for a row with a floor, the floor's reference, which
    differs from the row by the term the floor replaces -- each guarded field rung's
    reference moves by more than :data:`DETECTION` times that rung's bound.
    Returns ``(rung, measured, bound)``.
    """
    built = row.build()
    _matches({path: _resolve(built, path) for path in row.expect}, row.expect)
    if row.metadata:
        backend = _backend(native_class, built)
        try:
            metadata = backend.metadata()
        finally:
            backend.close()
        _matches(metadata, row.metadata)
    assert set(row.unguarded) < set(row.rungs), "a row guards at least one rung"
    assert not row.statistics or "steps" in row.rungs
    floor = None if row.floor is None else row.floor()
    off = dataclasses.replace(built, problem=row.switch_off(built.problem))
    names = tuple(row.statistics)
    found: dict[str, float] = {}
    expected: dict[str, float] = {}
    rows = []
    for rung, bound_of in row.rungs.items():
        bound = bound_of(built)
        assert bound > 0
        measure = RUNGS[rung]
        if rung == "steps":
            ours, found = steps(native_class, built, names)
            theirs, expected = steps(NumpyRefBackend, built, names)
        else:
            ours = measure(native_class, built, row)
            theirs = measure(NumpyRefBackend, built, row)
        reference_off = None
        if floor is not None:
            reference_off = measure(NumpyRefBackend, floor, row)
            bound += disagreement(measure(native_class, floor, row), reference_off)
        rows.append((rung, disagreement(ours, theirs), bound))
        if rung not in row.unguarded:
            if reference_off is None:
                reference_off = measure(NumpyRefBackend, off, row)
            share = _share(reference_off, theirs)
            assert share > DETECTION * bound, (rung, "guard", share, bound)
    for name, bound_of in row.statistics.items():
        assert expected[name] > 0
        rows.append((name, abs(found[name] - expected[name]), bound_of(built)))
    outside = [one for one in rows if not one[1] < one[2]]
    assert not outside, f"{row.id}: {outside}"
    return rows


def _resolve(built: Built, path: str) -> object:
    """Return the attribute at a dotted path from the built row, a method called."""
    found: Any = built
    for name in filter(None, path.split(".")):
        found = getattr(found, name)
    return found() if inspect.ismethod(found) else found


def _matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Assert every key `expected` names: equal, or satisfying it if it is callable."""
    for key, value in expected.items():
        if callable(value):
            assert value(actual[key]), (key, actual[key])
        else:
            assert actual[key] == value, (key, actual[key], value)


def _share(off, on) -> float:
    """Return how far the term-off output is from the row's, relative to the row's."""
    return max(
        float(np.linalg.norm(a - b) / np.linalg.norm(b)) for a, b in _pairs(off, on)
    )
