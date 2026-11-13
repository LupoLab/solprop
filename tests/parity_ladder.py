# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The parity ladder: a catalogue row on native against ``numpy_ref``, rung by rung.

The data model and the rungs; the rows and the design are in
``tests/capability_routes.py``. Every rung constructs both backends on the row's built
problem and goes through the native binding (the native state's own trial, march and
projections) or the driver, never a Python re-implementation. A comparison is relative
to a finite, non-zero reference, strictly inside a bound the row declares from
``tests/bounds.py`` before the run, in the row's metric: normwise
(``||ours - theirs|| / ||theirs||``) or peak-normalised
(``max |ours - theirs| / max |theirs|``).
"""

import contextlib
import copy
import dataclasses
import inspect
import math
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import Any, Literal, NamedTuple

import numpy as np

from bounds import departure
from builders import launch_derivative
from sessions.runs import native_source, saved, to_planes
from sessions.runs import run as session_run
from sessions.runs import summary as run_summary
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.passes import apply_hard_mask
from solprop.backends.session import ExecutionOptions, Start
from solprop.driver import Simulation
from solprop.stats import StatsSpec
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest

FIELD_RUNGS = (
    "source",
    "derivative",
    "projected",
    "linear",
    "one_step",
    "steps",
    "run",
)
"""The field rungs, cheapest first; a row's statistics follow them."""

LINEAR_SCALE = 1e-20
"""Factor on the launch of the ``linear`` rung: its cubic terms are ``1e-40`` of it."""

DETECTION = 1000.0
"""How far above a rung's bound switching the row's term off must move the reference.

A backend that dropped the term then misses by a thousand bounds, and one that got the
term wrong by a tenth of a per cent of itself fails the rung: the guard the optmat
response parity tests held their first stage to, which their rows
(``tests/backends/optmat_rows.py``) now hold at every guarded rung. A rung that cannot
clear it lowers it with a :class:`Detection` that says why, never below
:data:`LEAST_DETECTION`."""

LEAST_DETECTION = 100.0
"""The lowest guard a rung may declare: a term wrong by one per cent still fails it."""


@dataclasses.dataclass(frozen=True)
class Detection:
    """A rung's guard set below :data:`DETECTION`, and why that is still enough.

    ``threshold`` is how many of the rung's bounds the term must move its reference by,
    at least :data:`LEAST_DETECTION`; ``reason`` says what limits it (the measured
    margin and what sets it).
    """

    threshold: float
    reason: str


@dataclasses.dataclass(frozen=True)
class Built:
    """A row's translated problem, its run's request, its grid and execution options.

    ``modes`` and ``simulation`` are the mode set and the driver's run the problem came
    from, when it came from one: a backend takes the mode set and the run's profiles
    (the statistics need them), and the ``run`` rung runs the driver. ``readings``
    holds the reference's measurement of each rung on this build, made once
    (:func:`reference`), so that a bound can read one; a copy made with
    :func:`dataclasses.replace` starts without them.
    """

    problem: PropagationProblem
    run_request: RunRequest
    grid: Any
    options: ExecutionOptions | None = None
    modes: tuple[Any, ...] | None = None
    simulation: Simulation | None = None
    readings: dict[str, Any] = dataclasses.field(
        default_factory=dict, init=False, repr=False, compare=False
    )


Bound = Callable[[Built], float]
"""A rung's bound from the built row, fixed before native runs.

A bound may read the reference's own measurement of a rung that needs nothing of the
row (``derivative``, ``run``) through :func:`reference`: a growth factor or a rate's
sensitivity taken from the reference run, never from the backend under test."""


@dataclasses.dataclass(frozen=True)
class Statistic:
    """A statistic's bound and how it is compared.

    Relative and peak-normalised by default (NaN where both sides are NaN); ``absolute``
    for a dimensionless diagnostic whose relative change measures nothing;
    ``elementwise`` relative at every element (:func:`elementwise_departure`, what
    ``numpy.allclose`` with ``rtol`` alone holds). ``positive`` also requires the
    reference's value to be positive somewhere, so that the comparison sees something.
    """

    bound: float
    absolute: bool = False
    elementwise: bool = False
    positive: bool = False


StatisticRule = Callable[[Built, str, float], Statistic]
"""A statistic's bound from the built row, its name and its rung's field bound."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class Row:
    """One catalogue row: a problem factory, the claim it holds and its rungs' bounds.

    ``why`` is the physical claim the row holds, in one sentence; every failure quotes
    it. ``bound_note`` explains a bound with a floor, a margin or a choice (what was
    measured against it, and what a wrong backend would miss by). ``rungs`` maps each
    field rung of :data:`FIELD_RUNGS` the row runs to its bound; ``statistics`` maps a
    statistic, read at the end of the row's schedule and compared absolutely, to its
    bound. ``switch_off`` returns the problem with the row's term off: the guard that
    the term moves every field rung's reference by more than :data:`DETECTION` bounds,
    or by a rung's own ``detection``. ``floor`` builds a row whose own native
    disagreement at the same rung is added to the bound (a term beside the row's that
    is known to disagree by more than the row's own budget). ``masked_launch``
    evaluates the source at the launch hard-masked to the active band on both sides, as
    the native state loads it; ``source_z`` is where the source is evaluated, m.
    ``metadata`` is what the native backend's ``metadata()`` must report (a callable
    value is a predicate on it); ``expect`` maps a dotted path into the built row
    (``"problem.transform.n_nodes"``; a method is called, ``""`` is the row itself) to
    what it must be; ``quiet`` holds the warning filters the row's run needs.
    ``unguarded`` names a rung the term does not move far enough to guard, with why the
    rung is kept; it still runs at its bound, and the row's other rungs carry its guard.
    A row whose term reaches none of its rungs -- an old claim the guard found vacuous,
    kept -- names them all. ``metric`` is how every field rung and guard compares
    (``"normwise"`` or ``"peak"``, :data:`METRICS`). ``window`` cuts the ``steps`` rung
    to the schedule's first steps (the whole schedule by default). ``every_statistic``
    compares every statistic of the request's own specification at the end of the
    ``steps`` rung and of the ``run`` rung, each with the :class:`Statistic` it returns.
    ``context`` makes the context the whole check runs in, so that every backend of the
    row, the reference's and the guard's included, is constructed inside it (the private
    gated plasma benchmark, ``tests/gated_source.py``).
    """

    id: str
    why: str
    axes: Mapping[str, str]
    build: Callable[[], Built]
    rungs: Mapping[str, Bound]
    switch_off: Callable[[PropagationProblem], PropagationProblem]
    bound_note: str = ""
    statistics: Mapping[str, Bound] = dataclasses.field(default_factory=dict)
    floor: Callable[[], Built] | None = None
    masked_launch: bool = False
    source_z: float = 0.0
    metadata: Mapping[str, object] = dataclasses.field(default_factory=dict)
    expect: Mapping[str, object] = dataclasses.field(default_factory=dict)
    medium: bool = False
    quiet: tuple[str, ...] = ()
    unguarded: Mapping[str, str] = dataclasses.field(default_factory=dict)
    detection: Mapping[str, Detection] = dataclasses.field(default_factory=dict)
    metric: Literal["normwise", "peak"] = "normwise"
    window: int | None = None
    every_statistic: StatisticRule | None = None
    context: Callable[[], AbstractContextManager[object]] | None = None


@dataclasses.dataclass(frozen=True)
class Reading:
    """One rung's or statistic's comparison: what was measured against what bound.

    ``guard`` is how many bounds switching the term off moved the rung's reference
    (``None`` for an unguarded rung or a statistic).
    """

    rung: str
    measured: float
    bound: float
    guard: float | None = None


# --- the rungs ----------------------------------------------------------------------


class Marched(NamedTuple):
    """A march's field (a complete run's: every saved plane's) and its statistics."""

    field: np.ndarray
    statistics: dict[str, np.ndarray]


def _backend(backend_class: type, built: Built, request: RunRequest | None = None):
    """Construct a backend on the built row, its own request by default.

    With the row's mode set and its run's profiles where it has them, as the driver
    constructs one: the statistics read them.
    """
    given: dict[str, Any] = {}
    if built.modes is not None:
        given["modes"] = built.modes
    if built.simulation is not None and built.simulation.profiles:
        given["profiles"] = built.simulation.profiles
    return backend_class(
        built.problem,
        built.run_request if request is None else request,
        grid=built.grid,
        options=built.options,
        **given,
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


def derivative(backend_class: type, built: Built) -> np.ndarray:
    """Return the first stage's derivative at the launch, ``N(a_0)``.

    The probe of ``tests/backends/test_radial_nonlinear.py``: the reference's own
    nonlinear pass hoisted to the launch (:func:`builders.launch_derivative`), and
    native's state's zero-length trial on its staged rows, whose first stage is the same
    quantity in the interaction picture at its origin. The reference's lab-frame sample
    (:func:`source`) is not hoisted, and departs from this by 4.3e-13 of the peak on the
    optmat argon capillary (measured), where native agrees with it to 1.3e-15.
    """
    problem = built.problem
    assert problem.initial.noise0 is None, "the rung's launches carry no noise"
    backend = _backend(backend_class, built)
    try:
        if backend.name == "native":
            return native_source(backend, problem.initial.field0)
        return np.asarray(launch_derivative(backend, problem))
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


def first_steps(request: RunRequest, count: int) -> RunRequest:
    """Return the request ending after the first `count` steps of its schedule.

    The schedule is kept whole, since a cut one would miss a hard kink further along
    it; the run observes the `count`-th edge and ends there
    (:func:`sessions.runs.to_planes`).
    """
    schedule = request.stepper.schedule
    assert schedule is not None and 0 < count <= len(schedule.steps)
    return to_planes(request, [float(schedule.edges[count])])


def first_step(request: RunRequest) -> RunRequest:
    """Return the request ending after the first step of its schedule."""
    return first_steps(request, 1)


def _march(
    backend_class: type,
    built: Built,
    request: RunRequest,
    names: tuple[str, ...] | None = (),
    start: Start | None = None,
):
    """Run the session to the request's last plane, observing the statistics `names`.

    ``None`` observes the request's own statistics. Returns the last field, the run's
    summary and the last observation's statistics.
    """
    backend = _backend(backend_class, built, request)
    stats = None if names is None else StatsSpec(names=names)
    try:
        # The session owns the backend and closes it, as a service's run does.
        result = session_run(
            backend, built.problem, request.for_run(stats=stats), start, owns=True
        )
    finally:
        backend.close()  # idempotent: for a run that failed to start
    last = result.observations[-1]
    plane = float(request.stepper.save_planes[-1])
    assert last.z == plane, (
        f"{backend.name}'s last observation is at z = {last.z} m, not at the request's "
        f"last plane, {plane} m"
    )
    statistics = {name: np.asarray(value) for name, value in last.stats.items()}
    return result.fields[-1], run_summary(result), statistics


def one_step(
    backend_class: type, built: Built, start: Start | None = None
) -> np.ndarray:
    """Return the field after the schedule's first step, which must be one step."""
    field, summary, _ = _march(
        backend_class, built, first_step(built.run_request), start=start
    )
    assert summary.n_steps == 1, (
        f"{backend_class.__name__} took {summary.n_steps} steps over the schedule's "
        "first step, not one"
    )
    return field


def linear(backend_class: type, built: Built) -> np.ndarray:
    """Return one step of the launch scaled by :data:`LINEAR_SCALE`.

    Every pass of the step runs, and the cubic and delayed terms are ``1e-40`` of the
    linear one: what the rung compares is the linear operator.
    """
    problem = built.problem
    launch = problem.initial.field0 * LINEAR_SCALE
    return one_step(
        backend_class, built, Start(field=launch, z=float(problem.initial.z))
    )


def steps(
    backend_class: type,
    built: Built,
    names: tuple[str, ...] | None = (),
    window: int | None = None,
) -> Marched:
    """Return the field after the schedule, every step taken, and its statistics.

    `window` ends the march after the schedule's first steps; the statistics `names`
    (``None``: the request's own) are read at its end, from the same run.
    """
    schedule = built.run_request.stepper.schedule
    assert schedule is not None, "the row's request has no fixed schedule to march"
    if window is None:
        request, count = built.run_request, len(schedule.steps)
        assert float(request.stepper.save_planes[-1]) == float(schedule.edges[-1]), (
            "the request's last plane is not the schedule's end"
        )
    else:
        request, count = first_steps(built.run_request, window), window
    field, summary, statistics = _march(backend_class, built, request, names)
    assert summary.n_steps == count, (
        f"{backend_class.__name__} took {summary.n_steps} steps of the "
        f"{count} it marches"
    )
    return Marched(field, statistics)


def run(backend_class: Any, built: Built) -> Marched:
    """Return the complete run's saved fields and statistics, every step taken.

    A driver's row runs :meth:`solprop.driver.Simulation.run` through the output
    layer: the saved planes in the optics convention, the per-plane statistics and the
    periodic rows (``rows/<name>``), on the row's problem -- the simulation's own
    translation unless the row's term was switched off. An assembled row runs at the
    seam, through the backend's session.
    """
    schedule = built.run_request.stepper.schedule
    assert schedule is not None, "the row's request has no fixed schedule to run"
    if built.simulation is None:
        backend = _backend(backend_class, built)
        try:
            # The session owns the backend and closes it, as a service's run does.
            result = session_run(backend, built.problem, built.run_request, owns=True)
        finally:
            backend.close()  # idempotent: for a run that failed to start
        planes = saved(result)
        field = np.array([state for _, state in planes])
        statistics = {
            name: np.array([np.asarray(event.stats[name]) for event, _ in planes])
            for name in planes[-1][0].stats
        }
        accepted = run_summary(result).n_steps
    else:
        # A shallow copy whose settings name the backend and whose translation is the
        # row's, so the translation is shared and a switched-off problem runs too.
        simulation = copy.copy(built.simulation)
        simulation.settings = dataclasses.replace(
            simulation.settings, backend=backend_class.name
        )
        simulation._problem = built.problem
        handler = simulation.run(progress="none")
        field = handler.fields("optics")
        statistics = {
            name: np.asarray(handler[f"stats/{name}"]) for name in handler.stat_names
        } | {
            f"rows/{name}": np.asarray(handler[f"step_stats/{name}"])
            for name in handler.step_stat_names
        }
        accepted = int(np.count_nonzero(handler["step_history/accepted"]))
    assert accepted == len(schedule.steps), (
        f"{backend_class.name}'s complete run accepted {accepted} steps of the "
        f"schedule's {len(schedule.steps)}"
    )
    return Marched(field, statistics)


RUNGS: dict[str, Callable[[type, Built, Any], Any]] = {
    "source": lambda backend_class, built, row: source(
        backend_class, built, masked=row.masked_launch, z=row.source_z
    ),
    "derivative": lambda backend_class, built, row: derivative(backend_class, built),
    "projected": lambda backend_class, built, row: projected(backend_class, built),
    "linear": lambda backend_class, built, row: linear(backend_class, built),
    "one_step": lambda backend_class, built, row: one_step(backend_class, built),
    "steps": lambda backend_class, built, row: steps(
        backend_class,
        built,
        None if row.every_statistic is not None else tuple(row.statistics),
        row.window,
    ),
    "run": lambda backend_class, built, row: run(backend_class, built),
}
"""Each field rung's measurement, by name."""

ROW_FREE_RUNGS = frozenset({"derivative", "projected", "linear", "one_step", "run"})
"""The rungs a bound may read the reference's measurement of (:func:`reference`)."""


def reference(built: Built, rung: str, row: Row | None = None) -> Any:
    """Return the reference's measurement of a rung on the built row, made once.

    A bound reads one without the row, of a rung in :data:`ROW_FREE_RUNGS`; the check
    passes its row, and both read the same measurement.
    """
    if rung not in built.readings:
        assert row is not None or rung in ROW_FREE_RUNGS, (
            f"the {rung} rung reads the row: a bound cannot read it"
        )
        built.readings[rung] = RUNGS[rung](NumpyRefBackend, built, row)
    return built.readings[rung]


# --- the comparison -------------------------------------------------------------------


def _pairs(ours, theirs) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the arrays a rung compares, the same ones present on both sides."""
    if not isinstance(theirs, list):
        return [(np.asarray(ours), np.asarray(theirs))]
    assert [one is None for one in ours] == [one is None for one in theirs], (
        "native and the reference hold different quadrature rules (None: no rule): "
        f"{[one is None for one in ours]} against {[one is None for one in theirs]}"
    )
    return [
        (np.asarray(a), np.asarray(b))
        for a, b in zip(ours, theirs, strict=True)
        if b is not None
    ]


def peak_departure(got, expected) -> float:
    """Return ``max |got - expected| / max |expected|``, NaN where both are NaN.

    A NaN on one side only is an infinite departure; against a zero reference, zero
    when the other side is zero too.
    """
    got, expected = np.asarray(got), np.asarray(expected)
    if np.any(np.isnan(got) != np.isnan(expected)):
        return math.inf
    missing = np.isnan(expected)
    scale = float(np.max(np.abs(np.where(missing, 0.0, expected)), initial=0.0))
    if scale == 0.0:
        return 0.0 if np.all(np.where(missing, 0.0, got) == 0.0) else math.inf
    return float(np.max(np.abs(np.where(missing, 0.0, got - expected))) / scale)


def elementwise_departure(got, expected) -> float:
    """Return the largest ``|got - expected| / |expected|`` over the elements.

    ``numpy.allclose(got, expected, rtol=r, atol=0)`` is this ``<= r``: a NaN on either
    side is an infinite departure, and so is a non-zero value against a zero reference;
    zero against zero is none.
    """
    got, expected = np.asarray(got, dtype=float), np.asarray(expected, dtype=float)
    if np.any(np.isnan(got)) or np.any(np.isnan(expected)):
        return math.inf
    difference, scale = np.abs(got - expected), np.abs(expected)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(
            scale > 0, difference / scale, np.where(difference == 0, 0.0, math.inf)
        )
    return float(np.max(ratios, initial=0.0))


METRICS: dict[str, Callable[[Any, Any], float]] = {
    "normwise": departure,
    "peak": peak_departure,
}
"""How a row compares its fields: ``||a - b|| / ||b||``, or the peak-normalised
``max |a - b| / max |b|`` of the optmat response parity, whose bounds count roundings
per sample of the peak."""


def disagreement(ours, theirs, metric: str = "normwise") -> float:
    """Return the worst relative departure over a rung's arrays, in `metric`.

    Every array finite, of one size on both sides (a backend may hold a column's
    polarisation axis flattened), and the reference's non-zero, so that a backend
    returning zeros cannot pass.
    """
    worst = 0.0
    for got, expected in _pairs(ours, theirs):
        a, b = np.ravel(got), np.ravel(expected)
        assert a.size == b.size, f"native holds {a.size} values, the reference {b.size}"
        assert np.all(np.isfinite(a)), "native's output is not finite"
        assert np.all(np.isfinite(b)), "the reference's output is not finite"
        assert np.linalg.norm(b) > 0, (
            "the reference's output is zero: nothing to compare"
        )
        worst = max(worst, METRICS[metric](a, b))
    return worst


def _field(measured):
    """Return a rung's field: a march's, or the measurement itself."""
    return measured.field if isinstance(measured, Marched) else measured


def check(native_class: type, row: Row) -> list[Reading]:
    """Run every rung of a row on native and the reference; assert each in its bound.

    Also asserts the row's expectations, its native metadata and the guard: with the
    row's term switched off -- for a row with a floor, the floor's reference, which
    differs from the row by the term the floor replaces -- each guarded field rung's
    reference moves by more than :data:`DETECTION` times that rung's bound (or the
    rung's own :class:`Detection`). Every failure names the row and quotes its claim
    and bound note; a rung outside its bound gives the measured value, the bound and
    their ratio. A row's ``context`` holds for the whole check.
    """
    context = contextlib.nullcontext() if row.context is None else row.context()
    try:
        with context:
            return _check(native_class, row)
    except AssertionError as error:
        raise AssertionError(_failure(row, str(error))) from error


def _failure(row: Row, what: str) -> str:
    """Return a failure's message: the row, its claim and bound note, what failed."""
    lines = [f"row {row.id!r}", f"  claim: {row.why}"]
    if row.bound_note:
        lines.append(f"  bound: {row.bound_note}")
    lines += [f"  failed: {line}" for line in what.splitlines() or ["(no message)"]]
    return "\n".join(lines)


def _check(native_class: type, row: Row) -> list[Reading]:
    """Run :func:`check`'s comparisons; the caller adds the row to a failure."""
    _check_declarations(row)
    built = row.build()
    _matches(
        {path: _resolve(built, path) for path in row.expect}, row.expect, "built row"
    )
    if row.metadata:
        backend = _backend(native_class, built)
        try:
            metadata = backend.metadata()
        finally:
            backend.close()
        _matches(metadata, row.metadata, "native metadata")
    floor = None if row.floor is None else row.floor()
    off = dataclasses.replace(built, problem=row.switch_off(built.problem))
    field_bounds: dict[str, float] = {}
    marched: dict[str, tuple[Marched, Marched]] = {}
    readings: list[Reading] = []
    failures: list[str] = []
    for rung, bound_of in row.rungs.items():
        bound = bound_of(built)
        assert bound > 0, f"{rung}: the bound is {bound}, not positive"
        ours = RUNGS[rung](native_class, built, row)
        theirs = reference(built, rung, row)
        if isinstance(theirs, Marched):
            marched[rung] = (ours, theirs)
        reference_off = None
        if floor is not None:
            reference_off = _field(reference(floor, rung, row))
            bound += disagreement(
                _field(RUNGS[rung](native_class, floor, row)), reference_off, row.metric
            )
        field_bounds[rung] = bound
        guard = None
        if rung not in row.unguarded:
            if reference_off is None:
                reference_off = _field(reference(off, rung, row))
            guard = _share(reference_off, _field(theirs), row.metric) / bound
            lowered = row.detection.get(rung)
            threshold = DETECTION if lowered is None else lowered.threshold
            if not guard > threshold:
                failures.append(
                    f"{rung} guard: switching the term off moves the reference by "
                    f"{guard:.3g} bounds, not more than {threshold:g}, so a backend "
                    "that dropped the term could pass"
                    + ("" if lowered is None else f" (lowered: {lowered.reason})")
                )
        measured = disagreement(_field(ours), _field(theirs), row.metric)
        readings.append(Reading(rung, measured, bound, guard))
    readings += _named_statistics(row, built, marched)
    readings += _every_statistic(row, built, marched, field_bounds)
    failures += [
        f"{one.rung}: native departs from the reference by {one.measured:.3e}, "
        f"not inside its bound {one.bound:.3e} ({one.measured / one.bound:.3g} bounds)"
        for one in readings
        if not one.measured < one.bound
    ]
    assert not failures, "\n".join(failures)
    return readings


def _named_statistics(
    row: Row, built: Built, marched: Mapping[str, tuple[Marched, Marched]]
) -> list[Reading]:
    """Compare the row's named statistics at the end of its steps, absolutely."""
    if not row.statistics:
        return []
    ours, theirs = marched["steps"]
    readings = []
    for name, bound_of in row.statistics.items():
        found, expected = float(ours.statistics[name]), float(theirs.statistics[name])
        assert expected > 0, f"{name}: the reference's statistic is not positive"
        readings.append(Reading(name, abs(found - expected), bound_of(built)))
    return readings


def _every_statistic(
    row: Row,
    built: Built,
    marched: Mapping[str, tuple[Marched, Marched]],
    field_bounds: Mapping[str, float],
) -> list[Reading]:
    """Compare every statistic the steps and run rungs read, as the row's rule says."""
    if row.every_statistic is None:
        return []
    readings = []
    for rung, (ours, theirs) in marched.items():
        assert set(ours.statistics) == set(theirs.statistics), (
            f"{rung}: native reads the statistics {sorted(ours.statistics)}, the "
            f"reference {sorted(theirs.statistics)}"
        )
        for name, expected in theirs.statistics.items():
            rule = row.every_statistic(built, name, field_bounds[rung])
            assert not (rule.absolute and rule.elementwise), (
                f"{rung} {name}: a statistic is compared absolutely or elementwise"
            )
            assert not rule.positive or np.max(expected) > 0, (
                f"{rung} {name}: the reference's statistic is nowhere positive, so the "
                "comparison sees nothing"
            )
            found = ours.statistics[name]
            if rule.absolute:
                measured = float(np.max(np.abs(found - expected)))
            elif rule.elementwise:
                measured = elementwise_departure(found, expected)
            else:
                measured = peak_departure(found, expected)
            readings.append(Reading(f"{rung} {name}", measured, rule.bound))
    return readings


def _check_declarations(row: Row) -> None:
    """Assert the row states its claim and declares its guards completely."""
    assert row.why.strip(), "the row states no claim (`why`)"
    assert set(row.unguarded) <= set(row.rungs), (
        "a row names as unguarded only rungs it runs"
    )
    assert all(reason.strip() for reason in row.unguarded.values()), (
        "an unguarded rung gives no reason"
    )
    assert not row.statistics or "steps" in row.rungs, (
        "statistics are read at the end of the steps rung, which the row does not run"
    )
    assert row.every_statistic is None or not row.statistics, (
        "a row compares its named statistics or every statistic, not both"
    )
    assert row.every_statistic is None or {"steps", "run"} & set(row.rungs), (
        "every statistic is read at the end of the steps or run rung, which the row "
        "does not run"
    )
    assert row.window is None or "steps" in row.rungs, (
        "a window cuts the steps rung, which the row does not run"
    )
    assert row.metric in METRICS, f"no metric {row.metric!r}"
    for rung, lowered in row.detection.items():
        assert rung in row.rungs and rung not in row.unguarded, (
            f"{rung}: a lowered guard names a guarded rung the row runs"
        )
        assert LEAST_DETECTION <= lowered.threshold < DETECTION, (
            f"{rung}: a lowered guard of {lowered.threshold:g} bounds is outside "
            f"[{LEAST_DETECTION:g}, {DETECTION:g})"
        )
        assert lowered.reason.strip(), f"{rung}: a lowered guard gives no reason"


def _resolve(built: Built, path: str) -> object:
    """Return the attribute at a dotted path from the built row, a method called."""
    found: Any = built
    for name in filter(None, path.split(".")):
        found = getattr(found, name)
    return found() if inspect.ismethod(found) else found


def _matches(actual: Mapping[str, Any], expected: Mapping[str, Any], what: str) -> None:
    """Assert every key `expected` names: equal, or satisfying it if it is callable."""
    for key, value in expected.items():
        if callable(value):
            assert value(actual[key]), (
                f"{what} {key or '(the row)'}: {actual[key]!r} fails "
                f"{getattr(value, '__name__', value)}"
            )
        else:
            assert actual[key] == value, (
                f"{what} {key}: {actual[key]!r}, expected {value!r}"
            )


def _share(off, on, metric: str) -> float:
    """Return how far the term-off output is from the row's, relative to the row's."""
    return max(METRICS[metric](a, b) for a, b in _pairs(off, on))
