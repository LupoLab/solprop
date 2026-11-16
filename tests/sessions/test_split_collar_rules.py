# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The split collar's sums across a session: SEP 52 section 5.2's rules, pinned.

Under ``absorber_scheme="split_step"`` each accepted step's finish reduces the field
after its first band mask and before the split: what the split takes, and the power it
finds halfway through its factor, which the guards are judged on (SEP 52 section 4,
``BoundaryMonitor.split_sums``). Today the reference keeps those sums on its driver
(``_finish`` into ``_split_removed`` and ``_split_found``, drained by
``_boundary_energies``, reset by ``_bind_monitor``) and native in its state and session.
SEP 52 phase 3 moves their ownership into the session; these tests hold the rules it
must keep, through sessions on both backends, before anything moves:

- the removal adds over every accepted step since the previous reading, and the next
  boundary reading drains it;
- the occupancies a reading judges are those the latest accepted step found;
- a rejected trial and a probe trial contribute nothing, and a dense observation cannot,
  since it is refused beside the split collar;
- loading a state empties the sums, and a monitor attached at a launch or a resume
  starts from zero.

The case is the corpus's walking fibre (``backend_corpus_cases.walking_fibre``): ten
fixed steps on a grid declared a hundred times longer than the fibre, so that a quiet
window is read about every fourth step and at the saved planes, and the split takes
some of the pulse at every step. Saving a plane at every step edge reads every step,
which gives each step's removal and occupancies; the run is otherwise the same discrete
march. Comparisons are to :func:`march_bounds.reading_tolerance` at the matched march's
budget (:func:`march_bounds.simulation_bound`), and each claim is shown non-vacuous:
the removal or occupancy it would add or confuse is more than a hundred tolerances.
``tests/test_monitor.py::test_a_deep_collar_reads_no_removal_from_residue`` holds the
measurement point itself.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from backend_corpus_cases import FIBRE, FIXED_ADAMS, STEPS, walking_fibre
from march_bounds import assert_record_within, reading_tolerance, simulation_bound
from sessions.executors import IDENTITY
from solprop.backends.events import ObservationReady, Terminal
from solprop.backends.registry import available, select
from solprop.backends.session import Start
from solprop.backends.sessions import start_session
from solprop.driver import Simulation
from solprop.exceptions import TranslationError
from solprop.output.store import PlaneStore
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, RungeKutta

pytestmark = pytest.mark.medium

BACKENDS = [
    "numpy_ref",
    pytest.param(
        "native",
        marks=pytest.mark.skipif(
            "native" not in available(), reason="native is not installed"
        ),
    ),
]

MARCHES = {
    "rk": RungeKutta(),
    "adams": FIXED_ADAMS,
    "low-storage": LowStorageRungeKutta(),
}
"""The fixed-schedule marches, each of which finishes its accepted steps itself."""

EVERY = tuple(FIBRE * step / STEPS for step in range(1, STEPS + 1))
"""A save plane at every step edge: every step's state is read."""

DETECTION = 100.0
"""How many tolerances what a rule rules out must exceed, for the test to see it."""


def _backend(simulation: Simulation) -> tuple[Any, Any, Any]:
    """Return a simulation's problem, its request with its monitor, and a backend."""
    problem = simulation.translate()
    request = simulation._run_request()
    backend = select(
        simulation.settings.backend,
        problem,
        request,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    return problem, request, backend


def _watched(simulation: Simulation, request: Any, start_plane: int = 0) -> Any:
    """Return the request with a fresh boundary monitor, as a run attaches one."""
    return request.for_run(
        monitor=simulation._boundary_monitor(), start_plane=start_plane
    )


def _drive(
    backend: Any,
    problem: Any,
    request: Any,
    start: Start,
    *,
    advances: int | None = None,
    planes: int | None = None,
) -> tuple[dict[str, Any], list[tuple[float, np.ndarray]]]:
    """Drive a session an attempt at a time; return its monitor record and planes.

    The session stops after `advances` yields (the launch's observation is the first)
    or once it observed `planes` save planes, or at its end; the observations' fields
    are exported as the session holds them.
    """
    session = start_session(backend, problem, request, start, owns=False)
    observed: list[tuple[float, np.ndarray]] = []
    try:
        count = 0
        while not session.finished:
            batch = session.advance(work_limit=1)
            for event in batch.events:
                if isinstance(event, Terminal) and event.error is not None:
                    raise event.error
                if isinstance(event, ObservationReady) and event.plane_index >= 0:
                    exported = session.export(event.ticket, IDENTITY)
                    observed.append((float(event.z), np.array(exported)))
            count += 1
            if (advances is not None and count >= advances) or (
                planes is not None and len(observed) >= planes
            ):
                break
            session.acknowledge(batch.receipt)
        return dict(session.monitor_record() or {}), observed
    finally:
        session.close()


def _record(simulation: Simulation) -> dict[str, Any]:
    """Return the monitor record of a whole run of `simulation` on a fresh backend."""
    problem, request, backend = _backend(simulation)
    try:
        record, _ = _drive(
            backend, problem, _watched(simulation, request), Start.entrance(problem)
        )
    finally:
        backend.close()
    return record


@functools.cache
def _runs(backend: str, march: str) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Return a sparsely read run's record, the every-step run's, and their budget."""
    sparse = walking_fibre(backend, MARCHES[march])
    return (
        _record(sparse),
        _record(walking_fibre(backend, MARCHES[march], planes=EVERY)),
        simulation_bound(sparse),
    )


def _per_step(every: Mapping[str, Any]) -> np.ndarray:
    """Return each step's removal from a run read at every step's plane."""
    rows = every["planes"]
    # The run carries no input noise, so the removal above the floor is the whole.
    return np.diff(np.asarray(rows["removed_fraction"], dtype=np.float64))


@pytest.mark.parametrize("march", sorted(MARCHES))
@pytest.mark.parametrize("backend", BACKENDS)
def test_the_removal_adds_every_step_since_the_last_reading(backend, march):
    sparse, every, bound = _runs(backend, march)
    rows = every["planes"]
    # Every step of the every-step run is a plane read when it was reached...
    assert len(rows["z"]) == STEPS + 1
    assert None not in rows["guard_high"]
    # ...while the sparse run read its state at fewer than half its steps (its checked
    # steps and at most its two planes'), so some readings drained several steps.
    assert sparse["steps"] + 2 < STEPS // 2
    assert sparse["skipped"] > 0
    total = float(every["raw_removed_fraction"])
    tolerance = reading_tolerance(total, bound, STEPS)
    steps = _per_step(every)
    # A step dropped, or a drained step counted again, moves the total by a step's
    # removal: at least a hundred tolerances.
    assert np.all(steps > DETECTION * tolerance), (steps.min(), tolerance)
    assert abs(float(sparse["raw_removed_fraction"]) - total) <= tolerance
    # The middle plane's reading drained the five steps before it.
    middle = STEPS // 2
    row = float(sparse["planes"]["removed_fraction"][1])
    expected = float(rows["removed_fraction"][middle])
    assert abs(row - expected) <= reading_tolerance(expected, bound, middle)


@pytest.mark.parametrize("march", sorted(MARCHES))
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_reading_judges_the_latest_step_s_occupancies(backend, march):
    sparse, every, bound = _runs(backend, march)
    rows, last = every["planes"], sparse["planes"]
    for name in ("guard_low", "guard_high", "collar_occupancy"):
        series = np.asarray(rows[name], dtype=np.float64)
        expected = float(series[-1])
        tolerance = reading_tolerance(expected, bound, 1)
        # The walk fills the collar and the guards step by step, so the steps since
        # the middle plane found less than the last did: the first of them, or their
        # sum, is no reading of the last.
        late = series[STEPS // 2 + 1 :]
        assert np.all(np.diff(late) > DETECTION * tolerance), (name, late)
        assert abs(float(last[name][-1]) - expected) <= tolerance, name


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_rejected_trial_adds_nothing(backend):
    adaptive = walking_fibre(backend, RungeKutta(), scheduled=False)
    output = adaptive.run(progress="none")
    accepted = np.asarray(output["step_history/accepted"], dtype=bool)
    assert not np.all(accepted), "the adaptive run rejected no trial"
    assert isinstance(output, PlaneStore)
    outcome = output.outcome
    assert outcome is not None and outcome.execution is not None
    found = outcome.execution["boundary_monitor"]
    assert isinstance(found, dict)
    # The same accepted steps, replayed with no rejection and read at every step.
    schedule = FixedSchedule.from_history(
        output["step_history/z"], output["step_history/h"], accepted
    )
    replay = walking_fibre(backend, RungeKutta())
    replay.settings = dataclasses.replace(
        replay.settings,
        schedule=schedule,
        save_planes=np.asarray(schedule.edges[1:]),
        dz_max=schedule.max_step,
    )
    every = _record(replay)
    bound = simulation_bound(replay)
    total = float(every["raw_removed_fraction"])
    tolerance = reading_tolerance(total, bound, schedule.n_steps)
    # A rejected trial is longer than the step taken after it from the same state, so
    # it would take more than that step's removal: at least a hundred tolerances.
    assert np.all(_per_step(every) > DETECTION * tolerance)
    assert abs(float(found["raw_removed_fraction"]) - total) <= tolerance


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_dense_observation_is_refused_beside_the_split_collar(backend):
    # Today the rule holds because the combination cannot run: a plane inside a step
    # would be read from a state the split never finished. Translation refuses it on
    # both backends alike.
    simulation = walking_fibre(
        backend, RungeKutta(), scheduled=False, interpolated=True
    )
    with pytest.raises(TranslationError, match="dense observation interpolates"):
        simulation.translate()


@pytest.mark.parametrize("march", ["adams", "rk"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_probe_trial_adds_nothing(backend, march):
    # The low-storage march's probe closes its backend, so no session can follow it.
    simulation = walking_fibre(backend, MARCHES[march])
    problem, request, own = _backend(simulation)
    try:
        probe = own.timing_probe(Start.entrance(problem))
        # Each trial is the schedule's first step from the launch, finished as a kept
        # step: had it counted, the run's first reading would carry one step more.
        probe.trial()
        probe.field()
        probe.trial(float(problem.initial.z), float(request.stepper.dz_init))
        probe.field()
        after, _ = _drive(
            own, problem, _watched(simulation, request), Start.entrance(problem)
        )
    finally:
        own.close()
    _, every, bound = _runs(backend, march)
    fresh = _record(walking_fibre(backend, MARCHES[march]))
    tolerance = reading_tolerance(float(fresh["raw_removed_fraction"]), bound, STEPS)
    assert _per_step(every)[0] > DETECTION * tolerance
    assert_record_within(after, fresh, bound, STEPS, what="after a probe")


def _dirty(backend: Any, simulation: Simulation, problem: Any, request: Any) -> None:
    """Leave three accepted steps' removal undrained in a backend's sums.

    A session is driven through three steps, none of them read (the grid's interval
    reaches the fourth), and closed there.
    """
    partial, _ = _drive(
        backend,
        problem,
        _watched(simulation, request),
        Start.entrance(problem),
        advances=4,
    )
    assert partial["steps"] == 0, "the interrupted session read a step"
    assert partial["raw_removed_fraction"] == 0.0


@pytest.mark.parametrize("march", sorted(MARCHES))
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_launch_starts_the_sums_from_zero(backend, march):
    simulation = walking_fibre(backend, MARCHES[march])
    problem, request, own = _backend(simulation)
    try:
        _dirty(own, simulation, problem, request)
        after, _ = _drive(
            own, problem, _watched(simulation, request), Start.entrance(problem)
        )
    finally:
        own.close()
    _, every, bound = _runs(backend, march)
    fresh = _record(walking_fibre(backend, MARCHES[march]))
    tolerance = reading_tolerance(float(fresh["raw_removed_fraction"]), bound, STEPS)
    # Carried over, the three steps would add their removal to the first reading.
    assert float(np.sum(_per_step(every)[:3])) > DETECTION * tolerance
    assert_record_within(after, fresh, bound, STEPS, what="after an interrupted run")


@pytest.mark.parametrize("march", sorted(MARCHES))
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_resume_starts_the_sums_from_zero(backend, march):
    simulation = walking_fibre(backend, MARCHES[march])
    # The state and the monitor's record at the middle plane, as a file keeps them.
    problem, request, own = _backend(simulation)
    try:
        middle, observed = _drive(
            own,
            problem,
            _watched(simulation, request),
            Start.entrance(problem),
            planes=1,
        )
    finally:
        own.close()
    [(z, field)] = observed

    def resumed() -> Start:
        """Return the start at the middle plane, the monitor's record continued."""
        return Start(
            field=field.copy(),
            z=z,
            plane_index=1,
            monitor={"record": middle, "plane": True, "unwatched": []},
            plane_pending=False,
            resume=True,
        )

    problem, request, own = _backend(simulation)
    try:
        fresh, _ = _drive(own, problem, _watched(simulation, request, 1), resumed())
    finally:
        own.close()
    problem, request, own = _backend(simulation)
    try:
        _dirty(own, simulation, problem, request)
        after, _ = _drive(own, problem, _watched(simulation, request, 1), resumed())
    finally:
        own.close()
    _, every, bound = _runs(backend, march)
    tolerance = reading_tolerance(float(fresh["raw_removed_fraction"]), bound, STEPS)
    assert float(np.sum(_per_step(every)[:3])) > DETECTION * tolerance
    assert fresh["resumed_at"] == [z]
    assert_record_within(after, fresh, bound, STEPS, what="a resume after a run")
    if march != "adams":
        # A march with no history continues the uninterrupted run's discrete steps:
        # the resumed record adds only the steps after the resume to the record it
        # continues.
        total = float(every["raw_removed_fraction"])
        assert abs(float(fresh["raw_removed_fraction"]) - total) <= reading_tolerance(
            total, bound, STEPS
        )


@pytest.mark.skipif("native" not in available(), reason="native is not installed")
@pytest.mark.parametrize("march", ["rk", "low-storage"])
def test_native_s_accept_record_carries_each_step_s_sums(march):
    """SEP 52 section 5.2: an accepted step's finish returns the split collar's sums.

    Native's engines (``_testing.engines``) report, in each step's ``AcceptRecord``,
    what that step's split took and the power it found, measured where native measures
    them, during the split, from the masked field before it. The removals of the steps
    since the monitor was bound add up to what the next reading drains, and the
    latest step's found sums are the ones that reading judges; the stiffness is not
    formed natively.
    """
    from solprop.backends.engines import AcceptRecord  # noqa: PLC0415
    from solprop.backends.records import run_records  # noqa: PLC0415
    from solprop.boundaries import SPLIT_RATE_BUCKETS  # noqa: PLC0415

    testing = pytest.importorskip("solprop_native._core")._testing
    simulation = walking_fibre("native", MARCHES[march])
    problem, request, own = _backend(simulation)
    try:
        watched = simulation._boundary_monitor()
        assert watched is not None
        monitor = dataclasses.replace(watched, crossing=problem.split_crossing)
        assert monitor.split
        workspace = own._workspace
        field = np.ascontiguousarray(problem.initial.field0).reshape(-1)
        testing.load(workspace, field, np.empty(0, dtype=np.complex128))
        testing.configure_march(
            workspace, run_records(problem, request, Start.entrance(problem))
        )
        crossing = problem.split_crossing
        occupancy = None
        if crossing is not None:
            occupancy = {
                "buckets": np.ascontiguousarray(crossing.buckets, np.uint8).ravel(),
                "count": SPLIT_RATE_BUCKETS,
            }
        testing.bind_monitor(workspace, own._monitor_binding(monitor), None, occupancy)
        schedule = request.stepper.schedule
        assert schedule is not None
        engines = testing.engines(workspace)
        records: list[AcceptRecord] = []
        try:
            for z, dz in zip(schedule.edges[:3], schedule.steps[:3], strict=True):
                engines.rk_trial(float(z), float(dz))
                records.append(engines.rk_accept(float(dz)))
        finally:
            engines.close()
        reading = np.asarray(testing.monitor_reading(workspace)).reshape(-1)
    finally:
        own.close()
    removed = [record.removed for record in records]
    assert all(value > 0.0 for value in removed)
    assert all(math.isnan(record.stiffness) for record in records)
    # One sum of three per-step sums against the accumulator's sum of the same
    # chunks' terms, in another association: three additions' rounding.
    assert reading[5] == pytest.approx(sum(removed), rel=3 * np.finfo(float).eps)
    found = records[-1].found
    assert found is not None
    assert found.shape == (4,)
    # The same view, read twice: the latest step's found sums, exactly.
    np.testing.assert_array_equal(found, reading[6:10])
