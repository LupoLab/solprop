# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference's engines hold SEP 52 section 5.2's commit contract.

The Runge-Kutta engine keeps its trial and commits only at ``accept``; the low-storage
engine is the same protocol with a destructive trial, in flight from ``trial`` until
``accept`` returns; the Adams engine takes the retained entries oldest first and the
host source's reservation as ``begin_attempt``/``end_attempt``. The split collar's sums
are taken by the workspace's finish on the masked field before the split and travel in
the :class:`~solprop.backends.engines.AcceptRecord`, which the session accumulates.

Every comparison here repeats one computation on one build, so equality is exact by
construction (structural, SEP 25): the engines move arrays, they do not recompute them.
"""

import math
from contextlib import closing
from typing import Any, cast

import numpy as np
import pytest

from backend_corpus_cases import FIBRE, STEPS, walking_fibre
from builders import load_reference
from sessions.runs import bare, run
from solprop.backends.engines import AcceptRecord, AttemptOutcome
from solprop.backends.events import Eligibility, Outcome
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.multistep import HostHistory
from solprop.backends.session import Start
from solprop.exceptions import BackendError
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, Multistep, RungeKutta

LENGTH = 0.02


def _loaded(soliton_problem, narrow_grid, **settings):
    """Return the soliton's reference backend loaded at the entrance."""
    problem, request = soliton_problem(narrow_grid, 1, save_planes=[LENGTH], **settings)
    backend = NumpyRefBackend(problem, request, grid=narrow_grid)
    load_reference(backend)
    return backend


def test_a_runge_kutta_trial_commits_nothing_until_it_is_accepted(
    soliton_problem, narrow_grid
):
    with closing(_loaded(soliton_problem, narrow_grid)) as backend:
        workspace = backend.workspace
        engine = backend._runge_kutta(workspace)
        committed = workspace.state
        assert committed is not None
        before = committed.copy()
        engine.trial(0.0, LENGTH / 8)
        # A trial is held beside the committed state, which it leaves alone.
        assert workspace.state is committed
        assert np.array_equal(committed, before)
        assert not np.array_equal(engine.trial_state, before)
        # A rejected trial is replaced by the next; the committed state still stands.
        engine.trial(0.0, LENGTH / 16)
        assert workspace.state is committed
        engine.trial(0.0, LENGTH / 8)
        expected, _ = workspace.finish(engine.trial_state, LENGTH / 8)
        record = engine.accept(LENGTH / 8)
        assert workspace.state is not committed
        assert np.array_equal(workspace.committed(), expected)
        # Without a split collar the finish takes no sums and forms no stiffness.
        assert record.removed == 0.0
        assert record.found is None
        assert math.isnan(record.stiffness)
        assert not engine.in_flight


def test_a_runge_kutta_engine_counts_every_stage_it_evaluates(
    soliton_problem, narrow_grid
):
    with closing(_loaded(soliton_problem, narrow_grid)) as backend:
        engine = cast("Any", backend.march).engine
        stages = backend.request.stepper.tableau.n_stages
        engine.trial(0.0, LENGTH / 8)
        engine.trial(0.0, LENGTH / 16)
        assert engine.evaluations == 2 * stages
        assert len(engine.stages) == stages
        assert backend.metadata()["source_evaluations"] == 2 * stages
        # The next run's engine is its own, and counts afresh.
        fresh = cast("Any", backend.method(None).march).engine
        assert fresh.evaluations == 0
        assert backend.metadata()["source_evaluations"] == 0
        with pytest.raises(BackendError, match="take a trial"):
            _ = fresh.trial_state


def test_the_low_storage_trial_is_in_flight_until_its_accept_returns(
    soliton_problem, narrow_grid
):
    schedule = FixedSchedule.uniform(0.0, LENGTH, LENGTH / 8)
    backend = _loaded(
        soliton_problem,
        narrow_grid,
        integrator=LowStorageRungeKutta(),
        schedule=schedule,
    )
    with closing(backend):
        engine = cast("Any", backend.march).engine
        workspace = backend.workspace
        before = np.array(workspace.state)
        assert not engine.in_flight
        dz = float(schedule.steps[0])
        assert engine.trial(0.0, dz) == 0.0
        # The trial overwrote the state, the register A: neither end of the step.
        assert engine.in_flight
        assert not np.array_equal(workspace.committed(), before)
        record = engine.accept(dz)
        assert not engine.in_flight
        assert math.isfinite(record.stiffness) and record.stiffness > 0.0
        assert record.found is None


def test_a_low_storage_finish_that_fails_leaves_the_step_in_flight(
    soliton_problem, narrow_grid, monkeypatch
):
    # A failure inside the accept, after the trial overwrote the state, leaves no
    # committed state at that position: the run ends failed with nothing to read.
    schedule = FixedSchedule.uniform(0.0, LENGTH, LENGTH / 8)
    backend = _loaded(
        soliton_problem,
        narrow_grid,
        integrator=LowStorageRungeKutta(),
        schedule=schedule,
    )
    with closing(backend):

        def failing(state, dz):
            raise RuntimeError("the finish failed")

        monkeypatch.setattr(backend.workspace, "finish", failing)
        problem, request = backend.workspace.problem, backend.request
        prepared = backend.prepare(problem, request)
        session: Any = prepared.start(Start.entrance(problem), owns=False)
        session.acknowledge(session.advance().receipt)
        terminal = session.advance().terminal
        assert terminal is not None and terminal.outcome is Outcome.FAILED
        assert isinstance(terminal.error, RuntimeError)
        assert terminal.eligibility is Eligibility.POISONED
        assert session.in_flight == (0.0, float(schedule.steps[0]))
        assert backend.metadata()["source_evaluations"] == 0
        with pytest.raises(BackendError, match="abandoned in flight"):
            session.march_snapshot_nbytes()
        session.close()
        monkeypatch.undo()
        # A new session loads afresh, with nothing in flight.
        with cast("Any", prepared.start(Start.entrance(problem), owns=False)) as again:
            assert again.in_flight is None


@pytest.mark.parametrize("formulation", ["interaction", "exponential"])
def test_an_adams_attempt_reserves_then_records_and_releases(
    soliton_problem, narrow_grid, monkeypatch, formulation
):
    problem, request = soliton_problem(
        narrow_grid,
        1,
        save_planes=[LENGTH / 4],
        integrator=Multistep(formulation=formulation),
    )
    backend = NumpyRefBackend(problem, request, grid=narrow_grid)
    calls = []
    with closing(backend):
        workspace = backend.workspace
        monkeypatch.setattr(
            workspace, "reserve_calls", lambda cost: calls.append(("reserve", cost))
        )
        monkeypatch.setattr(
            workspace,
            "record_trial",
            lambda *, accepted: calls.append(("record", accepted)),
        )
        monkeypatch.setattr(
            workspace, "release_calls", lambda: calls.append(("release",))
        )
        history, _ = run(backend, problem, bare(request), export=None).joined
    # One reservation per attempt, then its decision, then the release: the startup
    # sample is charged to the first.
    attempts = [calls[index : index + 3] for index in range(0, len(calls), 3)]
    assert len(attempts) == history.n_steps
    assert attempts[0][0] == ("reserve", 2)
    for attempt, accepted in zip(attempts, history.accepted, strict=True):
        assert attempt[0][0] == "reserve"
        assert attempt[1:] == [("record", bool(accepted)), ("release",)]


def test_an_abandoned_adams_attempt_only_releases(soliton_problem, narrow_grid):
    backend = _loaded(soliton_problem, narrow_grid, integrator=Multistep())
    with closing(backend):
        engine = cast("Any", backend.march)._engine
        calls = []
        workspace = backend.workspace
        workspace.record_trial = lambda *, accepted: calls.append(("record", accepted))
        workspace.release_calls = lambda: calls.append(("release",))
        engine.end_attempt(AttemptOutcome.ABANDONED)
        engine.end_attempt(AttemptOutcome.REJECTED)
        assert calls == [("release",), ("record", False), ("release",)]


def test_an_adams_accept_takes_the_kept_entries_oldest_first(
    soliton_problem, narrow_grid, monkeypatch
):
    problem, request = soliton_problem(
        narrow_grid, 1, save_planes=[LENGTH / 4], integrator=Multistep()
    )
    backend = NumpyRefBackend(problem, request, grid=narrow_grid)
    with closing(backend):
        seen = []
        accept = HostHistory.accept

        def spy(engine, dz, free, stencil, kept):
            seen.append(([entry.z for entry in kept], [e.row for e in kept], free))
            return accept(engine, dz, free, stencil, kept)

        # The session's own engine, built for its run.
        monkeypatch.setattr(HostHistory, "accept", spy)
        run(backend, problem, bare(request), export=None)
    assert len(seen) > 2
    for positions, rows, free in seen:
        assert positions == sorted(positions)
        assert free not in rows


def _split_backend():
    """Return the walking fibre's reference backend, its workspace loaded and watched,
    and its monitor."""
    simulation = walking_fibre("numpy_ref", RungeKutta())
    problem = simulation.translate()
    request = simulation._run_request()
    backend = NumpyRefBackend(problem, request, grid=simulation.grid)
    load_reference(backend)
    monitor = simulation._boundary_monitor()
    assert monitor is not None and monitor.split
    # As a session's attached monitor: the crossing limit read, the workspace bound.
    monitor.crossing = problem.split_crossing
    backend.workspace.bind_monitor(monitor)
    return backend, monitor


@pytest.mark.medium
def test_the_finish_takes_the_split_sums_on_the_masked_field_before_the_split():
    backend, monitor = _split_backend()
    with closing(backend):
        workspace = backend.workspace
        dz = FIBRE / STEPS
        engine = backend._runge_kutta(workspace)
        engine.trial(0.0, dz)
        raw = engine.trial_state
        finished, record = workspace.finish(raw, dz)
        sums = monitor.split_sums(workspace.mask(raw), dz)
        assert record.removed == float(sums[0])
        assert record.found is not None
        assert np.array_equal(record.found, sums[1:])
        # The same reduction of the finished, damped field is another number: the
        # measurement point is the field before the split (SEP 52 section 4).
        after = monitor.split_sums(finished, dz)
        assert record.removed > 0.0
        assert abs(float(after[0]) - record.removed) > 1e-3 * record.removed


@pytest.mark.medium
def test_the_session_adds_the_removal_and_keeps_the_latest_occupancies():
    backend, monitor = _split_backend()
    with closing(backend):
        problem, request = backend.workspace.problem, backend.request
        prepared = backend.prepare(problem, request)
        session = cast("Any", prepared.start(Start.entrance(problem), owns=False))
        # The monitor bound as the session's attachment binds it.
        session._bind_monitor(monitor)
        first = np.array([1.0, 0.5, 0.25, 0.125])
        latest = np.array([2.0, 1.0, 0.5, 0.25])
        session.absorb(AcceptRecord(removed=1.5, found=first))
        session.absorb(AcceptRecord(removed=0.25, found=latest))
        # A step that took no sums (no split, or no monitor) leaves both alone.
        session.absorb(AcceptRecord())
        assert session._split_removed == 1.75
        assert session._split_found is latest
        # The next reading drains the removal and judges the latest occupancies.
        reading = session._energies()
        assert reading.removed == 1.75
        assert reading.found is latest
        assert session._split_removed == 0.0
        # Attaching a monitor starts the accumulator from zero.
        session.absorb(AcceptRecord(removed=3.0, found=first))
        session._bind_monitor(monitor)
        assert session._split_removed == 0.0
        assert session._split_found is None
        session.close()
        # Loading a state, as every session does, starts it empty.
        with cast("Any", prepared.start(Start.entrance(problem), owns=False)) as again:
            assert again._split_removed == 0.0
            assert again._split_found is None
