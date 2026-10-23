# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The shared step loop, driven through a stage evaluation that is not a backend.

:class:`solprop.backends.driver.StepDriver` is the marching half of a backend with the
kernels left abstract, so it can be tested with a stage evaluation whose answers are
scripted. That is the point: every claim here is about the *step control* -- landing,
acceptance, the controller's state, the history, the noise models, the guards --
and none of them depends on any physics being evaluated correctly, which is what makes
the same assertions true of every backend that inherits it.
"""

import dataclasses
import inspect
import io
import logging

import numpy as np
import pytest

from bounds import assert_within
from solprop.backends.driver import (
    INITIAL_STEP_STATS_CAPACITY,
    StepDriver,
    reached,
)
from solprop.backends.protocol import BatchBackend, StepStats
from solprop.exceptions import BackendError
from solprop.log import BarReporter, LogReporter, NullReporter
from solprop.noise import input_noise, modified_noise
from solprop.stats import StatsSpec
from solprop.stepper import PIController
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.linop import LinearOperator, delta_phi
from solprop.translate.problem import SCHEMA_VERSION
from solprop.translate.spatial import SeparableOperator
from solprop.translate.stage import StageScalars, StageTerms, scalars

LENGTH = 0.02


def present(value):
    """Return an optional array, having asserted that it is there.

    The state and the noise are ``None`` before ``set_initial`` and after ``close``, so
    the seam types them optional; a test that has just loaded one says so here rather
    than asserting around it at every use.
    """
    assert value is not None
    return value


class ScriptedDriver(StepDriver):
    """A driver whose stage evaluation multiplies by a constant and scores from a list.

    The state advances by a factor per step so that a rejected step can be seen not to
    have been kept, and the error norms come from a script so that acceptance,
    rejection and the controller's response are exercised without needing a problem
    whose physics happens to produce them.
    """

    name = "scripted"

    def __init__(self, problem, *, norms=(), factor=1.0, scale=LENGTH / 8, **kwargs):
        super().__init__(problem, **kwargs)
        self._norms = list(norms)
        self._factor = factor
        self._scale = scale
        self.calls: list[tuple[float, float]] = []
        self.proposals: list[float] = []
        self.accepted_hooks = 0
        self.planes_begun = 0

    def _trial_step(self, z, dz):
        self.calls.append((z, dz))
        self.proposals.append(self._proposal)
        # A fifth-order embedded estimate: the norm has to fall like dz^5 or the
        # controller's equivalent_norm reads every short landing step as a failure and
        # the proposal collapses. A constant would test the loop against a stand-in the
        # loop was never written for.
        norm = self._norms.pop(0) if self._norms else (dz / self._scale) ** 5
        state = self._require_state() * self._factor
        return norm, self._apply_mask(state)

    def _apply_mask(self, state):
        return state * self._problem.grids.active_mask

    def _on_accepted_step(self):
        self.accepted_hooks += 1

    def stats(self, spec) -> dict[str, np.ndarray]:
        """Return one scriptable number per name: the position the row was taken at."""
        state = self._require_state()
        return {name: np.full((state.shape[0],), self._z) for name in spec.names}

    def _begin_plane(self):
        self.planes_begun += 1

    def close(self):
        """Release the state, as the seam requires."""
        self._state = None
        self._closed = True


@pytest.fixture
def driver(soliton_problem, narrow_grid):
    """Return ``(**kwargs) -> ScriptedDriver`` loaded at the entrance."""

    def build(*, planes=(LENGTH,), field=None, noise=None, **kwargs):
        problem = soliton_problem(narrow_grid, 1, save_planes=list(planes))
        made = ScriptedDriver(problem, grid=narrow_grid, **kwargs)
        made.set_initial(problem.initial.field0 if field is None else field, noise)
        return made

    return build


def test_a_march_lands_exactly_on_its_save_plane(driver):
    """The position is the plane itself, not an accumulation of steps that nears it."""
    made = driver()
    summary = made.propagate_to(LENGTH)
    assert made._z == LENGTH
    assert summary.z == LENGTH
    assert summary.n_steps > 1
    assert summary.n_rejected == 0
    assert made.planes_begun == 1
    assert made.accepted_hooks == summary.n_steps


def test_the_landing_guard_admits_the_target_and_nothing_short_of_it():
    """``reached`` is the eight-ulp guard the schedule's own landing uses."""
    assert reached(1.0, 1.0)
    assert reached(1.0 - 4 * np.spacing(1.0), 1.0)
    assert not reached(1.0 - 1e-12, 1.0)


def test_a_rejected_step_is_recorded_and_not_kept(driver):
    """The state after a rejection is the state before it, and the history says so."""
    # 5.0 is far above the controller's threshold of one, so the first step is refused
    # and the second, from the same position, is taken.
    made = driver(norms=[5.0, 0.1], factor=2.0)
    before = present(made._state).copy()
    made.propagate_to(LENGTH)
    observed = made.observe()
    assert not bool(observed.step_history.accepted[0])
    assert bool(observed.step_history.accepted[1])
    assert made.calls[0][0] == made.calls[1][0] == 0.0
    # The refused step's shorter successor is what the controller asked for next.
    assert made.calls[1][1] < made.calls[0][1]
    # The scripted step doubles the state, so the division by powers of two is exact.
    assert np.array_equal(
        before,
        np.asarray(observed.field) / 2.0 ** (int(observed.step_history.accepted.sum())),
    )
    assert made.accepted_hooks == int(observed.step_history.accepted.sum())


def test_a_rejected_step_is_retried_at_the_rejection_factor_alone(
    soliton_problem, narrow_grid
):
    """A refused step is retried shorter by the rejection factor, with no growth on top.

    The controller's growth factor belongs to the accepted history and is applied once
    per accepted step. Applied again after a rejection it undid the shrink: with a
    growth cap of two, a step whose norm sat just above one was refused unchanged
    forever.
    """
    controller = PIController(rtol=1e-10, atol=1e-18, max_factor=2.0)
    problem = soliton_problem(
        narrow_grid, 1, save_planes=[LENGTH], controller=controller
    )
    # Three easy steps fill the lag and ask for growth; the fourth fails narrowly.
    made = ScriptedDriver(problem, grid=narrow_grid, norms=[0.01, 0.01, 0.01, 1.0001])
    made.set_initial(problem.initial.field0, None)
    made.propagate_to(LENGTH)
    accepted = made.observe().step_history.accepted
    assert [bool(flag) for flag in accepted[:4]] == [True, True, True, False]
    assert controller.propose([0.01, 0.01, 0.01]) > 1.1
    rejected, retried = made.calls[3][1], made.calls[4][1]
    assert retried == pytest.approx(rejected * controller.rejection_factor(1.0001))


def test_the_history_covers_only_the_steps_since_the_last_observation(driver):
    """``observe`` empties the history, as the protocol promises."""
    made = driver(planes=(LENGTH / 2, LENGTH))
    made.propagate_to(LENGTH / 2)
    first = made.observe().step_history
    made.propagate_to(LENGTH)
    second = made.observe().step_history
    assert first.n_steps > 0
    assert second.n_steps > 0
    assert float(first.z[-1]) == LENGTH / 2
    assert float(second.z[0]) > LENGTH / 2


def test_a_replayed_schedule_takes_its_steps_as_given(
    driver, soliton_problem, narrow_grid
):
    """A replay never rejects and reads its position off the schedule's edges."""
    schedule = FixedSchedule.uniform(0.0, LENGTH, LENGTH / 8)
    problem = soliton_problem(narrow_grid, 1, save_planes=[LENGTH], schedule=schedule)
    made = ScriptedDriver(problem, grid=narrow_grid, norms=[5.0] * 8)
    made.set_initial(problem.initial.field0, None)
    summary = made.propagate_to(LENGTH)
    assert summary.n_steps == 8
    assert summary.n_rejected == 0
    assert made._z == LENGTH
    # Every step is the schedule's, not the controller's.
    assert [dz for _, dz in made.calls] == pytest.approx([LENGTH / 8] * 8)
    assert not summary.met_tolerance


def test_a_replay_that_runs_out_says_so(soliton_problem, narrow_grid):
    """A schedule too short for the target is an error, not a quiet stop."""
    full = soliton_problem(
        narrow_grid,
        1,
        save_planes=[LENGTH],
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / 8),
    )
    # The operator table still covers the whole run, so the guard that fires is the
    # schedule's own and not the table's.
    problem = dataclasses.replace(
        full,
        stepper=dataclasses.replace(
            full.stepper, schedule=FixedSchedule.uniform(0.0, LENGTH / 2, LENGTH / 8)
        ),
    )
    made = ScriptedDriver(problem, grid=narrow_grid)
    made.set_initial(problem.initial.field0, None)
    made.propagate_to(LENGTH / 2)
    with pytest.raises(BackendError, match="cannot reach"):
        made.propagate_to(LENGTH)


def test_a_schedule_of_equal_steps_lands_on_a_plane_its_running_sum_misses(
    soliton_problem, narrow_grid
):
    """A running sum of rounded steps misses its plane by more than eight ulps.

    ``FixedSchedule(np.full(400, L / 400))`` puts its middle edge 19 ulps short of
    ``L / 2`` (and its end 17 past ``L``): the drift of a sum of ``n`` rounded steps,
    which the replay's landing slack scales with. The run reaches the plane, stops on
    the edge the plane belongs to, and resumes there.
    """
    steps = np.full(400, LENGTH / 400)
    schedule = FixedSchedule(steps)
    middle = float(schedule.edges[200])
    assert LENGTH / 2 - middle > 8 * float(np.spacing(LENGTH / 2))
    problem = soliton_problem(
        narrow_grid, 1, save_planes=[LENGTH / 2, LENGTH], schedule=schedule
    )
    made = ScriptedDriver(problem, grid=narrow_grid)
    made.set_initial(problem.initial.field0, None)
    made.propagate_to(LENGTH / 2)
    assert made._schedule_index == 200
    assert made._z == middle
    made.propagate_to(LENGTH)
    assert made._schedule_index == 400
    resumed = ScriptedDriver(problem, grid=narrow_grid)
    resumed.set_initial(problem.initial.field0, None, z=LENGTH / 2)
    assert resumed._schedule_index == 200


def test_a_resume_continues_a_replay_from_the_edge_it_stopped_at(
    soliton_problem, narrow_grid
):
    """``set_initial(z=)`` finds the schedule index the position belongs to."""
    schedule = FixedSchedule.uniform(0.0, LENGTH, LENGTH / 8)
    problem = soliton_problem(
        narrow_grid, 1, save_planes=[LENGTH / 2, LENGTH], schedule=schedule
    )
    made = ScriptedDriver(problem, grid=narrow_grid)
    made.set_initial(problem.initial.field0, None, z=LENGTH / 2)
    assert made._schedule_index == 4
    made.propagate_to(LENGTH)
    assert len(made.calls) == 4


def test_a_resume_off_the_schedule_s_edges_is_refused(soliton_problem, narrow_grid):
    """A position the run never stopped at cannot continue a replay in step."""
    schedule = FixedSchedule.uniform(0.0, LENGTH, LENGTH / 8)
    problem = soliton_problem(narrow_grid, 1, save_planes=[LENGTH], schedule=schedule)
    made = ScriptedDriver(problem, grid=narrow_grid)
    with pytest.raises(BackendError, match="one of its edges"):
        made.set_initial(problem.initial.field0, None, z=LENGTH / 3)


def test_the_input_noise_model_adds_its_realisation_once(
    driver, soliton_problem, narrow_grid
):
    """Added at the entrance and never again, so a resume is not given a second draw."""
    problem = soliton_problem(
        narrow_grid,
        1,
        save_planes=[LENGTH],
        noise=input_noise(narrow_grid, np.random.default_rng(7)),
    )
    entrance = ScriptedDriver(problem, grid=narrow_grid)
    entrance.set_initial(problem.initial.field0, None)
    resumed = ScriptedDriver(problem, grid=narrow_grid)
    resumed.set_initial(problem.initial.field0, None, z=LENGTH / 2)
    masked = problem.initial.field0 * problem.grids.active_mask
    assert not np.array_equal(present(entrance._state), masked)
    assert np.array_equal(present(resumed._state), masked)
    assert entrance._noise is None


def test_the_modified_noise_model_keeps_its_realisation_apart(
    soliton_problem, narrow_grid
):
    """The propagating field never carries it; it is advanced by the unitary phase."""
    problem = soliton_problem(
        narrow_grid,
        1,
        save_planes=[LENGTH],
        noise=modified_noise(narrow_grid, np.random.default_rng(7)),
    )
    made = ScriptedDriver(problem, grid=narrow_grid)
    made.set_initial(problem.initial.field0, None)
    assert np.array_equal(
        present(made._state), problem.initial.field0 * problem.grids.active_mask
    )
    drawn = present(made._noise)
    advanced = present(made._noise_at(made._stage_scalars(0.0, LENGTH / 2)))
    # Dispersion yes, loss and gain no: the magnitude is untouched by the advance.
    assert np.allclose(np.abs(advanced), np.abs(drawn))
    assert not np.array_equal(advanced, drawn)
    assert present(made._noise_phase()).shape == problem.shape


def test_a_given_realisation_replaces_the_contract_s(soliton_problem, narrow_grid):
    """The realisation given to ``set_initial`` wins over the assembled one."""
    problem = soliton_problem(
        narrow_grid,
        1,
        save_planes=[LENGTH],
        noise=modified_noise(narrow_grid, np.random.default_rng(7)),
    )
    made = ScriptedDriver(problem, grid=narrow_grid)
    made.set_initial(
        problem.initial.field0,
        modified_noise(narrow_grid, np.random.default_rng(11)),
    )
    assert not np.array_equal(present(made._noise), problem.initial.noise0)


def test_a_run_without_noise_reports_no_phase(driver, narrow_grid):
    """``noise_model = "none"`` has neither a realisation nor a phase to report."""
    made = driver()
    assert (
        made._noise_field(modified_noise(narrow_grid, np.random.default_rng(3))) is None
    )
    assert made._noise_phase() is None
    assert made._noise_at(made._stage_scalars(0.0, 0.0)) is None


def test_the_guards_refuse_what_cannot_be_done(driver, soliton_problem, narrow_grid):
    """Closed, unloaded, backwards, off the table, and the wrong shape."""
    problem = soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    empty = ScriptedDriver(problem, grid=narrow_grid)
    with pytest.raises(BackendError, match="no initial state"):
        empty.propagate_to(LENGTH)
    with pytest.raises(BackendError, match="expected"):
        empty.set_initial(np.zeros((1, 1, 1, 3), dtype=complex), None)
    with pytest.raises(BackendError, match="inside the operator table"):
        empty.set_initial(problem.initial.field0, None, z=2 * LENGTH)

    made = driver()
    made.propagate_to(LENGTH)
    with pytest.raises(BackendError, match="backwards"):
        made.propagate_to(0.0)
    with pytest.raises(BackendError, match="must be finite"):
        made.propagate_to(float("nan"))
    with pytest.raises(BackendError, match="beyond the operator table"):
        made.propagate_to(1e3)

    made.close()
    with pytest.raises(BackendError, match="closed"):
        made.observe()
    with pytest.raises(BackendError, match="closed"):
        made.set_initial(problem.initial.field0, None)


def test_statistics_without_a_grid_say_which_backend_to_give_one_to(
    soliton_problem, narrow_grid
):
    """The message names the backend, so it is actionable for whichever one it is."""
    problem = soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    made = ScriptedDriver(problem)
    with pytest.raises(BackendError, match="'scripted'"):
        made._require_grid()
    assert ScriptedDriver(problem, grid=narrow_grid)._require_grid() is narrow_grid


# --------------------------------------------------------------- progress reporting


class Events(NullReporter):
    """Record every progress call in the order it arrived."""

    def __init__(self):
        self.order = []
        self.steps = []
        self.planes = []
        self.extent = None

    def started(self, z_start, z_end, n_planes):
        self.order.append("started")
        self.extent = (z_start, z_end, n_planes)

    def step(self, z, dz, accepted):
        self.order.append("step")
        self.steps.append((z, dz, accepted))

    def plane(self, z, summary):
        self.order.append("plane")
        self.planes.append((z, summary))

    def finished(self, total):
        self.order.append("finished")


def test_a_reporter_sees_every_attempted_step_including_the_rejected_ones(driver):
    """`_record` is the one funnel every attempt goes through, so nothing is missed."""
    reporter = Events()
    made = driver(norms=[5.0, 0.1], factor=2.0)
    made.set_progress(reporter)
    summary = made.propagate_to(LENGTH)
    made._progress.plane(LENGTH, summary)

    assert reporter.order[0] == "started"
    assert reporter.extent == (0.0, LENGTH, 1)
    assert reporter.order.count("step") == summary.n_steps
    assert sum(1 for _, _, accepted in reporter.steps if not accepted) == 1
    assert summary.n_rejected == 1
    # The rejected attempt is reported at the position it would have reached, which the
    # retry then reaches more slowly: the reporter, not the driver, is what has to know.
    rejected_at = next(z for z, _, accepted in reporter.steps if not accepted)
    assert rejected_at > reporter.steps[1][0]


def test_started_is_announced_once_per_load_and_a_resume_announces_it_again(driver):
    reporter = Events()
    made = driver(planes=(LENGTH / 2, LENGTH))
    made.set_progress(reporter)
    made.propagate_to(LENGTH / 2)
    made.propagate_to(LENGTH)
    assert reporter.order.count("started") == 1
    assert reporter.extent == (0.0, LENGTH, 2)

    made.set_initial(present(made._state), None, z=LENGTH / 2)
    made.propagate_to(LENGTH)
    assert reporter.order.count("started") == 2
    assert reporter.extent == (LENGTH / 2, LENGTH, 2)


def test_the_reporter_is_the_backend_s_and_the_default_is_silent(driver):
    """Nothing is reported until one is attached, so a bare backend costs nothing."""
    made = driver()
    assert isinstance(made._progress, NullReporter)
    made.propagate_to(LENGTH)


@pytest.mark.parametrize("period", [0.0, 1e9])
def test_a_run_with_a_reporter_is_the_run_without(
    soliton_problem, narrow_grid, backend_class, period
):
    """Layer D throughout: the calls are observers, made after the step is decided.

    ``wall_time`` is excluded because it is a clock reading rather than a result -- two
    runs of the same problem differ in it whether or not a reporter is attached. The
    other four history columns and the exit field are the whole of what the step loop
    computed. Two runs of one problem are one computation repeated, on a fixed
    schedule so that they take the same steps: the schedule and the acceptances are
    exact, the field is held to the eight-step ladder's ``8 x 16 eps`` and the error
    norm, a defect over ``rtol``, to seven stages of that over ``rtol``.
    """
    columns = ("z", "dz", "accepted")
    bound = 8 * 16 * np.finfo(float).eps
    schedule = FixedSchedule.uniform(0.0, LENGTH, LENGTH / 8)

    def run(reporter):
        problem = soliton_problem(
            narrow_grid, 1, save_planes=[LENGTH / 2, LENGTH], schedule=schedule
        )
        made = backend_class(problem, grid=narrow_grid)
        made.set_progress(reporter)
        made.set_initial(problem.initial.field0, None)
        fields, histories = [], []
        for plane in (LENGTH / 2, LENGTH):
            made.propagate_to(plane)
            observed = made.observe()
            fields.append(observed.field)
            histories.append(observed.step_history)
        made.close()
        return fields, histories

    quiet_fields, quiet_history = run(NullReporter())
    rtol = soliton_problem(narrow_grid, 1, save_planes=[LENGTH]).stepper.controller.rtol
    reporters = [
        LogReporter(
            period=period, clock=lambda: 0.0, logger=logging.getLogger("quiet")
        ),
        # A bar redrawing on every step, drawing on a string.
        BarReporter(period=1e-9, file=io.StringIO(), ncols=60, ascii=True),
    ]
    for reporter in reporters:
        loud_fields, loud_history = run(reporter)
        for quiet, loud in zip(quiet_fields, loud_fields, strict=True):
            assert_within(loud, quiet, bound, what="field")
        for quiet, loud in zip(quiet_history, loud_history, strict=True):
            assert quiet.n_steps == loud.n_steps > 0
            for name in columns:
                assert np.array_equal(getattr(quiet, name), getattr(loud, name)), name
            np.testing.assert_allclose(
                loud.error_norm, quiet.error_norm, rtol=0, atol=7 * bound / rtol
            )


def test_both_backends_report_the_same_calls_in_the_same_order(
    soliton_problem, narrow_grid, backend_class
):
    reporter = Events()
    planes = [LENGTH / 2, LENGTH]
    problem = soliton_problem(narrow_grid, 1, save_planes=planes)
    made = backend_class(problem, grid=narrow_grid)
    made.set_progress(reporter)
    made.set_initial(problem.initial.field0, None)
    summaries = []
    for plane in planes:
        summary = made.propagate_to(plane)
        summaries.append(summary)
        reporter.plane(plane, summary)
    reporter.finished(1.0)
    made.close()

    assert reporter.order[0] == "started"
    assert reporter.order[-1] == "finished"
    assert reporter.order.count("started") == 1
    assert reporter.order.count("finished") == 1
    assert reporter.order.count("plane") == len(planes)
    assert reporter.order.count("step") == sum(one.n_steps for one in summaries)
    # One `started`, then steps and planes interleaved with a plane last before the end.
    assert reporter.order[-2] == "plane"


# -------------------------------------------- the statistics channel between planes


def periodic(problem, names=("energy",), *, period=1):
    """Return `problem` with a periodic statistics subset of `names`."""
    spec = StatsSpec(names=names, periodic=names, period=period)
    return dataclasses.replace(problem, stats_spec=spec)


def rows_of(made):
    """Propagate to the one plane and return the channel it flushed."""
    made.propagate_to(LENGTH)
    return made.observe().step_stats


def test_a_row_is_taken_at_the_position_the_accepted_step_reached(
    soliton_problem, narrow_grid
):
    """The trap this test exists for, and the reason the call is not in a hook.

    ``_march_to`` records an attempt *before* it finishes the state and advances ``z``;
    ``_replay_to`` records it *after* both. So a channel evaluated from inside
    ``_on_accepted_step`` reads the previous step's position and state on an adaptive
    run and the current step's on a replayed one -- and a cost gate is naturally
    specified on a replayed schedule, where the wrong version passes. Both paths are
    asserted here, against the same expectation.
    """
    schedule = FixedSchedule.uniform(0.0, LENGTH, LENGTH / 8)
    for extra in ({}, {"schedule": schedule}):
        problem = periodic(
            soliton_problem(narrow_grid, 1, save_planes=[LENGTH], **extra)
        )
        made = ScriptedDriver(problem, grid=narrow_grid)
        made.set_initial(problem.initial.field0, None)
        rows = rows_of(made)
        # `ScriptedDriver.stats` returns `self._z`, so the recorded value is the
        # position the driver held when the row was taken.
        assert rows.n_rows > 0
        np.testing.assert_array_equal(rows.values["energy"][:, 0], rows.z)
        # And that position is the end of an accepted step, never the start.
        assert rows.z[-1] == pytest.approx(LENGTH)
        assert rows.z[0] > 0.0


def test_the_channel_is_empty_unless_the_run_asks_for_one(driver):
    """The default spec has no periodic subset, so nothing is buffered or flushed."""
    made = driver()
    rows = rows_of(made)
    assert rows.n_rows == 0
    assert rows.names == ()


def test_a_row_indexes_the_attempt_it_was_taken_after(soliton_problem, narrow_grid):
    """The index is into the plane's own history, which counts rejected attempts."""
    problem = periodic(soliton_problem(narrow_grid, 1, save_planes=[LENGTH]))
    made = ScriptedDriver(problem, grid=narrow_grid, norms=[5.0, 0.1, 0.1])
    made.set_initial(problem.initial.field0, None)
    made.propagate_to(LENGTH)
    observed = made.observe()
    rows, history = observed.step_stats, observed.step_history
    assert history.n_rejected == 1
    # The first attempt was rejected, so the first row indexes attempt 1, not 0.
    assert rows.step[0] == 1
    accepted = np.flatnonzero(history.accepted)
    np.testing.assert_array_equal(rows.step, accepted)
    np.testing.assert_allclose(rows.z, history.z[accepted])


def test_the_period_counts_accepted_steps_and_a_rejection_is_not_one(
    soliton_problem, narrow_grid
):
    """A rejected attempt neither advances the phase nor records a row."""
    problem = periodic(soliton_problem(narrow_grid, 1, save_planes=[LENGTH]), period=2)
    made = ScriptedDriver(problem, grid=narrow_grid, norms=[5.0, 0.1, 0.1, 0.1, 0.1])
    made.set_initial(problem.initial.field0, None)
    made.propagate_to(LENGTH)
    observed = made.observe()
    accepted = np.flatnonzero(observed.step_history.accepted)
    # Every second *accepted* attempt, so the rejection shifts nothing.
    np.testing.assert_array_equal(observed.step_stats.step, accepted[1::2])


def test_the_rows_of_one_plane_are_flushed_with_it(soliton_problem, narrow_grid):
    """`observe` drains the buffer, so a plane's rows are its own."""
    planes = [LENGTH / 2, LENGTH]
    problem = periodic(soliton_problem(narrow_grid, 1, save_planes=planes))
    made = ScriptedDriver(problem, grid=narrow_grid)
    made.set_initial(problem.initial.field0, None)
    seen = []
    for plane in planes:
        made.propagate_to(plane)
        observed = made.observe()
        seen.append(observed.step_stats)
        # Each plane's rows index that plane's own history, from zero.
        assert observed.step_stats.step[0] < observed.step_history.n_steps
    assert all(one.n_rows for one in seen)
    assert seen[0].z[-1] == pytest.approx(LENGTH / 2)
    assert seen[1].z[0] > LENGTH / 2


def test_the_buffer_grows_across_a_plane_with_more_steps_than_its_capacity(
    soliton_problem, narrow_grid
):
    """More rows than `INITIAL_STEP_STATS_CAPACITY`, so the doubling is exercised."""
    n_steps = 3 * INITIAL_STEP_STATS_CAPACITY
    schedule = FixedSchedule.uniform(0.0, LENGTH, LENGTH / n_steps)
    problem = periodic(
        soliton_problem(narrow_grid, 1, save_planes=[LENGTH], schedule=schedule)
    )
    made = ScriptedDriver(problem, grid=narrow_grid)
    made.set_initial(problem.initial.field0, None)
    rows = rows_of(made)
    assert rows.n_rows == n_steps
    assert np.all(np.diff(rows.z) > 0.0)
    np.testing.assert_array_equal(rows.step, np.arange(n_steps))


def test_a_load_starts_the_period_again(soliton_problem, narrow_grid):
    """`set_initial` resets the phase, so a resume is not the uninterrupted run."""
    problem = periodic(soliton_problem(narrow_grid, 1, save_planes=[LENGTH]), period=3)
    made = ScriptedDriver(problem, grid=narrow_grid)
    made.set_initial(problem.initial.field0, None)
    assert made._accepted_steps == 0
    made.propagate_to(LENGTH)
    assert made._accepted_steps > 0
    made.set_initial(problem.initial.field0, None)
    assert made._accepted_steps == 0
    assert made.observe().step_stats.n_rows == 0


@pytest.mark.parametrize("replay", [False, True])
def test_alternate_statistics_storage_hooks_keep_shared_acceptance_and_period(
    soliton_problem, narrow_grid, replay
):
    class DeferredRows(ScriptedDriver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pending = []
            self.resets = []
            self.drains = 0

        def stats(self, spec):
            raise AssertionError("a periodic device row must not call host stats")

        def _reset_step_stats(self, *, release):
            self.pending.clear()
            self.resets.append(release)

        def _record_step_stats(self, spec, z, step):
            assert spec.names == ("energy",)
            assert spec.periodic == ()
            assert z == self._z
            self.pending.append((z, step))

        def _drain_step_stats(self):
            self.drains += 1
            if not self.pending:
                return StepStats.empty()
            positions = np.array([z for z, _ in self.pending])
            indices = np.array([step for _, step in self.pending])
            self.pending.clear()
            return StepStats(positions, indices, {"energy": positions[:, None]})

    extra = {"schedule": FixedSchedule.uniform(0, LENGTH, LENGTH / 8)} if replay else {}
    problem = periodic(
        soliton_problem(narrow_grid, 1, save_planes=[LENGTH / 2, LENGTH], **extra),
        period=3,
    )
    norms = () if replay else (5.0, 0.1)
    deferred = DeferredRows(problem, grid=narrow_grid, norms=norms)
    default = ScriptedDriver(problem, grid=narrow_grid, norms=norms)
    for owner in (deferred, default):
        owner.set_initial(problem.initial.field0, None)
    for plane in (LENGTH / 2, LENGTH):
        for owner in (deferred, default):
            owner.propagate_to(plane)
        actual, expected = deferred.observe(), default.observe()
        for name in ("z", "step"):
            np.testing.assert_array_equal(
                getattr(actual.step_stats, name), getattr(expected.step_stats, name)
            )
        for name in expected.step_stats.names:
            np.testing.assert_array_equal(
                actual.step_stats.values[name], expected.step_stats.values[name]
            )
        np.testing.assert_array_equal(actual.field, expected.field)
        np.testing.assert_array_equal(
            actual.step_history.accepted, expected.step_history.accepted
        )
    assert deferred.resets == [True, False, False]
    assert deferred.drains == 2
    assert deferred.observe().step_stats.n_rows == 0
    deferred.set_initial(problem.initial.field0, None, z=LENGTH / 2)
    assert deferred.resets[-1] is True
    assert not deferred.pending
    deferred.close()
    default.close()


def test_a_statistic_that_changes_shape_within_a_plane_is_refused(
    soliton_problem, narrow_grid
):
    """The buffer is laid out from the first row and indexes by that shape.

    Nothing the package builds can do this -- the spec is fixed when the backend is
    built -- but a backend whose `stats` disagreed with itself would otherwise be
    caught by a broadcast error deep in numpy rather than by a message naming the
    statistic.
    """

    class Widening(ScriptedDriver):
        def stats(self, spec) -> dict[str, np.ndarray]:
            width = 1 + len(self._history) // 2
            return {name: np.zeros((1, width)) for name in spec.names}

    problem = periodic(soliton_problem(narrow_grid, 1, save_planes=[LENGTH]))
    made = Widening(problem, grid=narrow_grid)
    made.set_initial(problem.initial.field0, None)
    with pytest.raises(BackendError, match="came back with shape"):
        made.propagate_to(LENGTH)


# ----------------------------------------------------------- the executed norm


def test_the_history_records_the_executed_norm_and_the_controller_the_equivalent(
    driver,
):
    """A landing step clipped short is written up at the norm it scored.

    Were the history to carry the controller's *equivalent* norm -- what the proposal
    would have scored, ``norm * (h_proposed / h_taken) ** 5`` -- the closing tolerance
    ratio of a run whose last steps were forced short would read as a failure; the full
    M2 case would close at 6.5e4. The controller still learns from the equivalent, so a
    plane does not restart it from a sliver; the history and everything read off it
    carry the norm each step actually committed.
    """
    made = driver()
    made.propagate_to(LENGTH)
    history = made.observe().step_history
    accepted = np.flatnonzero(history.accepted)
    last = accepted[-1]
    taken = float(history.dz[last])
    proposal = made.proposals[last]
    # The last accepted step is the landing on the plane, and it is not the proposal.
    assert taken != proposal
    executed = (taken / made._scale) ** 5
    assert float(history.error_norm[last]) == pytest.approx(executed)
    assert made._accepted_norms[-1] == pytest.approx(executed * (proposal / taken) ** 5)
    # And no accepted step of an adaptive run scores above one, by construction.
    assert np.all(history.error_norm[accepted] <= 1.0)


# --- the seam a compiled backend subclasses ----------------------------------------

FIELD = "jaxtyping.Complex[ndarray, 'n_batch n_modes n_pol n_omega']"

DRIVER_SURFACE = {
    "__init__": (
        "(self, problem: solprop.translate.problem.KernelProblem, *, "
        "grid: solprop.grid.TimeGrid | None = None, "
        "modes: collections.abc.Sequence[solprop.modes.protocol.Mode] | None = None, "
        "profiles: collections.abc.Mapping[str, solprop.profiles.ZProfile] | None = "
        "None"
        ") -> None"
    ),
    "_trial_step": "(self, z: float, dz: float) -> tuple[float, _StateT]",
    "_delta_phi": (
        "(self, bundle: solprop.translate.stage.StageScalars) -> "
        "jaxtyping.Complex[ndarray, 'n_batch n_modes n_pol n_omega']"
    ),
    "_unitary_phase": (
        "(self, bundle: solprop.translate.stage.StageScalars) -> "
        "jaxtyping.Float[ndarray, 'n_batch n_modes n_pol n_omega']"
    ),
    "_prefactor": (
        "(self, bundle: solprop.translate.stage.StageScalars) -> "
        "jaxtyping.Complex[ndarray, 'n_modes n_pol n_omega']"
    ),
    "_apply_mask": (
        "(self, state: jaxtyping.Complex[ndarray, 'n_batch n_modes n_pol n_omega'])"
        " -> _StateT"
    ),
    "_finish": "(self, state: _StateT, dz: float) -> _StateT",
    "_observed_field": (
        "(self, state: _StateT)"
        " -> jaxtyping.Complex[ndarray, 'n_batch n_modes n_pol n_omega']"
    ),
    "_reserve_source_calls": "(self, count: int) -> None",
    "_release_source_calls": "(self) -> None",
    "_adams_source": (
        f"(self, lab: {FIELD}, anchor: float, position: float, *, "
        "purpose: Literal['startup', 'predictor', 'corrected', 'diagnostic'] "
        "= 'predictor') -> "
        f"tuple[{FIELD}, {FIELD}]"
    ),
    "_adams_lab_source": (
        f"(self, lab: {FIELD}, position: float, *, "
        "purpose: Literal['startup', 'predictor', 'corrected', 'diagnostic'] "
        f"= 'predictor') -> {FIELD}"
    ),
    "_adams_block": "(self, z: float, dz: float) -> object",
    "_adams_engine": (
        "(self, method: solprop.stepper.multistep.Multistep, "
        "compensation: Literal['transported', 'untransported']) -> object"
    ),
    "_low_storage_engine": (
        "(self, method: solprop.stepper.low_storage.LowStorageRungeKutta) -> object"
    ),
    "_adams_propagator": f"(self, anchor: float, position: float) -> {FIELD}",
    "_adams_state": f"(self) -> {FIELD}",
    "_adams_adopt": f"(self, advanced: {FIELD}, dz: float) -> {FIELD}",
    "_adams_restore": f"(self, z: float, state: {FIELD}) -> None",
    "_dense_stages": "(self) -> list[numpy.ndarray]",
    "_begin_plane": "(self) -> None",
    "_on_accepted_step": "(self) -> None",
    "_reset_step_stats": "(self, *, release: bool) -> None",
    "_record_step_stats": (
        "(self, spec: solprop.stats.StatsSpec, z: float, step: int) -> None"
    ),
    "_drain_step_stats": "(self) -> solprop.backends.protocol.StepStats",
    "stats": "(self, spec: solprop.stats.StatsSpec) -> dict[str, numpy.ndarray]",
    "_hoist": "(self, z0: float) -> None",
    "_stage_scalars": (
        "(self, z0: float, z1: float) -> solprop.translate.stage.StageScalars"
    ),
    "_stage_bundles": (
        "(self, z0: float, positions: collections.abc.Sequence[float])"
        " -> tuple[solprop.translate.stage.StageScalars, ...]"
    ),
    "_on_axis_synthesis": (
        "(self, modes: collections.abc.Sequence[solprop.modes.protocol.Mode])"
        " -> jaxtyping.Float[ndarray, 'n_rows n_modes']"
    ),
    "_require_grid": "(self) -> solprop.grid.TimeGrid",
    "_require_state": "(self) -> _StateT",
    "metadata": "(self) -> dict[str, object]",
    "restore_call_ledgers": (
        "(self, logical: collections.abc.Mapping[str, object], "
        "atomic: collections.abc.Mapping[str, object] | None = None) -> None"
    ),
    "observe": "(self, *, field: bool = True) -> solprop.backends.protocol.Observation",
    "held_observation": "(self) -> solprop.backends.protocol.Observation | None",
    "_hold": "(self, state: _StateT | None) -> None",
    "_held_field": f"(self) -> {FIELD}",
    "set_initial": (
        "(self, field: jaxtyping.Complex[ndarray, 'n_batch n_modes n_pol n_omega'], "
        "noise: solprop.noise.NoiseRealisation | None, *, z: float = 0.0) -> None"
    ),
    "set_progress": "(self, reporter: solprop.log.ProgressReporter) -> None",
    "propagate_to": "(self, z: float) -> solprop.backends.protocol.StepSummary",
    "available_device_memory": "(self) -> int | None",
}
"""Every ``StepDriver`` member a subclass overrides or calls, with its signature.

The literal is the seam a separately built backend is compiled against, and
``SCHEMA_VERSION`` is its version: change a signature here only together with a
bump of the version and a changelog line (``AGENTS.md``, "Backends, kernels and hot
paths"). ``solprop-native`` looks these names up at construction (its
``DRIVER_SURFACE``).
"""

DRIVER_ATTRIBUTES = (
    "_adams",
    "_low_storage",
    "_in_flight",
    "_problem",
    "_terms",
    "_grid",
    "_modes",
    "_profiles",
    "_z",
    "_state",
    "_noise",
    "_closed",
    "helpers",
)
"""The attributes a subclass reads or sets, present from construction.

``helpers`` (schema 34) is a class attribute: each backend sets its own record of the
per-stage and per-step evaluations, which the shared code passes down.
"""

SEPARABLE_OPERATOR_FIELDS = (
    "k_squared",
    "kperp_sq",
    "k_squared_low",
    "kperp_sq_low",
    "frame",
    "spectral_rate",
    "prefactor_numerator",
    "prefactor_clamped",
    "active",
    "evanescent_rate",
    "n_pol",
    "z_span",
    "kappa_sq",
    "kperp_rate",
)
"""The separable leaf's fields; schema 15 appended ``kappa_sq``, 31 ``kperp_rate``."""

STAGE_TERMS_FIELDS = ("phi", "prefactor", "separable")
STAGE_SCALARS_FIELDS = (
    "z0",
    "z1",
    "phi_delta",
    "phi_at",
    "prefactor",
    "density",
    "taper_scale",
    "raman_rates",
    "kerr_density",
    "plasma_densities",
    "plasma_preionisation",
    "raman_densities",
    "flow_section",
    "flow_weights",
    "transform_weights",
    "dispersive_kerr_densities",
)
"""The stage bundles a backend packs a trial from, field by field, in order."""


def test_a_trials_bundles_taken_together_are_kept_until_the_next_hoist(driver):
    """``_stage_bundles`` reads the tables once and ``_stage_scalars`` takes the kept.

    The bundles equal the per-position ones bit for bit; a later stage at the same
    anchor and position is the kept bundle itself, and the next step's hoist drops
    them, so nothing a later step reads was taken for an earlier one.
    """
    made = driver()
    z0, dz = LENGTH / 4, LENGTH / 8
    positions = [z0 + dz, z0, z0 + 0.5 * dz, z0 + dz]
    made._hoist(z0)
    together = made._stage_bundles(z0, positions)
    assert together[0] is together[3]
    for z1, bundle in zip(positions, together, strict=True):
        alone = scalars(made._problem, z0, z1)
        assert bundle.z0 == z0 and bundle.z1 == z1
        assert np.array_equal(present(bundle.phi_delta), present(alone.phi_delta))
        assert np.array_equal(present(bundle.prefactor), present(alone.prefactor))
        assert bundle.density == alone.density
        assert made._stage_scalars(z0, z1) is bundle
    # Asked again at the same anchor, nothing is read: the kept bundles come back.
    again = made._stage_bundles(z0, positions[1:3])
    assert again[0] is together[1] and again[1] is together[2]
    made._hoist(z0 + dz)
    assert made._stage_scalars(z0, z0 + dz) is not together[0]
    made.close()


def test_the_surface_a_backend_subclasses_is_frozen(driver):
    """The driver's protected surface and the stage bundles are ``SCHEMA_VERSION`` 15's.

    A separately built backend is compiled against these names and shapes, and its build
    stays valid across solprop commits because this test, not a version pin, guards
    them: a change here is a change of the kernel contract and bumps the version.
    Version 6 added the four mixture bundles -- ``kerr_density``, ``plasma_densities``,
    ``plasma_preionisation`` and ``raman_densities`` -- beside the total ``density`` and
    per-term ``raman_rates`` of version 5, so a compiled backend packs one scalar per
    plasma channel and per Raman term rather than one. Version 13 added ``flow_section``
    and ``flow_weights`` at the end, which is the whole of that bump: a varying coupled
    flow's frozen section and the weights its residual is evaluated over are host
    scalars, and no driver signature and no `LinearOperator` row moved with them.
    Version 14 added ``transform_weights`` at the end, the weights a moving guide's
    synthesis and projection maps are combined over, and nothing else. Version 15 moved
    none of these: it added the ``"slab"`` spatial kind and appended ``kappa_sq`` to the
    separable leaf, whose literal is :data:`SEPARABLE_OPERATOR_FIELDS`. Version 17 added
    the five ``_adams_*`` hooks of the shared Adams march (SEP 14 phase 2) and version
    18 ``_dense_stages``, the accepted step's stages for DOPRI5's
    continuous extension. Version 20 added ``_adams_lab_source``, the exponential
    Adams formulation's untransported source (SEP 16), and version 21 ``_adams_engine``,
    the Adams history's engine (SEP 16 phase 3b), and version 22 ``_adams_block``, a
    coupled flow's frozen block (SEP 16 phase 5). Version 23 moved no hook. Version 24
    appended ``dispersive_kerr_densities``, each dispersive electronic channel's
    number density at the stage (SEP 6 phase 1), and moved no hook. Version 25 moved
    none of these: it appended the plasma column's running-integral rule (SEP 18).
    Version 26 moved none either: it stored the transform's kind (SEP 19). Version 27
    moved none: it widened the TDSE descriptor to haCC and atomic windows (SEP 20).
    Version 28 added ``_stage_bundles``, a trial's stage bundles read together (SEP 25).
    Version 29 added ``_low_storage_engine`` and the attributes ``_low_storage`` and
    ``_in_flight``, the low-storage march's engine hook, march and in-flight guard
    (SEP 24 phase 1). Version 30 moved none: it widened the TDSE descriptor to the
    version-4 ``gated_derivative`` source filter (SEP 26 phase 1). Version 31 gave
    ``observe`` its ``field`` keyword and added ``held_observation`` (SEP 22 phase 2),
    and appended ``kperp_rate`` to the separable leaf. Version 32 added ``_hold`` and
    ``_held_field``, where a backend holds the low-storage march's field-free plane
    and copies it to the host, and the engine protocol's ``atomic`` (SEP 24 phase 3).
    Version 33 adds the analytic collar derivative to Grids (SEP 27), without
    changing these driver or stage signatures. Version 34 adds the class attribute
    ``helpers``, the backend's per-stage and per-step evaluations (SEP 29 phase 2).
    Version 35 adds the problem's plane measure ``flux`` (SEP 34), without changing
    these driver or stage signatures.
    """
    assert SCHEMA_VERSION == 35, "the seam changed: rewrite the literals of this test"
    for name, signature in {
        "observe_members": (
            "(self) -> tuple[solprop.backends.protocol.Observation, ...]"
        ),
        "member_metadata": "(self) -> tuple[dict[str, object], ...]",
    }.items():
        assert str(inspect.signature(getattr(BatchBackend, name))) == signature, name
    for name, signature in DRIVER_SURFACE.items():
        assert str(inspect.signature(getattr(StepDriver, name))) == signature, name
    made = driver()
    for name in DRIVER_ATTRIBUTES:
        assert hasattr(made, name), name
    made.close()
    assert tuple(f.name for f in dataclasses.fields(StageTerms)) == STAGE_TERMS_FIELDS
    assert (
        tuple(f.name for f in dataclasses.fields(StageScalars)) == STAGE_SCALARS_FIELDS
    )


def test_separable_operator_fields_are_frozen_with_the_driver_seam():
    assert tuple(field.name for field in dataclasses.fields(SeparableOperator)) == (
        SEPARABLE_OPERATOR_FIELDS
    )


def test_linear_operator_fields_are_frozen_with_the_driver_seam():
    # Schema 11 appended the four power rows and schema 12 the two general-modal
    # leaves; neither altered the stage bundles or the driver surface.
    assert tuple(field.name for field in dataclasses.fields(LinearOperator)) == (
        "phi_representation",
        "phi_breakpoints",
        "phi_coefficients",
        "phi_degrees",
        "hard_kinks",
        "phi_constant",
        "phi_expansion",
        "beta0",
        "beta1",
        "separable",
        "power_weight_representation",
        "power_weight",
        "power_weight_expansion",
        "power_weight_constant",
        "coupled_flow",
        "modal_metric",
    )


def test_the_adams_and_dense_hooks_default_to_refusal(driver):
    # A backend that executes neither the Adams march nor dense observation inherits
    # these defaults; the propagator is the shared one, exp(Phi(b) - Phi(a)).
    made = driver()
    field = made._require_state()
    with pytest.raises(BackendError, match="low-call Adams integrator"):
        made._adams_source(field, 0.0, LENGTH / 4)
    with pytest.raises(BackendError, match="does not observe densely"):
        made._dense_stages()
    propagator = made._adams_propagator(0.0, LENGTH / 4)
    expected = np.exp(delta_phi(made._problem.linear, 0.0, LENGTH / 4))
    np.testing.assert_allclose(
        propagator.reshape(-1), np.broadcast_to(expected, propagator.shape).reshape(-1)
    )
