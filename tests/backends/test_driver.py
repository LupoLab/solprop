# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference's step control, driven through a stage evaluation that is scripted.

The reference's session (:class:`solprop.backends.numpy_ref.session.HostSession`)
schedules a run and drives its march; here the march's engine is a script
(:class:`scripted_driver.HostScriptedDriver`), so every claim is about the *step
control* -- landing, acceptance, the controller's state, the history, the noise models,
the guards, the statistics rows -- and none of them depends on any physics being
evaluated correctly. The descriptor, the stage bundles and the backend surface a
compiled backend is built against are frozen here too (``docs/dev/kernel-contract.md``).
"""

import dataclasses
import io
import logging
import typing
from typing import Any, cast

import numpy as np
import pytest

from bounds import assert_within
from scripted_driver import HostScriptedDriver
from sessions.executors import IDENTITY, Started, drive
from sessions.runs import run
from solprop.backends.events import ObservationReady
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.dense import RungeKuttaDense
from solprop.backends.numpy_ref.session import INITIAL_STEP_STATS_CAPACITY
from solprop.backends.protocol import Backend
from solprop.backends.session import Start
from solprop.exceptions import BackendError
from solprop.log import BarReporter, LogReporter, NullReporter
from solprop.noise import input_noise, modified_noise
from solprop.stats import StatsSpec
from solprop.stepper import PIController
from solprop.stepper.multistep import MarchPolicy
from solprop.stepper.schedule import FixedSchedule, reached
from solprop.translate import routes
from solprop.translate.linop import LinearOperator, delta_phi
from solprop.translate.problem import PropagationProblem, Translation
from solprop.translate.run_request import RunRequest
from solprop.translate.spatial import SeparableOperator
from solprop.translate.stage import StageScalars, StageTerms, scalars

LENGTH = 0.02


def present(value):
    """Return an optional array, having asserted that it is there.

    The state and the noise are ``None`` before a start is loaded and after ``close``,
    so the seam types them optional; a test that has just loaded one says so here
    rather than asserting around it at every use.
    """
    assert value is not None
    return value


def resumed(run_request, problem, z, *, field=None, noise=None):
    """Return a start resuming at `z`, whose plane the run already saved."""
    planes = np.asarray(run_request.stepper.save_planes)
    return Start(
        field=problem.initial.field0 if field is None else field,
        z=float(z),
        noise=noise,
        plane_index=int(np.searchsorted(planes, z, side="right")),
        plane_pending=False,
    )


def session_of(made, problem, run_request, start=None, **options):
    """Return a session started on `made`, which it leaves open."""
    return made.prepare(problem, run_request).start(
        Start.entrance(problem) if start is None else start, owns=False, **options
    )


def planes_of(result):
    """Return each observation past the launch, with its assembled history and rows."""
    return [
        (event, history, rows)
        for event, history, rows in result.assembled()
        if isinstance(event, ObservationReady) and event.plane_index >= 0
    ]


@dataclasses.dataclass
class Scripted:
    """A scripted backend with its problem and request."""

    made: Any
    problem: PropagationProblem
    request: RunRequest

    def run(self, start=None, **options):
        """Drive a session from `start` to its end, exporting every field."""
        return run(self.made, self.problem, self.request, start, **options)


@pytest.fixture
def scripted():
    """The scripted reference backend."""
    return HostScriptedDriver


@pytest.fixture
def driver(soliton_problem, narrow_grid, scripted):
    """Return ``(**kwargs) -> Scripted`` on the soliton's planes."""

    def build(*, planes=(LENGTH,), **kwargs):
        problem, run_request = soliton_problem(narrow_grid, 1, save_planes=list(planes))
        made = scripted(problem, run_request, grid=narrow_grid, **kwargs)
        return Scripted(made, problem, run_request)

    return build


def test_a_march_lands_exactly_on_its_save_plane(driver):
    """The position is the plane itself, not an accumulation of steps that nears it."""
    made = driver()
    result = made.run()
    final = result.observations[-1]
    assert final.z == LENGTH
    assert made.made.march.state.z == LENGTH
    assert final.summary.z == LENGTH
    assert final.summary.n_steps > 1
    assert final.summary.n_rejected == 0


def test_the_landing_guard_admits_the_target_and_nothing_short_of_it():
    """``reached`` is the eight-ulp guard the schedule's own landing uses."""
    assert reached(1.0, 1.0)
    assert reached(1.0 - 4 * np.spacing(1.0), 1.0)
    assert not reached(1.0 - 1e-12, 1.0)


def test_a_rejected_step_is_recorded_and_not_kept(driver):
    """The state after a rejection is the state before it, and the history says so."""
    # 5.0 is far above the controller's threshold of one, so the first step is refused
    # and the second, from the same position, is taken.
    scripted = driver(norms=[5.0, 0.1], factor=2.0)
    made = scripted.made
    result = scripted.run()
    before = result.fields[0]
    ((_, history, _),) = planes_of(result)
    assert not bool(history.accepted[0])
    assert bool(history.accepted[1])
    assert made.calls[0][0] == made.calls[1][0] == 0.0
    # The refused step's shorter successor is what the controller asked for next.
    assert made.calls[1][1] < made.calls[0][1]
    # The scripted step doubles the state, so the division by powers of two is exact.
    assert np.array_equal(
        before, result.fields[-1] / 2.0 ** (int(history.accepted.sum()))
    )


def test_a_rejected_step_is_retried_at_the_rejection_factor_alone(
    scripted, soliton_problem, narrow_grid
):
    """A refused step is retried shorter by the rejection factor, with no growth on top.

    The controller's growth factor belongs to the accepted history and is applied once
    per accepted step. Applied again after a rejection it undid the shrink: with a
    growth cap of two, a step whose norm sat just above one was refused unchanged
    forever.
    """
    controller = PIController(rtol=1e-10, atol=1e-18, max_factor=2.0)
    problem, run_request = soliton_problem(
        narrow_grid, 1, save_planes=[LENGTH], controller=controller
    )
    # Three easy steps fill the lag and ask for growth; the fourth fails narrowly.
    made = scripted(
        problem, run_request, grid=narrow_grid, norms=[0.01, 0.01, 0.01, 1.0001]
    )
    history, _ = run(made, problem, run_request, export=None).joined
    accepted = history.accepted
    assert [bool(flag) for flag in accepted[:4]] == [True, True, True, False]
    assert controller.propose([0.01, 0.01, 0.01]) > 1.1
    rejected, retried = made.calls[3][1], made.calls[4][1]
    assert retried == pytest.approx(rejected * controller.rejection_factor(1.0001))


@pytest.mark.parametrize("diverged", [np.inf, np.nan], ids=["infinite", "nan"])
def test_a_diverged_trial_is_rejected_shrunk_and_recorded_as_infinity(driver, diverged):
    """A non-finite norm, as an overflowing propagator scores, rejects.

    The controller retries at its smallest factor, and the history records the trial as
    infinity, so the observation at the next plane reads it back rather than refusing
    the record and ending the run.
    """
    scripted = driver(norms=[diverged, 0.1], factor=2.0)
    made = scripted.made
    history, _ = scripted.run(export=None).joined
    assert not bool(history.accepted[0])
    assert history.error_norm[0] == np.inf
    assert np.all(np.isfinite(history.error_norm[1:]))
    assert bool(history.accepted[1])
    controller = made.request.stepper.controller
    assert made.calls[1][0] == 0.0
    assert made.calls[1][1] == pytest.approx(made.calls[0][1] * controller.min_factor)


def test_the_history_covers_only_the_steps_since_the_last_observation(driver):
    """An observation's chunks are the attempts since the previous one."""
    result = driver(planes=(LENGTH / 2, LENGTH)).run(export=None)
    (_, first, _), (_, second, _) = planes_of(result)
    assert first.n_steps > 0
    assert second.n_steps > 0
    assert float(first.z[-1]) == LENGTH / 2
    assert float(second.z[0]) > LENGTH / 2


def test_a_replayed_schedule_takes_its_steps_as_given(
    scripted, soliton_problem, narrow_grid
):
    """A replay never rejects and reads its position off the schedule's edges."""
    schedule = FixedSchedule.uniform(0.0, LENGTH, LENGTH / 8)
    problem, run_request = soliton_problem(
        narrow_grid, 1, save_planes=[LENGTH], schedule=schedule
    )
    made = scripted(problem, run_request, grid=narrow_grid, norms=[5.0] * 8)
    final = run(made, problem, run_request, export=None).observations[-1]
    summary = final.summary
    assert summary.n_steps == 8
    assert summary.n_rejected == 0
    assert final.z == LENGTH
    # Every step is the schedule's, not the controller's.
    assert [dz for _, dz in made.calls] == pytest.approx([LENGTH / 8] * 8)
    assert not summary.met_tolerance


def test_a_replay_that_runs_out_says_so(scripted, soliton_problem, narrow_grid):
    """A schedule too short for the target is an error, not a quiet stop."""
    problem, run_request = soliton_problem(
        narrow_grid,
        1,
        save_planes=[LENGTH / 2, LENGTH],
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / 8),
    )
    made = scripted(problem, run_request, grid=narrow_grid)
    run(made, problem, run_request, export=None, observations=2)
    # The operator table still covers the whole run, so the guard that fires is the
    # schedule's own and not the table's: the march replays a schedule that ends at
    # the plane it stands on (the translation refuses one that misses a save plane).
    short = FixedSchedule.uniform(0.0, LENGTH / 2, LENGTH / 8)
    with pytest.raises(BackendError, match="cannot reach"):
        made.march.replay_to(short, LENGTH)


def test_a_schedule_of_equal_steps_lands_on_a_plane_its_running_sum_misses(
    scripted, soliton_problem, narrow_grid
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
    problem, run_request = soliton_problem(
        narrow_grid, 1, save_planes=[LENGTH / 2, LENGTH], schedule=schedule
    )
    made = scripted(problem, run_request, grid=narrow_grid)
    held = []

    def march_state(session, batch):
        del session
        if any(isinstance(event, ObservationReady) for event in batch.events):
            held.append((made.march.state.cursor, made.march.state.z))

    run(made, problem, run_request, export=None, between=march_state)
    assert held[1] == (200, middle)
    assert held[2][0] == 400
    with session_of(
        made, problem, run_request, resumed(run_request, problem, LENGTH / 2)
    ):
        assert made.march.state.cursor == 200


def test_a_resume_continues_a_replay_from_the_edge_it_stopped_at(
    scripted, soliton_problem, narrow_grid
):
    """A start at a plane finds the schedule index the position belongs to."""
    schedule = FixedSchedule.uniform(0.0, LENGTH, LENGTH / 8)
    problem, run_request = soliton_problem(
        narrow_grid, 1, save_planes=[LENGTH / 2, LENGTH], schedule=schedule
    )
    made = scripted(problem, run_request, grid=narrow_grid)
    start = resumed(run_request, problem, LENGTH / 2)
    with session_of(made, problem, run_request, start):
        assert made.march.state.cursor == 4
    run(made, problem, run_request, resumed(run_request, problem, LENGTH / 2))
    assert len(made.calls) == 4


def test_a_resume_off_the_schedule_s_edges_is_refused(
    scripted, soliton_problem, narrow_grid
):
    """A position the run never stopped at cannot continue a replay in step."""
    schedule = FixedSchedule.uniform(0.0, LENGTH, LENGTH / 8)
    problem, run_request = soliton_problem(
        narrow_grid, 1, save_planes=[LENGTH], schedule=schedule
    )
    made = scripted(problem, run_request, grid=narrow_grid)
    with pytest.raises(BackendError, match="one of its edges"):
        session_of(
            made, problem, run_request, resumed(run_request, problem, LENGTH / 3)
        )


def test_the_input_noise_model_adds_its_realisation_once(
    scripted, soliton_problem, narrow_grid
):
    """Added at the entrance and never again, so a resume is not given a second draw."""
    problem, run_request = soliton_problem(
        narrow_grid,
        1,
        save_planes=[LENGTH],
        noise=input_noise(narrow_grid, np.random.default_rng(7)),
    )
    masked = problem.initial.field0 * problem.grids.active_mask
    entrance = scripted(problem, run_request, grid=narrow_grid)
    with session_of(entrance, problem, run_request):
        assert not np.array_equal(present(entrance.workspace.state), masked)
        assert entrance.workspace.noise is None
    later = scripted(problem, run_request, grid=narrow_grid)
    with session_of(
        later, problem, run_request, resumed(run_request, problem, LENGTH / 2)
    ):
        assert np.array_equal(present(later.workspace.state), masked)


def test_the_modified_noise_model_keeps_its_realisation_apart(
    scripted, soliton_problem, narrow_grid
):
    """The propagating field never carries it; it is advanced by the unitary phase."""
    problem, run_request = soliton_problem(
        narrow_grid,
        1,
        save_planes=[LENGTH],
        noise=modified_noise(narrow_grid, np.random.default_rng(7)),
    )
    made = scripted(problem, run_request, grid=narrow_grid)
    with session_of(made, problem, run_request):
        assert np.array_equal(
            present(made.workspace.state),
            problem.initial.field0 * problem.grids.active_mask,
        )
        drawn = present(made.workspace.noise)
        advanced = present(
            made.workspace.noise_at(made.workspace.stage_scalars(0.0, LENGTH / 2))
        )
        # Dispersion yes, loss and gain no: the magnitude is untouched by the advance.
        assert np.allclose(np.abs(advanced), np.abs(drawn))
        assert not np.array_equal(advanced, drawn)
        assert present(made.workspace.noise_phase(0.0)).shape == problem.shape


def test_a_given_realisation_replaces_the_contract_s(
    scripted, soliton_problem, narrow_grid
):
    """The realisation a start carries wins over the assembled one."""
    problem, run_request = soliton_problem(
        narrow_grid,
        1,
        save_planes=[LENGTH],
        noise=modified_noise(narrow_grid, np.random.default_rng(7)),
    )
    made = scripted(problem, run_request, grid=narrow_grid)
    start = Start.entrance(
        problem, noise=modified_noise(narrow_grid, np.random.default_rng(11))
    )
    with session_of(made, problem, run_request, start):
        assert not np.array_equal(present(made.workspace.noise), problem.initial.noise0)


def test_a_run_without_noise_reports_no_phase(driver, narrow_grid):
    """``noise_model = "none"`` has neither a realisation nor a phase to report."""
    scripted = driver()
    workspace = scripted.made.workspace
    with session_of(scripted.made, scripted.problem, scripted.request):
        assert (
            workspace._noise_field(
                modified_noise(narrow_grid, np.random.default_rng(3))
            )
            is None
        )
        assert workspace.noise_phase(0.0) is None
        assert workspace.noise_at(workspace.stage_scalars(0.0, 0.0)) is None


def test_the_guards_refuse_what_cannot_be_done(scripted, soliton_problem, narrow_grid):
    """Closed, unloaded, off the table, and the wrong shape."""
    problem, run_request = soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    empty = scripted(problem, run_request, grid=narrow_grid)
    with pytest.raises(BackendError, match="no initial state"):
        empty.workspace.committed()
    with pytest.raises(BackendError, match="expected"):
        session_of(
            empty,
            problem,
            run_request,
            Start(field=np.zeros((1, 1, 3), dtype=complex), z=0.0),
        )
    with pytest.raises(BackendError, match="inside the operator table"):
        session_of(
            empty, problem, run_request, Start(problem.initial.field0, 2 * LENGTH)
        )

    empty.close()
    with pytest.raises(BackendError, match="closed"):
        session_of(empty, problem, run_request)
    with pytest.raises(BackendError, match="closed"):
        empty.timing_probe(Start.entrance(problem))


def test_statistics_without_a_grid_say_which_backend_to_give_one_to(
    scripted, soliton_problem, narrow_grid
):
    """The message names the backend, so it is actionable for whichever one it is."""
    problem, run_request = soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    made = scripted(problem, run_request)
    with pytest.raises(BackendError, match="'scripted'"):
        made.workspace.statistics._require_grid()
    assert (
        scripted(
            problem, run_request, grid=narrow_grid
        ).workspace.statistics._require_grid()
        is narrow_grid
    )


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


def reported(made, problem, run_request, reporter, start=None, **options):
    """Drive a session whose attempts and start go to `reporter`; return the run."""
    session = session_of(
        made,
        problem,
        run_request,
        start,
        listener=reporter.step,
        started=reporter.started,
    )
    try:
        return drive(Started(session), **options)
    finally:
        session.close()


def test_a_reporter_sees_every_attempted_step_including_the_rejected_ones(driver):
    """Every attempt reaches the session's listener, so nothing is missed."""
    reporter = Events()
    scripted = driver(norms=[5.0, 0.1], factor=2.0)
    result = reported(scripted.made, scripted.problem, scripted.request, reporter)
    summary = result.observations[-1].summary
    assert reporter.order[0] == "started"
    assert reporter.extent == (0.0, LENGTH, 1)
    assert reporter.order.count("step") == summary.n_steps
    assert sum(1 for _, _, accepted in reporter.steps if not accepted) == 1
    assert summary.n_rejected == 1
    # The rejected attempt is reported at the position it would have reached, which the
    # retry then reaches more slowly: the reporter, not the session, has to know.
    rejected_at = next(z for z, _, accepted in reporter.steps if not accepted)
    assert rejected_at > reporter.steps[1][0]


def test_started_is_announced_once_per_session_and_a_resume_announces_it_again(
    driver,
):
    reporter = Events()
    scripted = driver(planes=(LENGTH / 2, LENGTH))
    made, problem, request = scripted.made, scripted.problem, scripted.request
    result = reported(made, problem, request, reporter, export=IDENTITY)
    assert reporter.order.count("started") == 1
    assert reporter.extent == (0.0, LENGTH, 2)
    middle = result.fields[1]
    reported(
        made,
        problem,
        request,
        reporter,
        resumed(request, problem, LENGTH / 2, field=middle),
    )
    assert reporter.order.count("started") == 2
    assert reporter.extent == (LENGTH / 2, LENGTH, 2)


@pytest.mark.parametrize("period", [0.0, 1e9])
def test_a_run_with_a_reporter_is_the_run_without(soliton_problem, narrow_grid, period):
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

    def marched(reporter):
        problem, run_request = soliton_problem(
            narrow_grid, 1, save_planes=[LENGTH / 2, LENGTH], schedule=schedule
        )
        made = NumpyRefBackend(problem, run_request, grid=narrow_grid)
        try:
            result = reported(made, problem, run_request, reporter, export=IDENTITY)
        finally:
            made.close()
        observed = planes_of(result)
        return result.fields[1:], [history for _, history, _ in observed]

    quiet_fields, quiet_history = marched(NullReporter())
    rtol = soliton_problem(
        narrow_grid, 1, save_planes=[LENGTH]
    ).request.stepper.controller.rtol
    reporters = [
        LogReporter(
            period=period, clock=lambda: 0.0, logger=logging.getLogger("quiet")
        ),
        # A bar redrawing on every step, drawing on a string.
        BarReporter(period=1e-9, file=io.StringIO(), ncols=60, ascii=True),
    ]
    for reporter in reporters:
        loud_fields, loud_history = marched(reporter)
        for quiet, loud in zip(quiet_fields, loud_fields, strict=True):
            assert_within(loud, quiet, bound, what="field")
        for quiet, loud in zip(quiet_history, loud_history, strict=True):
            assert quiet.n_steps == loud.n_steps > 0
            for name in columns:
                assert np.array_equal(getattr(quiet, name), getattr(loud, name)), name
            np.testing.assert_allclose(
                loud.error_norm, quiet.error_norm, rtol=0, atol=7 * bound / rtol
            )


def test_the_reference_reports_its_calls_in_order(soliton_problem, narrow_grid):
    # The native backend reports through its session's listener too (the lifecycle
    # suite's progress cases, ``tests/sessions``).
    reporter = Events()
    planes = [LENGTH / 2, LENGTH]
    problem, run_request = soliton_problem(narrow_grid, 1, save_planes=planes)
    made = NumpyRefBackend(problem, run_request, grid=narrow_grid)

    def plane(session, batch):
        del session
        for event in batch.events:
            if isinstance(event, ObservationReady) and event.plane_index >= 0:
                reporter.plane(event.z, event.summary)

    try:
        reported(made, problem, run_request, reporter, between=plane)
    finally:
        made.close()
    reporter.finished(1.0)
    assert reporter.order[0] == "started"
    assert reporter.order[-1] == "finished"
    assert reporter.order.count("started") == 1
    assert reporter.order.count("finished") == 1
    assert reporter.order.count("plane") == len(planes)
    assert reporter.order.count("step") == sum(
        summary.n_steps for _, summary in reporter.planes
    )
    # One `started`, then steps and planes interleaved with a plane last before the end.
    assert reporter.order[-2] == "plane"


# -------------------------------------------- the statistics channel between planes


def periodic(translation, names=("energy",), *, period=1):
    """Return a translation whose request asks for `names` periodically too."""
    problem, run_request = translation
    spec = StatsSpec(names=names, periodic=names, period=period)
    return Translation(problem, run_request.for_run(stats=spec))


def observed_planes(made, problem, run_request):
    """Run to the end; return each plane's assembled history and rows."""
    result = run(made, problem, run_request, export=None)
    return [(history, rows) for _, history, rows in planes_of(result)]


def rows_of(made, problem, run_request):
    """Run to the one plane and return the rows it assembled."""
    ((_, rows),) = observed_planes(made, problem, run_request)
    return rows


def test_a_row_is_taken_at_the_position_the_accepted_step_reached(
    scripted, soliton_problem, narrow_grid
):
    """The trap this test exists for: a row describes the step just committed.

    An adaptive march records an attempt *before* it finishes the state and advances
    ``z``, a replay *after* both. So a channel evaluated from the attempt's record
    reads the previous step's position and state on an adaptive run and the current
    step's on a replayed one -- and a cost gate is naturally specified on a replayed
    schedule, where the wrong version passes. Both paths are asserted here, against the
    same expectation.
    """
    schedule = FixedSchedule.uniform(0.0, LENGTH, LENGTH / 8)
    for extra in ({}, {"schedule": schedule}):
        problem, run_request = periodic(
            soliton_problem(narrow_grid, 1, save_planes=[LENGTH], **extra)
        )
        made = scripted(problem, run_request, grid=narrow_grid)
        rows = rows_of(made, problem, run_request)
        # The scripted statistics read the position they were taken at.
        assert rows.n_rows > 0
        np.testing.assert_array_equal(rows.values["energy"][:, 0], rows.z)
        # And that position is the end of an accepted step, never the start.
        assert rows.z[-1] == pytest.approx(LENGTH)
        assert rows.z[0] > 0.0


def test_the_channel_is_empty_unless_the_run_asks_for_one(driver):
    """The default spec has no periodic subset, so nothing is buffered or flushed."""
    scripted = driver()
    rows = rows_of(scripted.made, scripted.problem, scripted.request)
    assert rows.n_rows == 0
    assert rows.names == ()


def test_a_row_indexes_the_attempt_it_was_taken_after(
    scripted, soliton_problem, narrow_grid
):
    """The index is into the plane's own history, which counts rejected attempts."""
    problem, run_request = periodic(
        soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    )
    made = scripted(problem, run_request, grid=narrow_grid, norms=[5.0, 0.1, 0.1])
    ((history, rows),) = observed_planes(made, problem, run_request)
    assert history.n_rejected == 1
    # The first attempt was rejected, so the first row indexes attempt 1, not 0.
    assert rows.step[0] == 1
    accepted = np.flatnonzero(history.accepted)
    np.testing.assert_array_equal(rows.step, accepted)
    np.testing.assert_allclose(rows.z, history.z[accepted])


def test_the_period_counts_accepted_steps_and_a_rejection_is_not_one(
    scripted, soliton_problem, narrow_grid
):
    """A rejected attempt neither advances the phase nor records a row."""
    problem, run_request = periodic(
        soliton_problem(narrow_grid, 1, save_planes=[LENGTH]), period=2
    )
    made = scripted(
        problem, run_request, grid=narrow_grid, norms=[5.0, 0.1, 0.1, 0.1, 0.1]
    )
    ((history, rows),) = observed_planes(made, problem, run_request)
    accepted = np.flatnonzero(history.accepted)
    # Every second *accepted* attempt, so the rejection shifts nothing.
    np.testing.assert_array_equal(rows.step, accepted[1::2])


def test_the_rows_of_one_plane_are_flushed_with_it(
    scripted, soliton_problem, narrow_grid
):
    """An observation drains the buffer, so a plane's rows are its own."""
    planes = [LENGTH / 2, LENGTH]
    problem, run_request = periodic(soliton_problem(narrow_grid, 1, save_planes=planes))
    made = scripted(problem, run_request, grid=narrow_grid)
    seen = []
    for history, rows in observed_planes(made, problem, run_request):
        seen.append(rows)
        # Each plane's rows index that plane's own history, from zero.
        assert rows.step[0] < history.n_steps
    assert all(one.n_rows for one in seen)
    assert seen[0].z[-1] == pytest.approx(LENGTH / 2)
    assert seen[1].z[0] > LENGTH / 2


def test_the_buffer_grows_across_a_plane_with_more_steps_than_its_capacity(
    scripted, soliton_problem, narrow_grid
):
    """More rows than `INITIAL_STEP_STATS_CAPACITY`, so the doubling is exercised."""
    n_steps = 3 * INITIAL_STEP_STATS_CAPACITY
    schedule = FixedSchedule.uniform(0.0, LENGTH, LENGTH / n_steps)
    problem, run_request = periodic(
        soliton_problem(narrow_grid, 1, save_planes=[LENGTH], schedule=schedule)
    )
    made = scripted(problem, run_request, grid=narrow_grid)
    rows = rows_of(made, problem, run_request)
    assert rows.n_rows == n_steps
    assert np.all(np.diff(rows.z) > 0.0)
    np.testing.assert_array_equal(rows.step, np.arange(n_steps))


def test_a_load_starts_the_period_again(scripted, soliton_problem, narrow_grid):
    """Each session counts the period from its load, so a resume is not the
    uninterrupted run."""
    problem, run_request = periodic(
        soliton_problem(narrow_grid, 1, save_planes=[LENGTH]), period=3
    )
    made = scripted(problem, run_request, grid=narrow_grid)
    session = cast("Any", session_of(made, problem, run_request))
    assert session._accepted_steps == 0
    drive(Started(session))
    assert session._accepted_steps > 0
    session.close()
    with cast("Any", session_of(made, problem, run_request)) as again:
        assert again._accepted_steps == 0
        assert again._rows.drain().n_rows == 0


def test_a_statistic_that_changes_shape_within_a_plane_is_refused(
    scripted, soliton_problem, narrow_grid
):
    """The buffer is laid out from the first row and indexes by that shape.

    Nothing the package builds can do this -- the spec is fixed when the backend is
    built -- but a workspace whose statistics disagreed with themselves would otherwise
    be caught by a broadcast error deep in numpy rather than by a message naming the
    statistic.
    """
    problem, run_request = periodic(
        soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    )
    made = scripted(problem, run_request, grid=narrow_grid)
    taken = [0]

    def widening(spec, z):
        del z
        taken[0] += 1
        return {name: np.zeros((1, taken[0])) for name in spec.names}

    made.workspace.stats = widening
    with pytest.raises(BackendError, match="came back with shape"):
        run(made, problem, run_request, export=None)


# ----------------------------------------------------------- the executed norm


def test_the_history_records_the_executed_norm_and_the_controller_the_equivalent(
    driver, monkeypatch
):
    """A landing step clipped short is written up at the norm it scored.

    Were the history to carry the controller's *equivalent* norm -- what the proposal
    would have scored, ``norm * (h_proposed / h_taken) ** 5`` -- the closing tolerance
    ratio of a run whose last steps were forced short would read as a failure; the full
    M2 case would close at 6.5e4. The controller still learns from the equivalent, so a
    plane does not restart it from a sliver; the history and everything read off it
    carry the norm each step actually committed.
    """
    scripted = driver()
    made = scripted.made
    proposals = []
    trial = made._trial_step

    def recorded(z, dz):
        proposals.append(made.march.state.proposal)
        return trial(z, dz)

    monkeypatch.setattr(made, "_trial_step", recorded)
    history, _ = scripted.run(export=None).joined
    accepted = np.flatnonzero(history.accepted)
    last = accepted[-1]
    taken = float(history.dz[last])
    proposal = proposals[last]
    # The last accepted step is the landing on the plane, and it is not the proposal.
    assert taken != proposal
    executed = (taken / made._scale) ** 5
    assert float(history.error_norm[last]) == pytest.approx(executed)
    assert made.march.state.norms[-1] == pytest.approx(
        executed * (proposal / taken) ** 5
    )
    # And no accepted step of an adaptive run scores above one, by construction.
    assert np.all(history.error_norm[accepted] <= 1.0)


def test_the_controller_keeps_only_the_norms_it_reads(driver):
    """The controller reads two norms, a lag back; a march keeps those, not every step.

    The first entry is still the start-up rule's cue: a session's march starts with no
    norms, and its length is one only after the first accepted step.
    """
    scripted = driver()
    made = scripted.made
    lag = made.request.stepper.controller.lag
    assert made.march.state.norms.maxlen == lag + 2
    summary = scripted.run(export=None).observations[-1].summary
    assert summary.n_steps > lag + 2
    assert len(made.march.state.norms) == lag + 2
    with session_of(made, scripted.problem, scripted.request):
        assert len(made.march.state.norms) == 0


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
    "user_rate",
)
"""The separable leaf's fields; schema 15 appended ``kappa_sq``, 31 ``kperp_rate``, 45
``user_rate``."""

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

MARCH_POLICY_FIELDS = (
    "order",
    "variable_order",
    "safety",
    "max_growth",
    "min_factor",
    "rejection_growth",
    "hysteresis",
    "restart_after",
    "collar_stability",
    "real_intervals",
    "collar_rate",
    "carrier",
    "carrier_cap",
    "max_calls",
    "equal_approach",
    "interval_cap",
    "dz_min",
    "dz_max",
    "stretch",
    "z_end",
    "save_planes",
    "hard_kinks",
    "breakpoints",
    "landing_ulps",
    "interpolate",
    "quiet_interval",
)
"""Every number the Adams march's algorithm reads, the record a native march takes
(schema 39, SEP 39 section 4.2)."""


def test_a_trials_bundles_taken_together_are_kept_until_the_next_hoist(driver):
    """``stage_bundles`` reads the tables once and ``stage_scalars`` takes the kept.

    The bundles equal the per-position ones bit for bit; a later stage at the same
    anchor and position is the kept bundle itself, and the next step's hoist drops
    them, so nothing a later step reads was taken for an earlier one.
    """
    made = driver().made
    z0, dz = LENGTH / 4, LENGTH / 8
    positions = [z0 + dz, z0, z0 + 0.5 * dz, z0 + dz]
    workspace = made.workspace
    workspace.hoist(z0)
    together = workspace.stage_bundles(z0, positions)
    assert together[0] is together[3]
    for z1, bundle in zip(positions, together, strict=True):
        alone = scalars(made.workspace.problem, z0, z1)
        assert bundle.z0 == z0 and bundle.z1 == z1
        assert np.array_equal(present(bundle.phi_delta), present(alone.phi_delta))
        assert np.array_equal(present(bundle.prefactor), present(alone.prefactor))
        assert bundle.density == alone.density
        assert workspace.stage_scalars(z0, z1) is bundle
    # Asked again at the same anchor, nothing is read: the kept bundles come back.
    again = workspace.stage_bundles(z0, positions[1:3])
    assert again[0] is together[1] and again[1] is together[2]
    workspace.hoist(z0 + dz)
    assert workspace.stage_scalars(z0, z0 + dz) is not together[0]
    made.close()


def test_the_bundles_a_compiled_backend_packs_are_frozen():
    """The stage bundles and the march policy are the seam's literal.

    A compiled backend is built against these names and shapes, and its build stays
    valid across solprop commits because this test, not a version pin, guards them: a
    change here is a change of the seam, which rewrites these literals and adds a
    release-note fragment in the same commit.
    """
    assert tuple(f.name for f in dataclasses.fields(StageTerms)) == STAGE_TERMS_FIELDS
    assert (
        tuple(f.name for f in dataclasses.fields(StageScalars)) == STAGE_SCALARS_FIELDS
    )
    assert tuple(f.name for f in dataclasses.fields(MarchPolicy)) == MARCH_POLICY_FIELDS


BACKEND_MEMBERS = (
    "available_device_memory",
    "close",
    "closed",
    "edge_memory",
    "memory_ledger",
    "metadata",
    "prepare",
    "request",
    "timing_probe",
)
"""The :class:`~solprop.backends.protocol.Backend` protocol's methods and properties,
complete (SEP 52 section 5.1): its one attribute is ``name``. A change here changes the
seam, in the same commit."""

REFERENCE_ATTRIBUTES = ("name",)
"""The class attributes the NumPy reference backend sets."""


def test_the_backend_surface_the_reference_implements_is_frozen():
    members = {
        name for name in dir(Backend) if not name.startswith("_") and name != "name"
    }
    assert members == set(BACKEND_MEMBERS)
    assert isinstance(NumpyRefBackend.__dict__["request"], property)
    for name in (*BACKEND_MEMBERS, *REFERENCE_ATTRIBUTES):
        assert hasattr(NumpyRefBackend, name), name


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


def test_the_workspace_propagator_is_the_operator_s_flow(driver):
    # The propagator is the shared one, exp(Phi(b) - Phi(a)); a dense output whose
    # engine holds no accepted step's extension refuses an observation.
    made = driver().made
    propagator = made.workspace.propagator(0.0, LENGTH / 4)
    expected = np.exp(delta_phi(made.workspace.problem.linear, 0.0, LENGTH / 4))
    np.testing.assert_allclose(
        propagator.reshape(-1), np.broadcast_to(expected, propagator.shape).reshape(-1)
    )
    dense = RungeKuttaDense(made.workspace, made._runge_kutta(made.workspace))
    with pytest.raises(BackendError, match="no accepted step is held"):
        dense.observe(LENGTH / 8)


# --- the descriptor (SEP 47 section 6.2) -------------------------------------------

PROPAGATION_PROBLEM_FIELDS = (
    "grid_kind",
    "shape",
    "flags",
    "grids",
    "linear",
    "responses",
    "initial",
    "route",
    "split_crossing",
)
"""The fields of the one descriptor that crosses into a backend, in order: the
numerical meaning alone, whose run's policy is the request's (SEP 47 section 5.2)."""

RUN_REQUEST_FIELDS = (
    "integration",
    "error_control",
    "replay",
    "observation",
    "boundary",
    "history",
    "snapshots",
)
"""The fields of the run request a backend is built with, in order."""

ROUTE_VARIANT_FIELDS = {
    "ModalRoute": ("transform", "tdse"),
    "GeneralRoute": ("transform",),
    "SpatialRoute": ("kind", "transform", "flux", "transverse"),
}
"""The closed route union: each variant and its fields, in order.

A route is added or changed here and in `kernel-contract.md` together (the contract's
extension procedure, section 10)."""

TDSE_SOURCE_FIELDS = ("descriptor", "current_numerator")
SPATIAL_KINDS = ("radial", "cartesian", "planar", "line")


def test_the_descriptor_and_its_route_union_are_frozen():
    assert (
        tuple(field.name for field in dataclasses.fields(PropagationProblem))
        == PROPAGATION_PROBLEM_FIELDS
    )
    assert (
        tuple(field.name for field in dataclasses.fields(RunRequest))
        == RUN_REQUEST_FIELDS
    )
    assert typing.get_args(routes.Route.__value__) == tuple(
        getattr(routes, name) for name in ROUTE_VARIANT_FIELDS
    )
    for name, expected in ROUTE_VARIANT_FIELDS.items():
        variant = getattr(routes, name)
        assert tuple(field.name for field in dataclasses.fields(variant)) == expected
    assert (
        tuple(field.name for field in dataclasses.fields(routes.TDSESource))
        == TDSE_SOURCE_FIELDS
    )
    assert routes.SPATIAL_KINDS == SPATIAL_KINDS
