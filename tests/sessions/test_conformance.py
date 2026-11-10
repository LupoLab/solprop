# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The lifecycle conformance suite of SEP 47: one contract, every session.

Each test states the section of SEP 47 it checks and runs once per executor of
``sessions.executors`` unless its outcome needs a script (failures, warnings,
rejections at a chosen trial), when it runs on the scripted executor. Nothing here
compares two executors: the reference is not the native session's oracle, nor the
other way round (section 6.6), and each is held to the contract alone. Equalities are
exact because they are identities of the protocol -- event order, cursor accounting,
lossless copies and yields that must be transport no-ops (sections 5.1 and 7.1).
"""

import itertools
import threading
import time
import warnings

import numpy as np
import pytest

from sessions.executors import (
    CONSUMER,
    COUPLED,
    COUPLED_KINDS,
    DENSE,
    FACTOR,
    IDENTITY,
    LENGTH,
    MONITOR,
    PLANES,
    RECIPES,
    STATISTICS,
    STORED,
    TRANSVERSE,
    Executor,
    Run,
    ScriptedError,
    ScriptedWarning,
    assert_same_run,
    coupled_flow,
    drive,
    edged,
    soliton,
    spy_monitor,
    window,
)
from solprop.backends.events import (
    Eligibility,
    EventBatch,
    ObservationKind,
    ObservationReady,
    Outcome,
    Progress,
    ServiceKind,
    ServiceRequired,
    Terminal,
    join_attempts,
)
from solprop.backends.session import (
    RunSession,
    SessionError,
    SessionState,
    SnapshotReceipt,
    StopToken,
)
from solprop.exceptions import BackendError, BoundaryError, StopRequested
from solprop.monitor import BoundaryCheck
from solprop.stats import StatsSpec
from solprop.stepper.multistep import Multistep

# The soliton's 1 ps window is tight for a monitor's guards; the window is not tested.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

FAIL_AT = 8
"""A scripted trial past the first plane, which the script reaches in five."""


def _next_observation(session: RunSession) -> tuple[EventBatch, ObservationReady]:
    """Advance to the next batch that carries an observation, leaving it pending.

    Batches before it -- progress, should a session yield on its bounds -- are
    acknowledged.

    Parameters
    ----------
    session : solprop.backends.session.RunSession
        A ready session.

    Returns
    -------
    tuple
        The pending batch and its observation.
    """
    while True:
        batch = session.advance()
        found = [event for event in batch.events if isinstance(event, ObservationReady)]
        if found:
            return batch, found[-1]
        assert batch.terminal is None, "the run ended before another observation"
        session.acknowledge(batch.receipt)


def _past_launch(session: RunSession) -> tuple[EventBatch, ObservationReady]:
    """Acknowledge the launch plane and advance to the first observation after it.

    Parameters
    ----------
    session : solprop.backends.session.RunSession
        A session that has not advanced.

    Returns
    -------
    tuple
        The pending batch and its observation.
    """
    batch, launch = _next_observation(session)
    assert launch.kind is ObservationKind.LAUNCH
    session.acknowledge(batch.receipt)
    return _next_observation(session)


def _terminal_field(store: dict[str, np.ndarray]):
    """Return a ``between`` hook that exports a terminal batch's field into `store`.

    Parameters
    ----------
    store : dict
        Receives ``"field"``, the identity export of the terminal ticket.

    Returns
    -------
    callable
        The hook.
    """

    def between(session: RunSession, batch: EventBatch) -> None:
        terminal = batch.terminal
        if terminal is not None and terminal.ticket is not None:
            store["field"] = np.array(session.export(terminal.ticket, IDENTITY))

    return between


def _deterministic(executor: Executor) -> None:
    """Skip a bitwise invariance test where the executor's arithmetic may vary.

    Parameters
    ----------
    executor : sessions.executors.Executor
        The executor.
    """
    if not executor.deterministic:
        pytest.skip(
            "a native run on several threads may vary its reduction order; the "
            "default lane runs one (SOLPROP_NATIVE_THREADS=1)"
        )


# --------------------------------------------------------------------- receipts


def test_a_stale_repeated_or_foreign_receipt_is_refused(start):
    """SEP 47 section 5.3: a receipt is good once, on its own session, while pending."""
    problem = soliton()
    one, other = start(problem).session, start(problem).session
    first = one.advance()
    one.acknowledge(first.receipt)
    with pytest.raises(SessionError):
        one.acknowledge(first.receipt)  # repeated
    second = one.advance()
    assert second.receipt.sequence > first.receipt.sequence
    with pytest.raises(SessionError):
        one.acknowledge(first.receipt)  # stale
    foreign = other.advance()
    with pytest.raises(SessionError):
        one.acknowledge(foreign.receipt)
    with pytest.raises(SessionError):
        other.acknowledge(second.receipt)
    # A refused acknowledgment releases nothing: both batches are still pending.
    assert one.status().state is SessionState.PENDING
    one.acknowledge(second.receipt)
    other.acknowledge(foreign.receipt)


def test_advancing_across_a_pending_batch_is_refused(start):
    """SEP 47 section 5.3: no barrier is crossed before its batch is acknowledged."""
    session = start(soliton()).session
    batch = session.advance()
    before = session.status()
    with pytest.raises(SessionError):
        session.advance()
    assert session.status() == before
    session.acknowledge(batch.receipt)
    session.advance()


def test_a_finished_session_refuses_to_advance(start):
    """SEP 47 section 5.2: nothing advances past a terminal outcome."""
    started = start(soliton())
    run = drive(started)
    assert run.terminal.outcome is Outcome.COMPLETED
    session = started.session
    assert session.finished
    assert session.status().state is SessionState.FINISHED
    with pytest.raises(SessionError):
        session.advance()
    with pytest.raises(SessionError):
        session.acknowledge(run.batches[-1].receipt)


def test_closing_is_idempotent(start):
    """SEP 47 section 5.2: close, and scoped ownership, may be repeated."""
    session = start(soliton()).session
    with session as scoped:
        assert scoped is session
        scoped.advance()
    assert session.closed
    session.close()
    session.close()
    assert session.status().state is SessionState.CLOSED


def test_a_closed_session_refuses_advance_acknowledge_and_export(start):
    """SEP 47 section 5.2: a closed session releases its pending exports."""
    session = start(soliton()).session
    batch = session.advance()
    (launch,) = batch.events
    assert isinstance(launch, ObservationReady)
    session.close()
    with pytest.raises(SessionError):
        session.advance()
    with pytest.raises(SessionError):
        session.acknowledge(batch.receipt)
    with pytest.raises(SessionError):
        session.export(launch.ticket, IDENTITY)
    with pytest.raises(SessionError):
        session.request_snapshot()


# ---------------------------------------------------------------------- tickets


def test_a_ticket_expires_when_its_batch_is_acknowledged(start):
    """SEP 47 section 5.4: a ticket is valid while its batch is pending, no longer."""
    session = start(soliton()).session
    batch, plane = _past_launch(session)
    session.export(plane.ticket, CONSUMER)
    session.acknowledge(batch.receipt)
    for recipe in RECIPES:
        with pytest.raises(SessionError):
            session.export(plane.ticket, recipe)


def test_a_ticket_of_another_session_is_refused(start):
    """SEP 47 section 5.4: an export is keyed by its own session's state generation."""
    problem = soliton()
    one, other = start(problem).session, start(problem).session
    _, mine = _past_launch(one)
    _, theirs = _past_launch(other)
    with pytest.raises(SessionError):
        one.export(theirs.ticket, IDENTITY)
    with pytest.raises(SessionError):
        other.export(mine.ticket, IDENTITY)


def test_repeated_exports_of_a_ticket_have_the_same_bytes(start):
    """SEP 47 section 5.4: one payload per generation and recipe, however often asked.

    The recipes are exported in turn and then again in both orders, into new arrays
    and into destinations; every export of a recipe has the bytes of its first. The
    optics conversion is checked against its definition, ``(-1)**k conj(.)``, which is
    exact.
    """
    session = start(soliton()).session
    _, plane = _past_launch(session)
    first = {
        recipe: np.array(session.export(plane.ticket, recipe)) for recipe in RECIPES
    }
    for recipe in (*RECIPES, *reversed(RECIPES)):
        again = session.export(plane.ticket, recipe)
        assert again.tobytes() == first[recipe].tobytes(), recipe
        out = np.empty_like(first[recipe])
        assert session.export(plane.ticket, recipe, out=out) is out
        assert out.tobytes() == first[recipe].tobytes(), recipe
    shift = np.where(np.arange(first[IDENTITY].shape[-1]) % 2, -1.0, 1.0)
    np.testing.assert_array_equal(first[STORED], shift * np.conj(first[CONSUMER]))


def test_an_unfit_export_destination_is_refused(start):
    """SEP 47 section 5.4: a destination is validated before the session fills it.

    Wrong shape, wrong dtype, a strided or read-only array and one whose elements
    overlap each other are refused, each by what is wrong with it; so is writing into
    an array the session lent rather than handed over. A refusal changes nothing the
    session exports next.
    """
    session = start(soliton()).session
    _, plane = _past_launch(session)
    expected = np.array(session.export(plane.ticket, IDENTITY))
    shape = expected.shape
    readonly = np.empty(shape, np.complex128)
    readonly.setflags(write=False)
    # Each unfit destination, with what its refusal names.
    unfit = [
        ("has shape", np.empty((*shape[:-1], shape[-1] + 1), np.complex128)),
        ("has dtype complex64", np.empty(shape, np.complex64)),
        (
            "C-contiguous",
            np.empty((*shape[:-1], 2 * shape[-1]), np.complex128)[..., ::2],
        ),
        ("read-only", readonly),
        # Elements that overlap one another are not contiguous either.
        (
            "C-contiguous",
            np.lib.stride_tricks.as_strided(
                np.zeros(1, np.complex128), shape=shape, strides=(0,) * len(shape)
            ),
        ),
    ]
    for recipe in (IDENTITY, STORED):
        for named, out in unfit:
            with pytest.raises(BackendError, match=named):
                session.export(plane.ticket, recipe, out=out)
                pytest.fail(f"a destination that {named} was accepted")
    held = session.export(plane.ticket, IDENTITY)
    if not held.flags.writeable:
        # The session lent its own array: it is not a destination.
        with pytest.raises(BackendError, match="read-only"):
            session.export(plane.ticket, STORED, out=held)
    assert session.export(plane.ticket, IDENTITY).tobytes() == expected.tobytes()


# -------------------------------------------------------------------- the order


def test_events_follow_the_committed_order(start, executor):
    """SEP 47 section 5.3, the ordering table, start to final endpoint.

    The launch plane comes first and is committed before the boundary monitor is
    attached (a spy monitor logs its launch against the test's acknowledgment, or a
    native monitor's record holds no launch yet); save planes follow in order, with a
    step plane at every third accepted step; the terminal event comes last. Every
    committed step is checked or skipped by the monitor once, in order, and each
    observation closes a monitor row. Plane statistics and periodic rows carry no batch
    axis, rows are taken every second accepted step at an accepted endpoint, and a
    batch holds at most one field-bearing event.
    """
    executor.require(MONITOR)
    log: list[tuple[str, float]] = []
    problem = soliton(step_planes=3, period=2)
    started = start(problem, problem.request.for_run(monitor=spy_monitor(log)))
    session = started.session
    launch = session.advance()
    (first,) = launch.events
    assert isinstance(first, ObservationReady)
    assert (first.kind, first.plane_index, first.z) == (ObservationKind.LAUNCH, -1, 0.0)
    assert first.history.n_steps == first.step_stats.n_rows == 0
    assert not log, "the monitor was attached before the launch plane was committed"
    if executor.native_monitor:
        record = session.monitor_record()
        assert record is not None
        assert record["launch_energy"] is None and record["planes"]["z"] == [], (
            "the monitor was attached before the launch plane was committed"
        )
    log.append(("acknowledged", first.z))
    session.acknowledge(launch.receipt)
    rest = drive(started)
    run = Run([launch, *rest.batches], [], rest.status)

    record = session.monitor_record()
    assert record is not None
    ends = run.accepted_ends()
    if executor.native_monitor:
        # The native monitor owns the record; the request's solprop monitor is unused.
        assert not log[1:]
        assert record["launch_energy"] is not None
        assert record["steps"] + record["skipped"] == ends.size
    else:
        assert log[:2] == [("acknowledged", 0.0), ("launch", 0.0)]
        assert {kind for kind, _ in log[2:]} == {"step"}
        watched = [z for _, z in log[2:]]
        assert watched == ends.tolist()
        assert watched[0] > 0.0 and watched[-1] <= LENGTH
    # The launch's row, then one row per observation, each at its position.
    observed = [event.z for event in run.observations]
    assert record["planes"]["z"] == observed

    _assert_committed_order(run)


@pytest.mark.parametrize(
    ("check", "outcome", "noise"),
    [
        (BoundaryCheck(transverse_budget=10.0), Outcome.COMPLETED, None),
        (BoundaryCheck(), Outcome.FAILED, None),
        (BoundaryCheck(transverse_budget=10.0), Outcome.COMPLETED, "input"),
    ],
    ids=["relaxed", "stopped", "relaxed-noisy"],
)
def test_a_transverse_edge_is_read_in_the_committed_order(
    start, executor, check, outcome, noise
):
    """SEP 47 section 5.3 with SEP 40 section 5.3: a free-space run's transverse edge.

    The edge is launched after the launch plane and read where its spacing falls due,
    the steps held to that spacing; it is read at every plane it has not read, and its
    record joins the monitor's. Relaxed, the re-review's packet is read to the end at
    most a quarter of the run apart (the backstop's largest); under the default budget
    the edge stops it at a read, a failure at that committed endpoint naming the edge.
    With the input noise, the launch takes the draw's allowance, which the record keeps.
    """
    executor.require(TRANSVERSE)
    if noise is not None and executor.name.endswith("low-storage"):
        pytest.skip(
            "the replay's half-millimetre steps exceed the split collar's crossing "
            "limit the input noise sets at the launch, which refuses the run there"
        )
    problem, request, context = edged(check, noise=noise)
    assert problem.initial.noise_model == (noise or "none")
    started = start(problem, request, context=context)
    launch = started.session.advance()
    (first,) = launch.events
    assert isinstance(first, ObservationReady)
    assert first.kind is ObservationKind.LAUNCH
    if executor.native_monitor:
        record = started.session.monitor_record()
        assert record is not None
        assert record["transverse"]["launch_energy"] is None, (
            "the edge was launched before the launch plane was committed"
        )
    started.session.acknowledge(launch.receipt)
    rest = drive(started)
    run = Run([launch, *rest.batches], [], rest.status)
    record = started.session.monitor_record()
    assert record is not None
    edge = record["transverse"]
    assert edge["launch_energy"] is not None and edge["reads"] > 0
    assert (edge["noise_allowance"] > 0.0) is (noise is not None)
    assert run.terminal.outcome is outcome
    length = request.stepper.save_planes[-1]
    if outcome is Outcome.COMPLETED:
        # A replayed uniform schedule's steps are longer than the edge's read spacing,
        # which the edge reports as coarse (SEP 40 section 5.3); adaptive steps are
        # held to the spacing.
        replayed = executor.name.endswith("low-storage")
        assert edge["transverse_status"] == ("coarse" if replayed else "clear")
        # Read at most a quarter of the run apart, or a replayed step apart.
        longest = max(
            float(history.dz.max()) for history, _ in run.chunks if history.n_steps
        )
        gap = max(length / 4, longest) if replayed else length / 4
        assert edge["largest_gap"] <= gap * (1 + 1e-9)
        planes = [e for e in run.observations if e.kind is ObservationKind.PLANE]
        assert [event.z for event in planes] == list(request.stepper.save_planes)
    else:
        assert edge["transverse_status"] == "stopped"
        assert isinstance(run.terminal.error, BoundaryError)
        assert "reached the transverse edge" in str(run.terminal.error)
        assert run.terminal.eligibility is Eligibility.ELIGIBLE
        assert run.terminal.z < length
    for batch in run.batches:
        assert (
            sum(getattr(event, "ticket", None) is not None for event in batch.events)
            <= 1
        )
    assert run.attempts == run.status.attempts > 0


def test_events_follow_the_committed_order_without_a_monitor(start):
    """SEP 47 section 5.3, the ordering table, start to final endpoint, unwatched.

    The order of :func:`test_events_follow_the_committed_order` for an executor that
    runs no monitor: the launch plane first, save planes in order with a step plane at
    every third accepted step, the terminal event last; statistics and periodic rows
    without a batch axis, rows every second accepted step at an accepted endpoint, and
    at most one field-bearing event a batch.
    """
    problem = soliton(step_planes=3, period=2)
    started = start(problem)
    session = started.session
    launch = session.advance()
    (first,) = launch.events
    assert isinstance(first, ObservationReady)
    assert (first.kind, first.plane_index, first.z) == (ObservationKind.LAUNCH, -1, 0.0)
    assert first.history.n_steps == first.step_stats.n_rows == 0
    session.acknowledge(launch.receipt)
    rest = drive(started)
    run = Run([launch, *rest.batches], [], rest.status)
    _assert_committed_order(run)


PERIODIC = (
    "omega0",
    "energy",
    "energy_window",
    "peak_power",
    "peak_power_window",
    "peak_power_total",
    "fwhm_t",
    "fwhm_t_outer",
)
"""Every periodic-capable statistic of the soliton's route."""


def test_every_periodic_statistic_s_row_is_its_plane_s(start, executor):
    """SEP 47 section 6.4: rows and planes use one implementation of each statistic.

    With a row and a step plane at every accepted step, each row is taken at its
    observation's committed state, and equals that plane's statistics: bitwise where
    repeated runs are, and every periodic-capable statistic of the route is a row.
    """
    executor.require(STATISTICS)
    grid = window()
    spec = StatsSpec(
        names=PERIODIC,
        periodic=PERIODIC,
        period=1,
        energy_windows=np.array([[0.99 * grid.omega0, 1.01 * grid.omega0]]),
    )
    problem, run_request = soliton(step_planes=1)
    run = drive(start(problem, run_request.for_run(stats=spec)))
    coincident = 0
    for event in run.observations[1:]:
        rows = event.step_stats
        if rows.n_rows == 0 or rows.z[-1] != event.z:
            # A plane at a position already observed carries no new row.
            continue
        assert rows.values.keys() == set(PERIODIC)
        coincident += 1
        for name in PERIODIC:
            row, plane = rows.values[name][-1], event.stats[name]
            assert row.shape == plane.shape, name
            if executor.deterministic:
                np.testing.assert_array_equal(row, plane, name)
            else:
                np.testing.assert_allclose(row, plane, rtol=1e-12, err_msg=name)
    assert coincident > len(PLANES)


def _assert_committed_order(run: Run) -> None:
    """Assert a step-plane soliton run's events follow the committed order.

    Parameters
    ----------
    run : sessions.executors.Run
        The run of ``soliton(step_planes=3, period=2)``, launch batch included.
    """
    terminals = [event for event in run.events if isinstance(event, Terminal)]
    assert terminals == [run.batches[-1].events[-1]]
    assert run.terminal.outcome is Outcome.COMPLETED
    assert (run.terminal.z, run.terminal.eligibility) == (LENGTH, Eligibility.ELIGIBLE)
    for batch in run.batches:
        assert (
            sum(getattr(event, "ticket", None) is not None for event in batch.events)
            <= 1
        )

    observations = run.observations
    assert [event.kind for event in observations].count(ObservationKind.LAUNCH) == 1
    planes = [event for event in observations if event.kind is ObservationKind.PLANE]
    assert [event.plane_index for event in planes] == list(range(len(PLANES)))
    assert [event.z for event in planes] == list(PLANES)
    positions = [event.z for event in observations]
    assert positions == sorted(positions) and len(set(positions)) == len(positions)
    _check_chunks(run, observations[-1].stats["energy"].shape, period=2)
    assert run.attempts == run.status.attempts > 0
    assert sum(int((~history.accepted).sum()) for history in run.histories) == (
        run.status.rejected
    )


def _check_chunks(run: Run, shape: tuple[int, ...], period: int) -> None:
    """Check every event's chunk of a run with step planes every third step.

    A row's step indexes its own chunk's attempts, rows fall on every `period`-th
    accepted step, step planes on every third, and statistics carry no batch axis.

    Parameters
    ----------
    run : sessions.executors.Run
        The run.
    shape : tuple
        The ``energy`` statistic's shape at a plane.
    period : int
        The rows' period, accepted steps.
    """
    accepted = 0
    for event, (history, rows) in zip(run.events, run.chunks, strict=True):
        before = accepted
        accepted += int(history.accepted.sum())
        if isinstance(event, ObservationReady) and event.kind is not (
            ObservationKind.LAUNCH
        ):
            assert event.stats.keys() == {"energy", "peak_power"}
            assert event.stats["energy"].shape == shape
        if isinstance(event, ObservationReady) and event.kind is ObservationKind.STEP:
            assert accepted % 3 == 0
            low = PLANES[event.plane_index - 1] if event.plane_index else 0.0
            assert low < event.z < PLANES[event.plane_index]
        if rows.n_rows:
            assert rows.values.keys() == {"energy"}
            assert rows.values["energy"].shape == (rows.n_rows, *shape)
            np.testing.assert_array_equal(history.z[rows.step], rows.z)
            assert history.accepted[rows.step].all()
            counted = before + np.cumsum(history.accepted)[rows.step]
            assert (counted % period == 0).all()


def test_attempt_records_add_up_to_the_status(start):
    """SEP 47 section 7.1: accurate actual-work totals, each attempt reported once."""
    started = start(soliton(step_planes=4))
    run = drive(started)
    assert run.attempts == run.status.attempts == started.session.status().attempts
    ends = run.accepted_ends()
    assert (np.diff(ends) > 0).all() and ends[-1] == LENGTH


@pytest.mark.medium
@pytest.mark.filterwarnings("ignore:the temporal absorber attenuates")
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.filterwarnings("ignore::solprop.exceptions.GrazingIncidenceWarning")
@pytest.mark.parametrize("kind", COUPLED_KINDS)
def test_a_wide_or_off_route_coupled_flow_keeps_the_contract(start, executor, kind):
    """SEP 47 section 6.3 (phase 4): the coupled flows the session took over.

    A 17-channel block and a block off its eigenbasis route, ten fixed Adams steps:
    the run completes at its end with every step accepted and recorded once, the
    planes are observed in order with finite fields, and a repeated native run is the
    same to the bit where the executor is deterministic.
    """
    executor.require(COUPLED)
    problem, request, context = coupled_flow(kind)
    schedule = request.stepper.schedule
    assert schedule is not None
    # The repeat is the native executors' to show; the reference's numpy run of a
    # 17-channel block is long, and its repeatability is the suite's other tests'.
    repeated = executor.deterministic and executor.name.startswith("native")
    runs = []
    for _ in range(2 if repeated else 1):
        started = start(problem, request, context=context)
        runs.append(drive(started, export=IDENTITY))
    run = runs[0]
    assert run.terminal.outcome is Outcome.COMPLETED
    assert run.attempts == run.status.attempts == len(schedule.steps)
    np.testing.assert_allclose(
        run.accepted_ends(), np.cumsum(schedule.steps), rtol=0.0, atol=1e-18
    )
    positions = [event.z for event in run.observations]
    assert positions == sorted(positions)
    assert len(run.fields) == len(run.observations) >= 2
    assert all(np.all(np.isfinite(field)) for field in run.fields)
    for again in runs[1:]:
        assert_same_run(again, run)


@pytest.mark.parametrize("work_limit", [1, 3, 10**6])
def test_every_attempt_and_row_is_delivered_exactly_once(start, work_limit):
    """SEP 47 section 5.3: event storage is bounded; chunks are delivered once.

    Each event, a yield's progress as much as a plane, a snapshot or the terminal
    event, delivers the attempts since the previous event. Taken in order the chunks
    are every attempt the listener heard, each once, and every periodic row, each
    once; a yield bound of one attempt yields progress events that carry them.
    """
    holder: list[RunSession] = []
    heard: list[tuple[float, float, bool]] = []

    def listener(z: float, dz: float, accepted: bool) -> None:
        heard.append((z, dz, accepted))
        if len(heard) % 5 == 0:
            holder[0].request_snapshot()

    started = start(soliton(step_planes=3, period=1), listener=listener)
    holder.append(started.session)
    run = drive(started, work_limit=work_limit)
    history, rows = join_attempts(run.chunks)
    np.testing.assert_array_equal(history.z, [z for z, _, _ in heard])
    np.testing.assert_array_equal(history.dz, [dz for _, dz, _ in heard])
    np.testing.assert_array_equal(history.accepted, [ok for _, _, ok in heard])
    assert run.attempts == run.status.attempts == len(heard)
    # Period one: a row at every accepted endpoint, each once, at its own attempt.
    np.testing.assert_array_equal(rows.z, history.z[history.accepted])
    np.testing.assert_array_equal(rows.step, np.flatnonzero(history.accepted))
    assert any(isinstance(event, ServiceRequired) for event in run.events)
    progress = [event for event in run.events if isinstance(event, Progress)]
    if work_limit == 1:
        assert progress and all(event.history.n_steps for event in progress)
    _check_chunks(run, run.observations[-1].stats["energy"].shape, period=1)


# ------------------------------------------------------------------- field-free


def test_a_plane_nobody_exports_transfers_no_field(start):
    """SEP 47 section 5.4: field-free events transfer no field; one export, one read."""
    started = start(soliton(step_planes=3))
    if started.field_reads is None:
        pytest.skip("the executor cannot count its field transfers")
    drive(started)
    assert started.field_reads() == 0

    started = start(soliton(step_planes=3))
    assert started.field_reads is not None

    def export_the_second_plane(session: RunSession, batch: EventBatch) -> None:
        for event in batch.events:
            if (
                isinstance(event, ObservationReady)
                and event.kind is ObservationKind.PLANE
                and event.plane_index == 1
            ):
                one = session.export(event.ticket, CONSUMER)
                two = session.export(event.ticket, CONSUMER)
                assert one.tobytes() == two.tobytes()

    drive(started, between=export_the_second_plane)
    assert started.field_reads() == 1


# ------------------------------------------------------------------- retention


def test_exported_arrays_are_never_written_again(start):
    """SEP 47 section 5.4: delivered arrays are detached and owned by the consumer.

    Every field-bearing event is exported with every recipe, into new arrays and into
    destinations, and all of them are kept. None changes as the run goes on, after it
    ends or after the session is closed, and, where the state is a host array, none
    shares memory with it.
    """
    started = start(soliton(step_planes=3))
    kept: list[tuple[np.ndarray, bytes]] = []

    def live_overlap(held: np.ndarray) -> bool:
        if started.live_state is None:
            return False
        state = started.live_state()
        return state is not None and np.shares_memory(held, state)

    def keep_every_export(session: RunSession, batch: EventBatch) -> None:
        for event in batch.events:
            ticket = getattr(event, "ticket", None)
            if ticket is None:
                continue
            for recipe in RECIPES:
                held = session.export(ticket, recipe)
                out = session.export(ticket, recipe, out=np.empty_like(held))
                kept.extend((array, array.tobytes()) for array in (held, out))
        for held, original in kept:
            assert held.tobytes() == original
            assert not live_overlap(held)

    drive(started, between=keep_every_export)
    assert len(kept) > 8 * len(PLANES)
    for held, original in kept:
        assert held.tobytes() == original
    started.session.close()
    for held, original in kept:
        assert held.tobytes() == original


# ------------------------------------------------------------------- failures


def test_a_failed_trial_ends_failed_at_an_eligible_committed_state(start_scripted):
    """SEP 47 section 5.5: a failure reports the last committed state, resumable.

    The terminal event is the failure, eligible, with a ticket for the field of the
    last accepted step -- the launch times ``FACTOR**k`` after ``k`` accepted steps --
    and the attempts since the last plane; nothing is lost from the totals.
    """
    started = start_scripted(soliton(), fail_at=FAIL_AT, reject_at=(FAIL_AT - 2,))
    held: dict[str, np.ndarray] = {}
    run = drive(started, export=IDENTITY, between=_terminal_field(held))
    terminal = run.terminal
    assert terminal.outcome is Outcome.FAILED
    assert isinstance(terminal.error, ScriptedError)
    assert terminal.eligibility is Eligibility.ELIGIBLE
    assert run.observations[-1].kind is ObservationKind.PLANE
    assert run.attempts == run.status.attempts == FAIL_AT - 1
    assert run.status.rejected == 1
    observed = sum(event.history.n_steps for event in run.observations)
    assert terminal.history is not None
    assert terminal.history.n_steps == FAIL_AT - 1 - observed
    ends = run.accepted_ends()
    assert terminal.z == ends[-1] == run.status.z
    np.testing.assert_array_equal(held["field"], run.fields[0] * FACTOR**ends.size)
    with pytest.raises(SessionError):
        started.session.advance()


def test_an_in_flight_failure_poisons_the_state(start_scripted):
    """SEP 47 section 5.5: a step abandoned in flight claims no checkpoint.

    The terminal event is the failure, poisoned, with no ticket; the attempts before
    the abandoned one are still accounted.
    """
    started = start_scripted(soliton(), fail_at=FAIL_AT, in_flight=True)
    run = drive(started)
    terminal = run.terminal
    assert terminal.outcome is Outcome.FAILED
    assert isinstance(terminal.error, ScriptedError)
    assert terminal.eligibility is Eligibility.POISONED
    assert terminal.ticket is None
    assert run.status.attempts == FAIL_AT - 1
    assert run.attempts == FAIL_AT - 1


def test_a_warning_raised_as_an_error_fails_at_its_committed_endpoint(start_scripted):
    """SEP 47 section 5.5: a warning the filters promote is a barrier, then a failure.

    The warning arises after an accepted step's commit; the failure reports that
    endpoint, eligible, with its field.
    """
    warn_at = 6
    held: dict[str, np.ndarray] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ScriptedWarning)
        run = drive(
            start_scripted(soliton(), warn_at=warn_at),
            export=IDENTITY,
            between=_terminal_field(held),
        )
    terminal = run.terminal
    assert terminal.outcome is Outcome.FAILED
    assert isinstance(terminal.error, ScriptedWarning)
    assert terminal.eligibility is Eligibility.ELIGIBLE
    ends = run.accepted_ends()
    assert ends.size == warn_at and terminal.z == ends[-1]
    np.testing.assert_array_equal(held["field"], run.fields[0] * FACTOR**warn_at)
    assert run.attempts == run.status.attempts


def test_an_allowed_warning_changes_nothing(start_scripted):
    """SEP 47 section 5.5: a warning the filters allow resumes the run, unchanged."""
    baseline = drive(start_scripted(soliton(step_planes=3)), export=IDENTITY)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ScriptedWarning)
        run = drive(start_scripted(soliton(step_planes=3), warn_at=6), export=IDENTITY)
    assert [warning.category for warning in caught] == [ScriptedWarning]
    assert_same_run(run, baseline)


def test_rejected_trials_are_recorded_and_not_kept(start_scripted):
    """SEP 47 sections 5.3 and 7.1: a rejection records its work and keeps no state."""
    rejected = (3, 4, FAIL_AT)
    run = drive(start_scripted(soliton(), reject_at=rejected), export=IDENTITY)
    flags = np.concatenate([history.accepted for history in run.histories])
    assert np.flatnonzero(~flags).tolist() == [trial - 1 for trial in rejected]
    assert run.status.rejected == len(rejected)
    assert run.attempts == run.status.attempts
    # Every chunk counts, in delivery order: a deadline's yield delivers its own.
    fields = dict(zip(map(id, run.observations), run.fields, strict=True))
    accepted = 0
    for event in run.events:
        history = getattr(event, "history", None)
        if history is not None:
            accepted += int(history.accepted.sum())
        if id(event) in fields:
            np.testing.assert_array_equal(
                fields[id(event)], run.fields[0] * FACTOR**accepted
            )


# ------------------------------------------------------------------------ stops


def test_a_stop_before_the_first_step_ends_at_the_entrance(start):
    """SEP 47 section 5.5: a stop set before the first trial leaves the launch state."""
    token = StopToken()
    session = start(soliton(), stop=token).session
    batch, launch = _next_observation(session)
    entrance_field = np.array(session.export(launch.ticket, IDENTITY))
    session.acknowledge(batch.receipt)
    token.set()
    batch = session.advance()
    terminal = batch.terminal
    assert terminal is not None
    assert (terminal.outcome, terminal.z) == (Outcome.STOPPED, 0.0)
    assert terminal.eligibility is Eligibility.ELIGIBLE
    assert isinstance(terminal.error, StopRequested)
    assert terminal.history is None or terminal.history.n_steps == 0
    assert terminal.ticket is not None
    assert (
        session.export(terminal.ticket, IDENTITY).tobytes() == entrance_field.tobytes()
    )
    session.acknowledge(batch.receipt)
    assert session.status().attempts == 0
    assert session.status().outcome is Outcome.STOPPED


def test_a_stop_from_another_thread_ends_at_a_committed_endpoint(start, executor):
    """SEP 47 section 5.5: a token set from another thread stops after the attempt.

    The listener hands the fourth attempt to a thread that sets the token, and waits
    for it; the attempt in hand finishes, no further trial is made, and the terminal
    event is that committed endpoint, eligible.
    """
    stop_at = 4
    token = StopToken()
    heard: list[tuple[float, float, bool]] = []

    def listener(z: float, dz: float, accepted: bool) -> None:
        heard.append((z, dz, accepted))
        if len(heard) == stop_at:
            setter = threading.Thread(target=token.set)
            setter.start()
            setter.join()

    held: dict[str, np.ndarray] = {}
    run = drive(
        start(soliton(), stop=token, listener=listener),
        export=IDENTITY,
        between=_terminal_field(held),
    )
    terminal = run.terminal
    assert terminal.outcome is Outcome.STOPPED
    assert isinstance(terminal.error, StopRequested)
    assert terminal.eligibility is Eligibility.ELIGIBLE
    assert len(heard) == stop_at
    assert run.attempts == run.status.attempts == stop_at
    ends = run.accepted_ends()
    assert terminal.z == ends[-1] == run.status.z
    assert np.isfinite(held["field"]).all()
    if executor.scripted:
        np.testing.assert_array_equal(held["field"], run.fields[0] * FACTOR**ends.size)


def test_stop_token_listeners_fire():
    """SEP 47 section 5.5: an attached listener hears every set, from any thread."""
    token = StopToken()
    calls: list[str] = []
    detach = token.attach(lambda: calls.append("early"))
    assert calls == []
    setter = threading.Thread(target=token.set)
    setter.start()
    setter.join()
    assert token.is_set() and calls == ["early"]
    token.attach(lambda: calls.append("late"))
    assert calls == ["early", "late"]  # attached to a set token: called at once
    detach()
    detach()  # a second detach does nothing
    token.clear()
    assert not token.is_set()
    token.set()
    assert calls == ["early", "late", "late"]


# -------------------------------------------------------------------- snapshots


def test_a_snapshot_requested_at_the_entrance_waits_for_an_accepted_endpoint(start):
    """SEP 47 section 5.5: a snapshot latch is serviced at the next accepted endpoint.

    The entrance is not an accepted endpoint, so a request made there is serviced at
    the end of the first accepted step, delivering the attempts that reached it; the
    next plane delivers only the attempts after them, and the two chunks assemble into
    the plane's history.
    """
    started = start(soliton())
    session = started.session
    batch, _ = _next_observation(session)
    session.acknowledge(batch.receipt)
    assert session.request_snapshot() == SnapshotReceipt(1, coalesced=False)
    batch = session.advance()
    assert not any(isinstance(event, ObservationReady) for event in batch.events)
    (snapshot,) = [
        event for event in batch.events if isinstance(event, ServiceRequired)
    ]
    assert snapshot.kind is ServiceKind.SNAPSHOT
    assert snapshot.history is not None and snapshot.ticket is not None
    assert 0.0 < snapshot.z == snapshot.history.z[-1] < PLANES[0]
    assert snapshot.history.accepted.tolist() == [False] * (
        snapshot.history.n_steps - 1
    ) + [True]
    assert np.isfinite(session.export(snapshot.ticket, IDENTITY)).all()
    session.acknowledge(batch.receipt)
    batch, plane = _next_observation(session)
    assert plane.kind is ObservationKind.PLANE and plane.plane_index == 0
    assert plane.history.n_steps > 0
    assert plane.history.z[0] > snapshot.z  # nothing delivered twice
    history, _ = join_attempts(
        [
            (snapshot.history, snapshot.step_stats),
            (plane.history, plane.step_stats),
        ]
    )
    assert history.accepted.sum() == 1 + plane.history.accepted.sum()
    assert history.z[-1] == plane.z


def test_a_snapshot_between_planes_carries_the_attempts_since_the_last_plane(start):
    """SEP 47 section 5.5: a running checkpoint includes, without consuming, the
    attempts and periodic rows since the last saved plane.

    The request is latched at the first trial after the first plane (from the attempt
    listener, standing in for a control token set at that attempt). It is serviced at
    the next accepted endpoint, delivering the attempts and rows since the first plane;
    the second plane delivers the rest, and its assembled record is the two chunks.
    """
    holder: list[RunSession] = []
    heard: list[tuple[float, float, bool]] = []
    receipts: list[SnapshotReceipt] = []

    def listener(z: float, dz: float, accepted: bool) -> None:
        heard.append((z, dz, accepted))
        if not receipts and z - dz >= PLANES[0] * (1 - 1e-12):
            receipts.append(holder[0].request_snapshot())

    started = start(soliton(period=1), listener=listener)
    holder.append(started.session)
    run = drive(started)
    assert receipts == [SnapshotReceipt(1, coalesced=False)]
    snapshots = [event for event in run.events if isinstance(event, ServiceRequired)]
    assert [event.kind for event in snapshots] == [ServiceKind.SNAPSHOT]
    (snapshot,) = snapshots
    requested = next(
        index
        for index, (z, dz, _) in enumerate(heard)
        if z - dz >= PLANES[0] * (1 - 1e-12)
    )
    end = next(z for z, _, accepted in heard[requested:] if accepted)
    assert snapshot.z == end
    _, first, second = run.observations[:3]
    assert (first.plane_index, second.plane_index) == (0, 1)
    assert first.z < snapshot.z < second.z
    assert snapshot.history is not None and snapshot.step_stats is not None
    # The attempts up to the endpoint, that one included, less every chunk an earlier
    # event delivered (bounded yields deliver theirs too: exactly once).
    reached = [z for z, _, _ in heard].index(end) + 1
    chunks = run.chunks
    delivered = sum(
        history.n_steps for history, _ in chunks[: run.events.index(snapshot)]
    )
    assert snapshot.history.n_steps == reached - delivered
    rows = snapshot.step_stats
    assert rows.n_rows == int(snapshot.history.accepted.sum())
    np.testing.assert_array_equal(snapshot.history.z[rows.step], rows.z)
    # The plane after it delivers only what followed; assembled, the record is whole.
    assert second.history.z[0] > snapshot.z
    ((_, history, assembled),) = [
        entry for entry in run.assembled() if entry[0] is second
    ]

    def delivered_through(event: object) -> tuple[int, int]:
        """Attempts and rows every event up to and including `event` delivered."""
        upto = chunks[: run.events.index(event) + 1]
        return (
            sum(history.n_steps for history, _ in upto),
            sum(rows.n_rows for _, rows in upto),
        )

    first_end, first_rows = delivered_through(first)
    second_end, _ = delivered_through(second)
    np.testing.assert_array_equal(
        history.z, [z for z, _, _ in heard[first_end:second_end]]
    )
    # The snapshot's rows sit in the plane's assembled rows where it delivered them.
    _, snapshot_rows = delivered_through(snapshot)
    offset = snapshot_rows - rows.n_rows - first_rows
    np.testing.assert_array_equal(assembled.z[offset : offset + rows.n_rows], rows.z)
    assert run.attempts == run.status.attempts


def test_a_second_snapshot_request_coalesces(start):
    """SEP 47 section 5.5: coalescing an outstanding request is explicit."""
    session = start(soliton()).session
    batch, _ = _next_observation(session)
    session.acknowledge(batch.receipt)
    first = session.request_snapshot()
    second = session.request_snapshot()
    assert first == SnapshotReceipt(1, coalesced=False)
    assert second == SnapshotReceipt(1, coalesced=True)
    batch = session.advance()
    snapshots = [event for event in batch.events if isinstance(event, ServiceRequired)]
    assert len(snapshots) == 1
    session.acknowledge(batch.receipt)
    assert session.request_snapshot() == SnapshotReceipt(2, coalesced=False)


def test_a_snapshot_at_a_step_plane_follows_its_observation(start):
    """SEP 47 sections 5.3 and 5.5: if a plane coincides, its transaction precedes the
    snapshot; a latched request does not take the step plane's place.

    The request is latched at the trial that ends the third accepted step, a step
    plane: the step plane is observed there and the snapshot follows at the same
    position.
    """
    holder: list[RunSession] = []
    heard: list[tuple[float, float, bool]] = []

    def listener(z: float, dz: float, accepted: bool) -> None:
        heard.append((z, dz, accepted))
        if sum(attempt[2] for attempt in heard) == 3 and accepted:
            holder[0].request_snapshot()

    started = start(soliton(step_planes=3), listener=listener)
    holder.append(started.session)
    run = drive(started)
    events = [
        event
        for event in run.events
        if isinstance(event, ObservationReady | ServiceRequired)
    ]
    step = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, ObservationReady) and event.kind is ObservationKind.STEP
    )
    assert isinstance(events[step + 1], ServiceRequired)
    assert events[step + 1].z == events[step].z


# ----------------------------------------------------------------- dense output


DENSE_METHODS = {"runge-kutta": None, "adams": Multistep()}
"""The marches whose planes are interpolated: the default tableau's DOPRI5 extension
and the Adams march's own interpolant."""


def _delivered_ends(run: Run) -> list[np.ndarray]:
    """Return, for each event of a run, the accepted ends delivered up to it, inclusive.

    Parameters
    ----------
    run : sessions.executors.Run
        The run.

    Returns
    -------
    list of numpy.ndarray
        One array per event, in order.
    """
    ends: list[float] = []
    found: list[np.ndarray] = []
    for history, _ in run.chunks:
        ends.extend((history.z[history.accepted]).tolist())
        found.append(np.array(ends))
    return found


def _assert_inside_crossing_steps(run: Run) -> int:
    """Assert each requested plane is observed inside the last step delivered before it.

    Parameters
    ----------
    run : sessions.executors.Run
        A dense run.

    Returns
    -------
    int
        How many planes fell strictly inside a step: interpolated, not landed.
    """
    inside = 0
    for event, ends in zip(run.events, _delivered_ends(run), strict=True):
        if not (
            isinstance(event, ObservationReady) and event.kind is ObservationKind.PLANE
        ):
            continue
        # The step that crossed the plane is the last one delivered: it ends at or past
        # the plane, and every step before it ends short of it.
        assert ends.size and ends[-1] >= event.z, (event.z, ends[-1:])
        assert ends.size == 1 or ends[-2] < event.z, (event.z, ends[-2:])
        inside += bool(ends[-1] > event.z)
    return inside


@pytest.mark.parametrize("method", sorted(DENSE_METHODS))
def test_dense_planes_follow_the_committed_order(start, executor, method):
    """SEP 47 section 5.3: dense planes are served in requested-position order, each at
    its own position inside the step that crossed it, neither replacing the committed
    endpoint nor advancing it.

    The run of :func:`test_events_follow_the_committed_order` with its planes
    interpolated: the launch first, the planes in order at exactly their requested
    positions, a step plane at every third accepted step -- after the planes its step
    crossed --, the terminal event last; each attempt and row delivered once, and the
    monitor's record holding one row per observation at its position.
    """
    executor.require(DENSE)
    executor.require(MONITOR)
    log: list[tuple[str, float]] = []
    problem = soliton(
        step_planes=3, period=2, interpolate=True, integrator=DENSE_METHODS[method]
    )
    started = start(problem, problem.request.for_run(monitor=spy_monitor(log)))
    run = drive(started)
    _assert_committed_order(run)
    assert _assert_inside_crossing_steps(run) > 0, "no plane was interpolated"
    # The run's end is landed on; the monitor closed one row at each observation.
    assert run.accepted_ends()[-1] == LENGTH
    record = started.session.monitor_record()
    assert record is not None
    assert record["planes"]["z"] == [event.z for event in run.observations]


@pytest.mark.parametrize("method", sorted(DENSE_METHODS))
def test_a_snapshot_waits_for_the_planes_its_step_crossed(start, executor, method):
    """SEP 47 section 5.5: before a snapshot, every earlier owed dense plane crossed by
    the accepted step is observed, and the snapshot is taken at an accepted endpoint.

    A snapshot is requested at every interpolated plane, while the session holds the
    plane's field between two endpoints. Each is serviced at the next accepted endpoint,
    after every plane short of it, so that no persisted state would need the step's
    extension to reconstruct a plane; and the planes, attempts and fields are the run's
    without snapshots.
    """
    executor.require(DENSE)
    problem = soliton(
        step_planes=0, period=2, interpolate=True, integrator=DENSE_METHODS[method]
    )
    baseline = drive(start(problem), export=IDENTITY)
    held: list[float] = []

    def between(session: RunSession, batch: EventBatch) -> None:
        for event in batch.events:
            if (
                isinstance(event, ObservationReady)
                and event.kind is ObservationKind.PLANE
            ):
                held.append(event.z)
                session.request_snapshot()

    run = drive(start(problem), export=IDENTITY, between=between)
    snapshots = [event for event in run.events if isinstance(event, ServiceRequired)]
    assert snapshots and held
    ends = _delivered_ends(run)
    for snapshot in snapshots:
        assert snapshot.kind is ServiceKind.SNAPSHOT
        index = run.events.index(snapshot)
        assert snapshot.z in ends[index].tolist(), "a snapshot between endpoints"
        before = [
            event.z
            for event in run.events[:index]
            if isinstance(event, ObservationReady)
            and event.kind is ObservationKind.PLANE
        ]
        # A plane at the endpoint itself is observed first too (section 5.3).
        assert before == [plane for plane in PLANES if plane <= snapshot.z]
    assert _assert_inside_crossing_steps(run) > 0, "no plane was interpolated"
    if executor.deterministic:
        assert_same_run(run, baseline)


@pytest.mark.parametrize("step_planes", [1, 2])
def test_a_step_plane_ending_on_the_next_plane_is_observed_once(
    start, executor, step_planes
):
    """SEP 47 section 5.3: a step owed as a step plane that crossed an interpolated
    plane and ends on the next one is observed by that plane alone.

    On the Adams soliton the step from 0.007 m crosses the plane at 0.01 m and lands on
    the one at 0.015 m: the plane there observes the step's end, which is not observed
    again as a step plane at the same position.
    """
    executor.require(DENSE)
    problem = soliton(
        step_planes=step_planes, interpolate=True, integrator=DENSE_METHODS["adams"]
    )
    run = drive(start(problem))
    ends = run.accepted_ends().tolist()
    planes = [
        event.z for event in run.observations if event.kind is ObservationKind.PLANE
    ]
    # The case: a step crossing a plane and ending exactly on the next.
    assert any(
        later in ends and ends[ends.index(later) - 1] < earlier < later
        for earlier, later in itertools.pairwise(planes)
    ), "no step crossed a plane and landed on the next"
    positions = [event.z for event in run.observations]
    assert len(set(positions)) == len(positions), positions


@pytest.mark.parametrize(
    "bounds",
    [
        {"work_limit": 1},
        {"work_limit": 7},
        {"work_limit": 10**6},
        {"deadline": 0.0},
    ],
    ids=["work-limit-1", "work-limit-7", "work-limit-1e6", "deadline-0"],
)
@pytest.mark.parametrize("method", sorted(DENSE_METHODS))
def test_yields_change_nothing_beside_dense_planes(start, executor, method, bounds):
    """SEP 47 sections 5.1 and 7.1: bounded yields are exact transport no-ops beside
    interpolated planes, a held plane and its step's extension included.

    The run with its planes interpolated and a step plane every third step, yielding
    after every attempt, every seventh, hardly ever or at once whatever was made, is
    bit-identical to the run with the default bounds: every attempt, row, observation
    and exported field.
    """
    executor.require(DENSE)
    _deterministic(executor)
    problem = soliton(
        step_planes=3, period=2, interpolate=True, integrator=DENSE_METHODS[method]
    )
    baseline = drive(start(problem), export=IDENTITY)
    run = drive(start(problem), export=IDENTITY, **bounds)
    assert_same_run(run, baseline)
    if bounds.get("work_limit") == 1:
        assert len(run.batches) > len(baseline.batches)


MANY_PLANES = tuple(float(z) for z in np.linspace(LENGTH / 64, LENGTH, 64))
"""Planes closer than the soliton's steps: several fall inside each step."""


@pytest.mark.parametrize("method", sorted(DENSE_METHODS))
def test_a_snapshot_among_close_planes_is_taken_at_the_crossing_step_s_end(
    start, executor, method
):
    """SEP 47 section 5.5: with several planes inside each step, a snapshot requested
    while a plane is held is taken at the end of the step that crossed it, once every
    plane that step crossed is observed -- not put off by every later step, each of
    which crosses another plane -- and the planes are the run's without it.
    """
    executor.require(DENSE)
    problem = soliton(
        interpolate=True, integrator=DENSE_METHODS[method], planes=MANY_PLANES
    )
    baseline = drive(start(problem), export=IDENTITY)
    requested: list[int] = []

    def between(session: RunSession, batch: EventBatch) -> None:
        if requested or batch.terminal is not None:
            return
        for event in batch.events:
            if (
                isinstance(event, ObservationReady)
                and event.kind is ObservationKind.PLANE
            ):
                requested.append(event.plane_index)
                session.request_snapshot()

    run = drive(start(problem), export=IDENTITY, between=between)
    (snapshot,) = [event for event in run.events if isinstance(event, ServiceRequired)]
    index = run.events.index(snapshot)
    ends = _delivered_ends(run)
    (first,) = [
        position
        for position, event in enumerate(run.events)
        if isinstance(event, ObservationReady)
        and event.plane_index == requested[0]
        and event.kind is ObservationKind.PLANE
    ]
    # The end of the step that crossed the plane it was requested at, which no other
    # step was delivered before.
    assert snapshot.z == ends[first][-1] == ends[index][-1]
    assert snapshot.z < LENGTH
    planes = [
        event.z
        for event in run.events[:index]
        if isinstance(event, ObservationReady) and event.kind is ObservationKind.PLANE
    ]
    assert planes == [plane for plane in MANY_PLANES if plane <= snapshot.z]
    assert len(planes) > requested[0] + 1, "the step crossed one plane only"
    assert [event.z for event in run.observations] == [
        event.z for event in baseline.observations
    ]
    if executor.deterministic:
        assert_same_run(run, baseline)


# ------------------------------------------------------------------- invariance


INVARIANCE_SCRIPT = {"reject_at": (2, 7, 12)}
"""Rejections for the scripted executor's invariance runs, so their traces have some."""


def _invariance_run(start, executor: Executor, **controls) -> Run:
    """Run the step-plane soliton with periodic rows, exporting every observation.

    Parameters
    ----------
    start : callable
        Starts a session of `executor`.
    executor : sessions.executors.Executor
        The executor.
    **controls
        ``stop``, ``listener`` and ``between``, ``work_limit``, ``deadline``.

    Returns
    -------
    sessions.executors.Run
        The run.
    """
    bounds = {
        name: controls.pop(name)
        for name in ("between", "work_limit", "deadline")
        if name in controls
    }
    script = INVARIANCE_SCRIPT if executor.scripted else {}
    started = start(soliton(step_planes=3, period=2), **controls, **script)
    return drive(started, export=IDENTITY, **bounds)


def test_acknowledgment_delay_changes_nothing(start, executor):
    """SEP 47 sections 5.1 and 7.1: when a batch is acknowledged changes nothing.

    Before each acknowledgment the consumer waits, reads the status, exports every
    ticket with every recipe and tries a stale receipt; the fields and the control
    trace are bit-identical to a prompt run.
    """
    _deterministic(executor)
    baseline = _invariance_run(start, executor)
    previous: list[EventBatch] = []

    def slow_consumer(session: RunSession, batch: EventBatch) -> None:
        time.sleep(1e-3)
        session.status()
        for event in batch.events:
            ticket = getattr(event, "ticket", None)
            if ticket is not None:
                for recipe in RECIPES:
                    session.export(ticket, recipe)
        if previous:
            with pytest.raises(SessionError):
                session.acknowledge(previous[-1].receipt)
        previous.append(batch)

    assert_same_run(_invariance_run(start, executor, between=slow_consumer), baseline)


@pytest.mark.parametrize(
    "when",
    ["entrance", "every observation", "trials 3, 7 and 11", "every trial"],
)
def test_snapshot_timing_changes_nothing(start, executor, when):
    """SEP 47 sections 5.1, 5.5 and 7.1: a snapshot forces no landing and changes no
    accepted itinerary, wherever it is requested.
    """
    _deterministic(executor)
    baseline = _invariance_run(start, executor)
    holder: list[RunSession] = []
    count = [0]

    def listener(z: float, dz: float, accepted: bool) -> None:
        del z, dz, accepted
        count[0] += 1
        if when == "every trial" or (
            when == "trials 3, 7 and 11" and count[0] in {3, 7, 11}
        ):
            holder[0].request_snapshot()

    def between(session: RunSession, batch: EventBatch) -> None:
        if batch.terminal is not None:
            return
        observed = any(isinstance(event, ObservationReady) for event in batch.events)
        if (when == "every observation" and observed) or (
            when == "entrance" and len(holder) == 1
        ):
            session.request_snapshot()
            if when == "entrance":
                holder.append(session)

    script = INVARIANCE_SCRIPT if executor.scripted else {}
    started = start(soliton(step_planes=3, period=2), listener=listener, **script)
    holder.append(started.session)
    run = drive(started, export=IDENTITY, between=between)
    assert any(isinstance(event, ServiceRequired) for event in run.events)
    assert_same_run(run, baseline)


def test_an_observational_listener_changes_nothing(start, executor):
    """SEP 47 sections 5.1 and 7.1: observational reporters change nothing."""
    _deterministic(executor)
    baseline = _invariance_run(start, executor)
    heard: list[tuple[float, float, bool]] = []
    run = _invariance_run(
        start,
        executor,
        listener=lambda z, dz, accepted: heard.append((z, dz, accepted)),
        stop=StopToken(),
    )
    assert len(heard) == run.status.attempts
    assert_same_run(run, baseline)


@pytest.mark.parametrize("work_limit", [1, 7, 10**6])
def test_the_attempt_bound_changes_nothing(start, executor, work_limit):
    """SEP 47 sections 5.1 and 7.1: attempt limits of one, seven and a large bound."""
    _deterministic(executor)
    baseline = _invariance_run(start, executor)
    assert_same_run(_invariance_run(start, executor, work_limit=work_limit), baseline)


@pytest.mark.parametrize("deadline", [1e-9, 1e-3, 1e6])
def test_the_deadline_changes_nothing(start, executor, deadline):
    """SEP 47 sections 5.1 and 7.1: deadlines triggered at different attempts."""
    _deterministic(executor)
    baseline = _invariance_run(start, executor)
    assert_same_run(_invariance_run(start, executor, deadline=deadline), baseline)
