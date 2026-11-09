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

import dataclasses
import threading
import time
import warnings

import numpy as np
import pytest

from sessions.executors import (
    CONSUMER,
    FACTOR,
    IDENTITY,
    LENGTH,
    MONITOR,
    PLANES,
    RECIPES,
    STATISTICS,
    STORED,
    Executor,
    Run,
    ScriptedError,
    ScriptedWarning,
    assert_same_run,
    drive,
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
    ServiceKind,
    ServiceRequired,
    Terminal,
)
from solprop.backends.session import (
    RunSession,
    SessionError,
    SessionState,
    SnapshotReceipt,
    StopToken,
)
from solprop.exceptions import BackendError, StopRequested
from solprop.stats import StatsSpec
from solprop.translate.run_request import describe_run

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
    overlap each other are refused; so is writing into an array the session lent
    rather than handed over. A refusal changes nothing the session exports next.
    """
    session = start(soliton()).session
    _, plane = _past_launch(session)
    expected = np.array(session.export(plane.ticket, IDENTITY))
    shape = expected.shape
    readonly = np.empty(shape, np.complex128)
    readonly.setflags(write=False)
    unfit = {
        "shape": np.empty((*shape[:-1], shape[-1] + 1), np.complex128),
        "dtype": np.empty(shape, np.complex64),
        "strided": np.empty((*shape[:-1], 2 * shape[-1]), np.complex128)[..., ::2],
        "readonly": readonly,
        "overlapping": np.lib.stride_tricks.as_strided(
            np.zeros(1, np.complex128), shape=shape, strides=(0,) * len(shape)
        ),
    }
    for recipe in (IDENTITY, STORED):
        for name, out in unfit.items():
            with pytest.raises(BackendError):
                session.export(plane.ticket, recipe, out=out)
                pytest.fail(f"a {name} destination was accepted")
    held = session.export(plane.ticket, IDENTITY)
    if not held.flags.writeable:
        # The session lent its own array: it is not a destination.
        with pytest.raises(BackendError):
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
    started = start(problem, describe_run(problem, monitor=spy_monitor(log)))
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
    problem = dataclasses.replace(soliton(step_planes=1), stats_spec=spec)
    run = drive(start(problem, describe_run(problem)))
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
    accepted = 0
    for event in observations[1:]:
        history = event.history
        before = accepted
        accepted += int(history.accepted.sum())
        assert event.stats.keys() == {"energy", "peak_power"}
        if event.kind is ObservationKind.STEP:
            assert accepted % 3 == 0
            low = PLANES[event.plane_index - 1] if event.plane_index else 0.0
            assert low < event.z < PLANES[event.plane_index]
        rows = event.step_stats
        assert rows.values.keys() == {"energy"}
        assert rows.values["energy"].shape == (
            rows.n_rows,
            *event.stats["energy"].shape,
        )
        if rows.n_rows:
            np.testing.assert_array_equal(history.z[rows.step], rows.z)
            assert history.accepted[rows.step].all()
            counted = before + np.cumsum(history.accepted)[rows.step]
            assert (counted % 2 == 0).all()
    assert run.attempts == run.status.attempts > 0
    assert sum(int((~history.accepted).sum()) for history in run.histories) == (
        run.status.rejected
    )


def test_attempt_records_add_up_to_the_status(start):
    """SEP 47 section 7.1: accurate actual-work totals, each attempt reported once."""
    started = start(soliton(step_planes=4))
    run = drive(started)
    assert run.attempts == run.status.attempts == started.session.status().attempts
    ends = run.accepted_ends()
    assert (np.diff(ends) > 0).all() and ends[-1] == LENGTH


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
    accepted = 0
    for event, field in zip(run.observations, run.fields, strict=True):
        accepted += int(event.history.accepted.sum())
        np.testing.assert_array_equal(field, run.fields[0] * FACTOR**accepted)


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
    token.clear()
    assert not token.is_set()
    token.set()
    assert calls == ["early", "late", "late"]


# -------------------------------------------------------------------- snapshots


def test_a_snapshot_requested_at_the_entrance_waits_for_an_accepted_endpoint(start):
    """SEP 47 section 5.5: a snapshot latch is serviced at the next accepted endpoint.

    The entrance is not an accepted endpoint, so a request made there is serviced at
    the end of the first accepted step, with the attempts that reached it; the next
    plane still carries them.
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
    for column in ("z", "dz", "error_norm", "accepted"):
        np.testing.assert_array_equal(
            getattr(plane.history, column)[: snapshot.history.n_steps],
            getattr(snapshot.history, column),
        )


def test_a_snapshot_between_planes_carries_the_attempts_since_the_last_plane(start):
    """SEP 47 section 5.5: a running checkpoint includes, without consuming, the
    attempts and periodic rows since the last saved plane.

    The request is latched at the first trial after the first plane (from the attempt
    listener, standing in for a control token set at that attempt). It is serviced at
    the next accepted endpoint; its history and rows are those since the first plane,
    and the second plane's are the same, followed by the rest.
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
    launch, first, second = run.observations[:3]
    assert (first.plane_index, second.plane_index) == (0, 1)
    assert first.z < snapshot.z < second.z
    assert snapshot.history is not None and snapshot.step_stats is not None
    # The attempts from the first plane to the endpoint, that one included.
    reached = [z for z, _, _ in heard].index(end) + 1
    count = reached - launch.history.n_steps - first.history.n_steps
    assert snapshot.history.n_steps == count
    for column in ("z", "dz", "error_norm", "accepted"):
        np.testing.assert_array_equal(
            getattr(second.history, column)[:count], getattr(snapshot.history, column)
        )
    rows = snapshot.step_stats
    assert rows.n_rows == int(snapshot.history.accepted.sum())
    np.testing.assert_array_equal(second.step_stats.z[: rows.n_rows], rows.z)
    np.testing.assert_array_equal(second.step_stats.step[: rows.n_rows], rows.step)
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
