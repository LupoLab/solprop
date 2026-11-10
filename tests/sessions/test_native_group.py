# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native group session (SEP 47 section 6.7): members it admits, and how they end.

:class:`solprop_native.group.NativeGroupSession` marches each member through the
common fixed schedule in its own native session, in rounds of common steps
(:data:`solprop_native.group.ROUND_STEPS`). A member's trajectory is its lone native
session's, which is also the transitional lockstep group's: these tests hold the two
groups to the same fields, statistics, attempts and rows, bit for bit at one thread. A
failure is injected natively (``adams_march_fault``) to place the group's failures: a
member ahead of the failing one in its round has committed and recorded its steps, and
each member ends at its own committed position. The protocol's conformance tests,
which both groups pass, are in ``test_group.py``.
"""

import dataclasses
import os
import warnings
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from typing import Any, cast

import numpy as np
import pytest

from march_bounds import FIXED_ORDER
from sessions.test_group import (
    IDENTITY,
    PLANES,
    STEP,
    constructed,
    drive,
    group_of,
    member,
)
from solprop.backends.driver_session import DriverGroupSession
from solprop.backends.events import (
    Eligibility,
    EventBatch,
    GroupBatch,
    ObservationReady,
    Outcome,
    Terminal,
)
from solprop.backends.session import SessionError, Start
from solprop.backends.sessions import start_group
from solprop.exceptions import BackendError, BoundaryWarning, StopRequested
from solprop.monitor import BoundaryCheck, BoundaryMonitor
from solprop.stepper import Multistep
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.run_request import describe_run

group_module = pytest.importorskip("solprop_native.group")

INTEGRATORS = {
    "runge_kutta": None,
    "adams_fixed_order": FIXED_ORDER,
    "adams": Multistep(),
    "low_storage": LowStorageRungeKutta(),
}
"""The marches a native group runs: each member's own, natively."""


def _one_thread() -> bool:
    """Return whether the native backend runs one thread, as the default lane sets it.

    Returns
    -------
    bool
        True when ``SOLPROP_NATIVE_THREADS`` or ``SOLPROP_THREADS`` is ``"1"``.
    """
    return "1" in {
        os.environ.get("SOLPROP_NATIVE_THREADS"),
        os.environ.get("SOLPROP_THREADS"),
    }


def _started(
    simulations: Sequence[Any],
    cleanup: ExitStack,
    *,
    faults: Mapping[int, tuple[int, str]] | None = None,
    transitional: bool = False,
) -> Any:
    """Start a group of members, arming native faults first; return the group.

    Parameters
    ----------
    simulations : sequence of solprop.driver.Simulation
        The members.
    cleanup : contextlib.ExitStack
        Owner of the backends and the group.
    faults : collections.abc.Mapping or None, optional
        Per member, how many runs of a march stage succeed and the stage that then
        fails (``adams_march_fault``, a backend failure).
    transitional : bool, optional
        Start the transitional group over the lockstep runner in place of the door's.

    Returns
    -------
    solprop.backends.session.GroupSession
        The started group.
    """
    problems = [one.translate() for one in simulations]
    owners = [constructed(one, cleanup) for one in simulations]
    for index, (after, stage) in (faults or {}).items():
        owners[index]._native_state.adams_march_fault(stage, after, "backend")
    requests = [describe_run(problem) for problem in problems]
    starts = [Start.entrance(problem) for problem in problems]
    kind = DriverGroupSession if transitional else start_group
    group = kind(owners, problems, requests, starts)
    cleanup.callback(group.close)
    return group


def _record(group: Any) -> list[list[dict[str, Any]]]:
    """Drive a group to its end; return each member's planes as plain arrays.

    Parameters
    ----------
    group : solprop.backends.session.GroupSession
        A started group.

    Returns
    -------
    list
        Per member, per plane: the field, the plane statistics, the attempts and the
        periodic rows.
    """
    planes: list[list[dict[str, Any]]] = [[] for _ in range(group.n_members)]
    while not group.finished:
        batch = group.advance()
        for index, events in enumerate(batch.members):
            for event in events:
                if isinstance(event, ObservationReady):
                    history, rows = event.history, event.step_stats
                    planes[index].append(
                        {
                            "field": np.array(
                                group.export(index, event.ticket, IDENTITY)
                            ),
                            "stats": {k: np.asarray(v) for k, v in event.stats.items()},
                            "attempts": np.stack(
                                [history.z, history.dz, history.error_norm]
                            ),
                            "accepted": history.accepted.copy(),
                            "row_z": rows.z.copy(),
                            "row_step": rows.step.copy(),
                            "rows": {k: v.copy() for k, v in rows.values.items()},
                        }
                    )
        group.acknowledge(batch.receipt)
    return planes


def _same(left: Any, right: Any) -> bool:
    """Return whether two recorded values agree exactly, nested containers included.

    Parameters
    ----------
    left, right : object
        Arrays, or dicts of them.

    Returns
    -------
    bool
        True when every array is bitwise equal.
    """
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same(left[key], right[key]) for key in left
        )
    return bool(np.array_equal(left, right))


@pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning")
@pytest.mark.parametrize("name", sorted(INTEGRATORS))
def test_a_native_member_marches_as_the_transitional_group_bit_for_bit(name):
    """Each member's fields, statistics, attempts and rows are the transitional
    group's exactly: both march the member alone through its own native kernels in
    one operation order, the lockstep's fused accepted update reducing each member's
    norm as its lone run does (structural identity, one thread).
    """
    if not _one_thread():
        pytest.skip(
            "a native run on several threads may vary its reduction order; the "
            "default lane runs one (SOLPROP_NATIVE_THREADS=1)"
        )
    integrator = INTEGRATORS[name]
    recorded = []
    for transitional in (False, True):
        with ExitStack() as cleanup:
            simulations = [member(index, "native", integrator) for index in range(3)]
            group = _started(simulations, cleanup, transitional=transitional)
            expected = (
                DriverGroupSession if transitional else group_module.NativeGroupSession
            )
            assert isinstance(group, expected)
            recorded.append(_record(group))
    native, lockstep = recorded
    for index, (ours, theirs) in enumerate(zip(native, lockstep, strict=True)):
        assert len(ours) == len(theirs) == len(PLANES) + 1
        for plane, (one, other) in enumerate(zip(ours, theirs, strict=True)):
            for key in one:
                assert _same(one[key], other[key]), (index, plane, key)


def test_the_execution_record_names_the_group_and_keeps_the_session_s_counters():
    with ExitStack() as cleanup:
        group = _started([member(index, "native") for index in range(2)], cleanup)
        drive(group)
        record = group.execution_record(1)
        assert record["backend"] == "native"
        assert record["batch_size"] == 2
        assert record["batch_strategy"] == "native_group_session"
        assert record["fused_batch_kernels"] is False
        counters = record["native_session"]
        assert isinstance(counters, dict)
        # The launch, two planes and the completion: one native batch each.
        assert counters["batches"] == len(PLANES) + 2


@pytest.mark.parametrize(("rounds", "ahead"), [(256, PLANES[1]), (1, PLANES[0] + STEP)])
def test_a_trial_failure_ends_the_group_at_each_member_s_own_position(
    monkeypatch, rounds, ahead
):
    """The second member's third trial fails, opening the second window. The first
    member took its turn first: with the default rounds it ran the whole window, with
    rounds of one step only the failed step, and in both it committed and recorded what
    it ran -- eligible where it stands, unlike the transitional group's stepped but
    unrecorded member. The failing member keeps its committed state; the third never
    took its turn (SEP 47 section 6.7).
    """
    monkeypatch.setattr(group_module, "ROUND_STEPS", rounds)
    simulations = [member(index, "native", FIXED_ORDER) for index in range(3)]
    with ExitStack() as cleanup:
        group = _started(simulations, cleanup, faults={1: (2, "trial")})
        for _ in range(2):  # the launch and the first plane
            window = group.advance()
            assert window.outcome is None
            group.acknowledge(window.receipt)
        ended = group.advance()
        assert ended.outcome is Outcome.FAILED
        assert ended.error is not None
        terminals = [events[0] for events in ended.members]
        assert all(isinstance(terminal, Terminal) for terminal in terminals)
        assert all(terminal.error is ended.error for terminal in terminals)
        assert [terminal.z for terminal in terminals] == pytest.approx(
            [ahead, PLANES[0], PLANES[0]]
        )
        assert all(
            terminal.eligibility is Eligibility.ELIGIBLE for terminal in terminals
        )
        assert [terminal.history.n_steps for terminal in terminals] == [
            round((ahead - PLANES[0]) / STEP),
            0,
            0,
        ]
        # Each member can be checkpointed where it stands, the first past the others.
        for index, terminal in enumerate(terminals):
            assert terminal.ticket is not None
            field = group.export(index, terminal.ticket, IDENTITY)
            assert np.all(np.isfinite(field))
            status = group.member_status(index)
            assert status.outcome is Outcome.FAILED
            assert status.z == pytest.approx(terminal.z)


def test_an_in_flight_failure_poisons_its_member_alone():
    """A low-storage step overwrites the state in place: abandoned partway, its member
    has no state to checkpoint; the member after it, not yet stepped, does.
    """
    simulations = [
        member(index, "native", LowStorageRungeKutta()) for index in range(2)
    ]
    with ExitStack() as cleanup:
        group = _started(simulations, cleanup, faults={0: (0, "trial")})
        launch = group.advance()
        group.acknowledge(launch.receipt)
        ended = group.advance()
        poisoned, untouched = (events[0] for events in ended.members)
        assert isinstance(poisoned, Terminal)
        assert isinstance(untouched, Terminal)
        assert poisoned.eligibility is Eligibility.POISONED
        assert poisoned.ticket is None
        assert poisoned.z == 0.0
        assert untouched.eligibility is Eligibility.ELIGIBLE
        assert untouched.ticket is not None
        assert untouched.error is poisoned.error is ended.error
        field = group.export(1, untouched.ticket, IDENTITY)
        assert np.all(np.isfinite(field))
        group.acknowledge(ended.receipt)
        assert group.finished


class _StopAfter:
    """A stop flag that is set from its `polls`-th poll on.

    Parameters
    ----------
    polls : int
        The poll at which it is first set.
    """

    def __init__(self, polls: int) -> None:
        self.polls = polls
        self.count = 0

    def is_set(self) -> bool:
        """Return whether the flag is set, counting the poll."""
        self.count += 1
        return self.count >= self.polls


def test_a_group_wide_stop_ends_every_member_at_the_common_step_between_rounds(
    monkeypatch,
):
    """With rounds of one step, the stop polled before the window's second round ends
    every member one step into the window: a common accepted endpoint, each eligible.
    """
    monkeypatch.setattr(group_module, "ROUND_STEPS", 1)
    stop = _StopAfter(2)
    with ExitStack() as cleanup:
        group, _ = group_of(
            [member(index, "native") for index in range(3)], cleanup, stop=stop
        )
        launch = group.advance()
        group.acknowledge(launch.receipt)
        assert stop.count == 0  # the launch takes no step and polls no stop
        stopped = group.advance()
        assert stopped.outcome is Outcome.STOPPED
        assert isinstance(stopped.error, StopRequested)
        for index, events in enumerate(stopped.members):
            (terminal,) = events
            assert isinstance(terminal, Terminal)
            assert terminal.outcome is Outcome.STOPPED
            assert terminal.z == pytest.approx(STEP)
            assert terminal.history is not None
            assert terminal.history.n_steps == 1
            assert terminal.ticket is not None
            assert np.all(np.isfinite(group.export(index, terminal.ticket, IDENTITY)))
        group.acknowledge(stopped.receipt)
        assert group.finished


def _warned(batches: Sequence[GroupBatch]) -> Terminal | None:
    """Return the first member's terminal event, if it ended before the group did.

    Parameters
    ----------
    batches : sequence of solprop.backends.events.GroupBatch
        Every window.

    Returns
    -------
    solprop.backends.events.Terminal or None
        Its terminal event.
    """
    for batch in batches:
        for event in batch.members[0]:
            if isinstance(event, Terminal):
                return event
    return None


def test_a_boundary_warning_raised_as_an_error_ends_its_member_alone():
    """A warning a member's monitor issues is served by the group at its barrier: under
    an ``"error"`` filter it ends that member alone, at the warning's committed
    position, eligible; the other member, unwatched, runs the whole schedule.
    """
    simulations = [member(index, "native") for index in range(2)]
    # A budget every launch exceeds, and a stop no guard reaches: a warning, no stop.
    noisy = BoundaryCheck(budget=1e-300, stop_factor=1e300)
    monitors = [BoundaryMonitor.build(simulations[0].grid, None, noisy), None]
    with ExitStack() as cleanup, warnings.catch_warnings():
        warnings.simplefilter("error", BoundaryWarning)
        group, _ = group_of(simulations, cleanup, monitors=monitors)
        batches = drive(group)
        ended = _warned(batches)
        assert ended is not None
        assert ended.outcome is Outcome.FAILED
        assert isinstance(ended.error, BoundaryWarning)
        assert ended.eligibility is Eligibility.ELIGIBLE
        assert ended.z == 0.0
        assert batches[-1].outcome is Outcome.COMPLETED
        assert batches[-1].error is None
        assert group.member_status(1).outcome is Outcome.COMPLETED
        assert group.monitor_record(0) is not None
    # Allowed, the warning leaves its member running to the end.
    with ExitStack() as cleanup, pytest.warns(BoundaryWarning):
        group, _ = group_of(
            [member(index, "native") for index in range(2)],
            cleanup,
            monitors=[BoundaryMonitor.build(simulations[0].grid, None, noisy), None],
        )
        batches = drive(group)
        assert _warned(batches[:-1]) is None
        assert group.member_status(0).outcome is Outcome.COMPLETED


def _requests(simulations: Sequence[Any]) -> tuple[list[Any], list[Any], list[Any]]:
    """Return the members' problems, run requests and starts.

    Parameters
    ----------
    simulations : sequence of solprop.driver.Simulation
        The members.

    Returns
    -------
    tuple
        Their problems, requests and starts.
    """
    problems = [one.translate() for one in simulations]
    return (
        problems,
        [describe_run(problem) for problem in problems],
        [Start.entrance(problem) for problem in problems],
    )


def test_the_native_group_refuses_by_name_what_it_does_not_run():
    simulations = [member(index, "native") for index in range(2)]
    problems, requests, starts = _requests(simulations)
    assert group_module.refusal(problems, requests, starts) is None
    adaptive = dataclasses.replace(requests[1], replay=None)
    assert (
        "an adaptive march: a group replays one common fixed schedule (member 1)"
        in group_module.refusal(problems, [requests[0], adaptive], starts)
    )
    stepped = dataclasses.replace(
        requests[0],
        observation=dataclasses.replace(requests[0].observation, step_planes=1),
    )
    assert (
        "step planes, which a group's save-plane windows do not observe (member 0)"
        in group_module.refusal(problems, [stepped, requests[1]], starts)
    )
    finer = member(
        1,
        "native",
        schedule=FixedSchedule.uniform(0.0, PLANES[-1], STEP / 2, save_planes=PLANES),
    ).translate()
    assert "member 1, which does not share member 0's schedule" in (
        group_module.refusal(
            [problems[0], finer], [requests[0], describe_run(finer)], starts
        )
    )
    with ExitStack() as cleanup:
        owners = [constructed(one, cleanup) for one in simulations]
        with pytest.raises(
            BackendError, match=r"does not run an adaptive march.*\(member 1\)"
        ):
            group_module.NativeGroupSession(
                owners, problems, [requests[0], adaptive], starts
            )
        reference = [
            constructed(member(index, "numpy_ref"), cleanup) for index in range(2)
        ]
        with pytest.raises(BackendError, match="takes native backends"):
            group_module.NativeGroupSession(reference, problems, requests, starts)


def test_a_group_the_native_group_refuses_runs_in_the_transitional_group():
    """A noise model is outside the native session: the door starts the transitional
    group, which runs it; a group neither runs is refused by the lockstep's own check.
    """
    simulations = [
        member(index, "native", noise="default", seed=index) for index in range(2)
    ]
    problems, requests, starts = _requests(simulations)
    refused = group_module.refusal(problems, requests, starts)
    assert refused is not None
    assert "noise model (member 0)" in refused
    with ExitStack() as cleanup:
        owners = [constructed(one, cleanup) for one in simulations]
        group = start_group(owners, problems, requests, starts)
        cleanup.callback(group.close)
        assert isinstance(group, DriverGroupSession)
        assert drive(group)[-1].outcome is Outcome.COMPLETED
    simulations = [member(index, "native", schedule=None) for index in range(2)]
    problems, requests, starts = _requests(simulations)
    assert "an adaptive march" in (
        group_module.refusal(problems, requests, starts) or ""
    )
    with ExitStack() as cleanup:
        owners = [constructed(one, cleanup) for one in simulations]
        with pytest.raises(BackendError, match="fixed schedule"):
            start_group(owners, problems, requests, starts)


def _terminals(batch: GroupBatch) -> list[Terminal]:
    """Return each member's one event of a batch that ended the group, checked typed.

    Parameters
    ----------
    batch : solprop.backends.events.GroupBatch
        The batch.

    Returns
    -------
    list of solprop.backends.events.Terminal
        Each member's terminal event.
    """
    terminals = []
    for events in batch.members:
        (terminal,) = events
        assert isinstance(terminal, Terminal)
        terminals.append(terminal)
    return terminals


def _steps(terminal: Terminal) -> int:
    """Return the attempts a terminal event delivers."""
    assert terminal.history is not None
    return terminal.history.n_steps


class _ManufacturedError(RuntimeError):
    """What a patched member session raises or reports."""


def _failing_at_launch(session: Any, error: BaseException) -> Any:
    """Return a member session's ``advance`` whose launch reports a failure.

    Parameters
    ----------
    session : solprop_native.session.NativeSession
        The member's session.
    error : BaseException
        The failure its launch reports.

    Returns
    -------
    callable
        The patched ``advance``: the launch batch, its observation replaced by a
        failed terminal event at the entrance, eligible, with the launch's ticket.
    """
    advance = session.advance

    def failed(**limits: Any) -> EventBatch:
        batch = advance(**limits)
        (launch,) = batch.events
        assert isinstance(launch, ObservationReady)
        return EventBatch(
            batch.receipt,
            (
                Terminal(
                    Outcome.FAILED,
                    launch.z,
                    Eligibility.ELIGIBLE,
                    error,
                    launch.ticket,
                    launch.history,
                    launch.step_stats,
                ),
            ),
        )

    return failed


def test_a_failure_at_launch_poisons_the_members_that_never_launched(monkeypatch):
    """The second of three members fails at its launch: the first launched and can be
    checkpointed at its entrance; the third never launched, holds no batch of its own
    to export from, and so claims no checkpoint (a terminal event without a ticket is
    poisoned).
    """
    error = _ManufacturedError("a manufactured launch failure")
    with ExitStack() as cleanup:
        group, _ = group_of([member(index, "native") for index in range(3)], cleanup)
        session = cast("Any", group)._sessions[1]
        monkeypatch.setattr(session, "advance", _failing_at_launch(session, error))
        ended = group.advance()
        assert ended.outcome is Outcome.FAILED
        assert ended.error is error
        terminals = _terminals(ended)
        assert all(terminal.error is error for terminal in terminals)
        assert all(terminal.z == 0.0 for terminal in terminals)
        assert [terminal.eligibility for terminal in terminals] == [
            Eligibility.ELIGIBLE,
            Eligibility.ELIGIBLE,
            Eligibility.POISONED,
        ]
        for index, terminal in enumerate(terminals):
            assert (terminal.ticket is None) is (index == 2)
            if terminal.ticket is not None:
                assert terminal.ticket.member == index
                field = group.export(index, terminal.ticket, IDENTITY)
                assert np.all(np.isfinite(field))
        group.acknowledge(ended.receipt)
        assert group.finished


class _Raising:
    """A member session's ``advance`` that raises on its `call`-th call.

    Parameters
    ----------
    session : solprop_native.session.NativeSession
        The member's session.
    call : int
        The call that raises.
    error : BaseException
        What it raises.
    """

    def __init__(self, session: Any, call: int, error: BaseException) -> None:
        self.advance = session.advance
        self.call = call
        self.error = error
        self.calls = 0

    def __call__(self, **limits: Any) -> EventBatch:
        """Advance the session, or raise on the chosen call."""
        self.calls += 1
        if self.calls == self.call:
            raise self.error
        return self.advance(**limits)


@pytest.mark.parametrize("kind", ["exception", "interrupt"])
def test_an_exception_escaping_a_member_ends_the_group_with_its_chunks(
    monkeypatch, kind
):
    """With rounds of one step, the second member's call raises in the window's second
    round, after it acknowledged its batch: the group ends failed, delivering each
    member's attempts so far. The first member reached the plane, the third stands one
    step in; the second has no batch of its own pending, so claims no checkpoint. An
    interrupt propagates, the group's failed batch left pending: it advances no more.
    """
    # Made here, not in the parameters: a raised exception's traceback holds the
    # group's frames, which must not outlive the test.
    error: BaseException = (
        _ManufacturedError("escaped") if kind == "exception" else KeyboardInterrupt()
    )
    monkeypatch.setattr(group_module, "ROUND_STEPS", 1)
    with ExitStack() as cleanup:
        group, _ = group_of([member(index, "native") for index in range(3)], cleanup)
        session = cast("Any", group)._sessions[1]
        # The launch, the window's first step, then the failing call.
        monkeypatch.setattr(session, "advance", _Raising(session, 3, error))
        launch = group.advance()
        group.acknowledge(launch.receipt)
        if isinstance(error, KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                group.advance()
            with pytest.raises(SessionError, match="acknowledge the pending batch"):
                group.advance()
            return
        ended = group.advance()
        assert ended.outcome is Outcome.FAILED
        assert ended.error is error
        terminals = _terminals(ended)
        assert all(terminal.error is error for terminal in terminals)
        assert [terminal.z for terminal in terminals] == pytest.approx(
            [PLANES[0], STEP, STEP]
        )
        assert [_steps(terminal) for terminal in terminals] == [2, 1, 1]
        assert [terminal.eligibility for terminal in terminals] == [
            Eligibility.ELIGIBLE,
            Eligibility.POISONED,
            Eligibility.ELIGIBLE,
        ]
        for index in (0, 2):
            ticket = terminals[index].ticket
            assert ticket is not None
            assert np.all(np.isfinite(group.export(index, ticket, IDENTITY)))
        group.acknowledge(ended.receipt)
        assert group.finished
        with pytest.raises(SessionError, match="has ended"):
            group.advance()
