# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The group protocol over the lockstep runners (SEP 47 section 6.7).

A group advances compatible members through one fixed schedule, one common window per
save plane, and reports each member's own events. A boundary stop ends its member
alone; a non-finite norm, a trial exception or an in-flight failure ends the group,
and each member then reports its own committed position and checkpoint eligibility:
members are stepped in turn, so one ahead of the failing member has already committed
the failed step.
"""

from collections.abc import Sequence
from contextlib import ExitStack
from typing import Any

import numpy as np
import pytest

from bounds import assert_within
from march_bounds import FIXED_ORDER
from solprop.backends import select
from solprop.backends.events import (
    Eligibility,
    GroupBatch,
    ObservationKind,
    ObservationReady,
    Outcome,
    Terminal,
)
from solprop.backends.export import ExportRecipe
from solprop.backends.session import (
    GroupSession,
    SessionError,
    SessionState,
    Start,
    StopToken,
)
from solprop.backends.sessions import start_group, start_session
from solprop.driver import Simulation
from solprop.exceptions import BackendError, BoundaryError
from solprop.monitor import BoundaryCheck, BoundaryMonitor
from solprop.stats import StatsSpec
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.run_request import describe_run

PLANES = (0.0005, 0.001)
"""The save planes, m: four fixed steps of 0.25 mm, two per window."""

STEP = 0.00025
"""The fixed step, m."""

FIELD_BOUND = 4 * 16 * np.finfo(float).eps
"""The singleton ladder's 16 eps a step over four steps: a member and its lone run take
one method on one schedule and differ by the order of their reductions alone
(``tests/backends/test_batch_lockstep.py``)."""

IDENTITY = ExportRecipe(power=False, optics=False)

BACKENDS = ["numpy_ref", "native"]


def member(index: int, backend: str, integrator: Any = None) -> Simulation:
    """Return one member: a soliton-order pulse on its own fibre, on the schedule."""
    return Simulation.from_gnlse(
        0.11 * (index + 1),
        PLANES[-1],
        [0.0, 0.0, -1.183e-26 * (index + 1)],
        wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=float(index + 1),
        shape="sech",
        save_planes=list(PLANES),
        schedule=FixedSchedule.uniform(0.0, PLANES[-1], STEP, save_planes=PLANES),
        stats=StatsSpec(names=("energy", "peak_power"), periodic=("energy",), period=2),
        noise=None,
        backend=backend,
        integrator=RungeKutta() if integrator is None else integrator,
        absorber=False,
    )


def constructed(simulation: Simulation, cleanup: ExitStack) -> Any:
    """Return a member's backend, constructed and holding no state."""
    owner = select(
        simulation.settings.backend,
        simulation.translate(),
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    cleanup.callback(owner.close)
    return owner


def group_of(
    simulations: Sequence[Simulation],
    cleanup: ExitStack,
    *,
    monitors: Sequence[BoundaryMonitor | None] | None = None,
    stop: StopToken | None = None,
) -> tuple[GroupSession, list[Any]]:
    """Start a group of members at their entrances; return it and its members."""
    problems = [one.translate() for one in simulations]
    owners = [constructed(one, cleanup) for one in simulations]
    watched = monitors if monitors is not None else [None] * len(problems)
    group = start_group(
        owners,
        problems,
        [
            describe_run(problem, monitor=monitor)
            for problem, monitor in zip(problems, watched, strict=True)
        ],
        [Start.entrance(problem) for problem in problems],
        stop=stop,
    )
    cleanup.callback(group.close)
    return group, owners


def drive(group: GroupSession) -> list[GroupBatch]:
    """Advance a group to its end, acknowledging each window; return every batch."""
    batches = []
    while not group.finished:
        batch = group.advance()
        batches.append(batch)
        group.acknowledge(batch.receipt)
    return batches


def available(backend: str) -> None:
    """Skip the native executor where its package is not installed."""
    if backend == "native":
        pytest.importorskip("solprop_native")


@pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning")
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_group_reports_each_member_s_planes_as_its_lone_run_does(backend):
    available(backend)
    simulations = [member(index, backend) for index in range(2)]
    with ExitStack() as cleanup:
        group, _ = group_of(simulations, cleanup)
        assert group.n_members == 2
        fields: list[list[np.ndarray]] = [[], []]
        windows: list[list[ObservationKind]] = []
        batches = []
        while not group.finished:
            batch = group.advance()
            batches.append(batch)
            kinds = []
            for index, events in enumerate(batch.members):
                for event in events:
                    if isinstance(event, ObservationReady):
                        kinds.append(event.kind)
                        fields[index].append(
                            group.export(index, event.ticket, IDENTITY).copy()
                        )
                        # One member's own attempts: each window's chunk, once.
                        assert event.history.n_steps == (
                            0 if event.plane_index < 0 else 2
                        )
            windows.append(kinds)
            group.acknowledge(batch.receipt)
        last = batches[-1]
        assert last.outcome is Outcome.COMPLETED
        assert last.error is None
        assert all(
            isinstance(events[0], Terminal) and events[0].outcome is Outcome.COMPLETED
            for events in last.members
        )
        for index in range(2):
            status = group.member_status(index)
            assert status.outcome is Outcome.COMPLETED
            assert status.z == pytest.approx(PLANES[-1])
            assert status.attempts == 4
            assert group.execution_record(index)["batch_size"] == 2
    assert windows[:3] == [
        [ObservationKind.LAUNCH] * 2,
        [ObservationKind.PLANE] * 2,
        [ObservationKind.PLANE] * 2,
    ]
    for index, simulation in enumerate(simulations):
        problem = simulation.translate()
        with ExitStack() as cleanup:
            owner = constructed(simulation, cleanup)
            lone = start_session(
                owner, problem, describe_run(problem), Start.entrance(problem)
            )
            cleanup.callback(lone.close)
            alone = []
            while not lone.finished:
                batch = lone.advance()
                for event in batch.events:
                    if isinstance(event, ObservationReady):
                        alone.append(lone.export(event.ticket, IDENTITY).copy())
                lone.acknowledge(batch.receipt)
        assert len(alone) == len(fields[index]) == 3
        for grouped, solo in zip(fields[index], alone, strict=True):
            assert_within(grouped, solo, FIELD_BOUND, what=f"member {index}")


def test_a_group_s_receipts_and_tickets_follow_the_session_lifecycle():
    with ExitStack() as cleanup:
        group, _ = group_of([member(index, "numpy_ref") for index in range(2)], cleanup)
        launch = group.advance()
        with pytest.raises(SessionError, match="acknowledge the pending batch"):
            group.advance()
        (event,) = launch.members[0]
        assert isinstance(event, ObservationReady)
        group.acknowledge(launch.receipt)
        with pytest.raises(SessionError, match="not the pending batch"):
            group.acknowledge(launch.receipt)
        with pytest.raises(SessionError, match="has expired"):
            group.export(0, event.ticket, IDENTITY)
        batch = group.advance()
        (plane,) = batch.members[0]
        assert isinstance(plane, ObservationReady)
        with pytest.raises(SessionError, match="not a member 2"):
            group.export(2, plane.ticket, IDENTITY)
        group.acknowledge(batch.receipt)
        group.close()
        assert group.closed
        assert group.member_status(0).state is SessionState.CLOSED
        group.close()


def test_a_group_starts_its_members_fresh_and_one_start_each():
    simulations = [member(index, "numpy_ref") for index in range(2)]
    problems = [one.translate() for one in simulations]
    with ExitStack() as cleanup:
        owners = [constructed(one, cleanup) for one in simulations]
        requests = [describe_run(problem) for problem in problems]
        with pytest.raises(SessionError, match="one problem, request and start"):
            start_group(owners, problems, requests, [Start.entrance(problems[0])])
        resumed = Start.entrance(problems[1])
        resumed.plane_index = 1
        with pytest.raises(SessionError, match="at their entrance"):
            start_group(
                owners, problems, requests, [Start.entrance(problems[0]), resumed]
            )


def test_a_boundary_stop_ends_its_member_alone_and_the_others_go_on():
    # A check strict enough to refuse any launch stops the first member at once, as
    # the batched scan always attached its monitors before the launch plane; the
    # second, unwatched, runs the whole schedule (SEP 21).
    simulations = [member(index, "numpy_ref") for index in range(2)]
    strict = BoundaryCheck(budget=1e-300, stop_factor=1.0)
    with ExitStack() as cleanup:
        group, _ = group_of(
            simulations,
            cleanup,
            monitors=[BoundaryMonitor.build(simulations[0].grid, None, strict), None],
        )
        batches = drive(group)
        ended = next(
            window
            for window, batch in enumerate(batches)
            if any(isinstance(event, Terminal) for event in batch.members[0])
        )
        (stopped,) = batches[ended].members[0]
        assert isinstance(stopped, Terminal)
        assert stopped.outcome is Outcome.FAILED
        assert isinstance(stopped.error, BoundaryError)
        assert stopped.eligibility is Eligibility.ELIGIBLE
        assert (ended, stopped.z) == (0, 0.0)
        # The stopped member reports nothing more; the other saves every plane.
        assert all(batch.members[0] == () for batch in batches[ended + 1 :])
        planes = [
            event
            for batch in batches
            for event in batch.members[1]
            if isinstance(event, ObservationReady)
        ]
        assert [event.plane_index for event in planes] == [-1, 0, 1]
        assert batches[-1].outcome is Outcome.COMPLETED
        assert batches[-1].error is None
        assert group.member_status(0).outcome is Outcome.FAILED
        assert group.member_status(1).outcome is Outcome.COMPLETED
        assert group.monitor_record(0) is not None
        assert group.monitor_record(1) is None


def test_a_non_finite_norm_poisons_every_member_that_stepped(monkeypatch):
    # Each Adams member's march takes the replayed step before the group judges the
    # norms; a refused norm leaves every member's state past its last recorded step,
    # perhaps not finite: none can be resumed there (SEP 47 section 6.7).
    simulations = [member(index, "numpy_ref", FIXED_ORDER) for index in range(3)]
    with ExitStack() as cleanup:
        group, owners = group_of(simulations, cleanup)
        march = owners[1]._adams
        replay = march.replay_step
        calls = [0]

        def poisoned(dz, end, slack):
            calls[0] += 1
            norm = replay(dz, end, slack)
            return float("nan") if calls[0] == 3 else norm

        monkeypatch.setattr(march, "replay_step", poisoned)
        batches = drive(group)
    ended = batches[-1]
    assert ended.outcome is Outcome.FAILED
    assert isinstance(ended.error, BackendError)
    assert "member 1" in str(ended.error)
    # The third step opens the second window and ends at 0.75 mm, where each march
    # stands with its step unrecorded.
    for events in ended.members:
        (terminal,) = events
        assert isinstance(terminal, Terminal)
        assert terminal.outcome is Outcome.FAILED
        assert terminal.error is ended.error
        assert terminal.z == pytest.approx(PLANES[0] + STEP)
        assert terminal.eligibility is Eligibility.POISONED
        assert terminal.ticket is None


def test_a_trial_failure_partway_reports_each_member_s_own_position(monkeypatch):
    # Members take a replayed step in turn, so a failure in the second leaves the first
    # stepped but unrecorded -- its march past the step, in flight, poisoned -- and
    # the second and third untouched at the step's start: no common cursor or
    # rollback is invented (SEP 47 section 6.7).
    simulations = [member(index, "numpy_ref", FIXED_ORDER) for index in range(3)]
    with ExitStack() as cleanup:
        group, owners = group_of(simulations, cleanup)
        march = owners[1]._adams
        replay = march.replay_step
        calls = [0]

        def failing(dz, end, slack):
            calls[0] += 1
            if calls[0] == 3:
                msg = "a manufactured trial failure"
                raise RuntimeError(msg)
            return replay(dz, end, slack)

        monkeypatch.setattr(march, "replay_step", failing)
        batches = drive(group)
    ended = batches[-1]
    assert ended.outcome is Outcome.FAILED
    assert isinstance(ended.error, RuntimeError)
    terminals = [events[0] for events in ended.members]
    assert all(isinstance(terminal, Terminal) for terminal in terminals)
    assert all(terminal.error is ended.error for terminal in terminals)  # pyright: ignore[reportAttributeAccessIssue]
    # The failed step is the third, from the first plane to 0.75 mm.
    assert [terminal.z for terminal in terminals] == pytest.approx(
        [PLANES[0] + STEP, PLANES[0], PLANES[0]]
    )  # pyright: ignore[reportAttributeAccessIssue]
    assert [terminal.eligibility for terminal in terminals] == [  # pyright: ignore[reportAttributeAccessIssue]
        Eligibility.POISONED,
        Eligibility.ELIGIBLE,
        Eligibility.ELIGIBLE,
    ]
    assert terminals[0].ticket is None  # pyright: ignore[reportAttributeAccessIssue]
    assert terminals[1].ticket is not None  # pyright: ignore[reportAttributeAccessIssue]


def test_an_in_flight_failure_poisons_its_member_alone(monkeypatch):
    # An in-place step abandoned partway leaves no valid state for its member, which
    # cannot be checkpointed; the member after it, not yet stepped, can be.
    simulations = [member(index, "numpy_ref", FIXED_ORDER) for index in range(2)]
    with ExitStack() as cleanup:
        group, owners = group_of(simulations, cleanup)
        march = owners[0]._adams

        def abandoned(dz, end, slack):
            owners[0]._in_flight = (owners[0]._z, dz)
            msg = "a manufactured in-flight failure"
            raise RuntimeError(msg)

        monkeypatch.setattr(march, "replay_step", abandoned)
        launch = group.advance()
        group.acknowledge(launch.receipt)
        ended = group.advance()
        poisoned, untouched = (events[0] for events in ended.members)
        assert isinstance(poisoned, Terminal) and isinstance(untouched, Terminal)
        assert poisoned.eligibility is Eligibility.POISONED
        assert poisoned.ticket is None
        assert poisoned.z == 0.0
        assert untouched.eligibility is Eligibility.ELIGIBLE
        assert untouched.ticket is not None
        assert untouched.ticket.z == untouched.z == 0.0
        # The untouched member can still be checkpointed where it stands.
        field = group.export(1, untouched.ticket, IDENTITY)
        assert np.all(np.isfinite(field))
        group.acknowledge(ended.receipt)
        assert group.finished


def test_a_window_s_failure_delivers_the_attempts_it_drained(monkeypatch):
    # The statistics fail after each member's attempts of the window were drained:
    # each terminal delivers them, and the status counts no attempt it did not.
    with ExitStack() as cleanup:
        group, owners = group_of(
            [member(index, "numpy_ref") for index in range(2)], cleanup
        )
        launch = group.advance()
        group.acknowledge(launch.receipt)

        def broken(self, spec):
            msg = "manufactured statistics failure"
            raise RuntimeError(msg)

        monkeypatch.setattr(type(owners[0]), "stats", broken)
        ended = group.advance()
        assert ended.outcome is Outcome.FAILED
        assert isinstance(ended.error, RuntimeError)
        for index, events in enumerate(ended.members):
            (terminal,) = events
            assert isinstance(terminal, Terminal)
            assert terminal.history is not None
            # The first window's two fixed steps, delivered once.
            assert terminal.history.n_steps == 2
            assert group.member_status(index).attempts == 2
            assert terminal.z == pytest.approx(PLANES[0])
            assert terminal.eligibility is Eligibility.ELIGIBLE
        group.acknowledge(ended.receipt)


def test_a_group_wide_stop_ends_every_member_at_a_common_endpoint():
    token = StopToken()
    with ExitStack() as cleanup:
        group, _ = group_of(
            [member(index, "numpy_ref") for index in range(2)], cleanup, stop=token
        )
        launch = group.advance()
        group.acknowledge(launch.receipt)
        token.set()
        stopped = group.advance()
        group.acknowledge(stopped.receipt)
        assert group.finished
    assert stopped.outcome is Outcome.STOPPED
    for events in stopped.members:
        (terminal,) = events
        assert isinstance(terminal, Terminal)
        assert terminal.outcome is Outcome.STOPPED
        assert terminal.z == 0.0
        assert terminal.eligibility is Eligibility.ELIGIBLE
