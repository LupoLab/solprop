# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The group protocol (SEP 47 section 6.7), on the reference's group and the native one.

A group advances compatible members through one fixed schedule, one common window per
save plane, and reports each member's own events. A boundary stop ends its member
alone; a non-finite norm, a trial exception or an in-flight failure ends the group,
and each member then reports its own committed position and checkpoint eligibility:
members are stepped in turn, so one ahead of the failing member has already advanced
past the failed step.

The conformance tests run on both backends: ``numpy_ref``, whose group is its own over
its lockstep runner (:class:`~solprop.backends.host.group_session.HostGroupSession`),
and ``native``, whose group is the native group session
(:class:`solprop_native.group.NativeGroupSession`, one native session per member).
The failures that reach into the reference's Python march to inject a fault are the
reference's alone; the native group's own failure positions, with natively injected
faults, are in ``test_native_group.py``.
"""

import warnings
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from bounds import assert_within
from driver.test_field_free_planes import cartesian
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
from solprop.exceptions import (
    BackendError,
    BoundaryError,
    BoundaryWarning,
    StopRequested,
)
from solprop.log import StopFlag
from solprop.monitor import BoundaryCheck, BoundaryMonitor
from solprop.stats import StatsSpec
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule

PLANES = (0.0005, 0.001)
"""The save planes, m: four fixed steps of 0.25 mm, two per window."""

STEP = 0.00025
"""The fixed step, m."""

FIELD_BOUND = 4 * 16 * np.finfo(float).eps
"""The singleton ladder's 16 eps a step over four steps: a member and its lone run take
one method on one schedule and differ by the order of their reductions alone
(``tests/backends/test_batch_lockstep.py``)."""

IDENTITY = ExportRecipe(power=False, optics=False)
CONSUMER = ExportRecipe(power=True, optics=False)

BACKENDS = ["numpy_ref", "native"]

STRATEGIES = {
    "numpy_ref": "numpy_stage_arrays",
    "native": "native_group_session",
}
"""Each group, as its members' execution records name it."""

GROUPS = BACKENDS
"""Every group implementation: the reference's and the native group session."""

LOCKSTEP = ["numpy_ref"]
"""The groups over a lockstep runner, whose runner and Python march a test patches:
the reference's alone; the native group's failures are injected natively
(``test_native_group.py``)."""


def member(
    index: int, backend: str, integrator: Any = None, **overrides: Any
) -> Simulation:
    """Return one member: a soliton-order pulse on its own fibre, on the schedule.

    `overrides` replace any of the member's settings (``Simulation.from_gnlse``).
    """
    settings: dict[str, Any] = {
        "wavelength": 835e-9,
        "wavelength_limits": (500e-9, 2e-6),
        "trange": 1e-12,
        "width": 100e-15,
        "peak_power": float(index + 1),
        "shape": "sech",
        "save_planes": list(PLANES),
        "schedule": FixedSchedule.uniform(0.0, PLANES[-1], STEP, save_planes=PLANES),
        "stats": StatsSpec(
            names=("energy", "peak_power"), periodic=("energy",), period=2
        ),
        "noise": None,
        "backend": backend,
        "integrator": RungeKutta() if integrator is None else integrator,
        "absorber": False,
    }
    return Simulation.from_gnlse(
        0.11 * (index + 1),
        PLANES[-1],
        [0.0, 0.0, -1.183e-26 * (index + 1)],
        **(settings | overrides),
    )


def constructed(simulation: Simulation, cleanup: ExitStack) -> Any:
    """Return a member's backend, constructed and holding no state."""
    owner = select(
        simulation.settings.backend,
        simulation.translate(),
        simulation._run_request(),
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
    stop: StopFlag | None = None,
) -> tuple[GroupSession, list[Any]]:
    """Start a group of members at their entrances; return it and its members."""
    problems = [one.translate() for one in simulations]
    owners = [constructed(one, cleanup) for one in simulations]
    watched = monitors if monitors is not None else [None] * len(problems)
    group = start_group(
        owners,
        problems,
        [
            one._run_request().for_run(monitor=monitor)
            for one, monitor in zip(simulations, watched, strict=True)
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


@pytest.mark.parametrize("backend", GROUPS)
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
            record = group.execution_record(index)
            assert record["batch_size"] == 2
            assert record["batch_strategy"] == STRATEGIES[backend]
    assert windows[:3] == [
        [ObservationKind.LAUNCH] * 2,
        [ObservationKind.PLANE] * 2,
        [ObservationKind.PLANE] * 2,
    ]
    for index, simulation in enumerate(simulations):
        problem = simulation.translate()
        run_request = simulation._run_request()
        with ExitStack() as cleanup:
            owner = constructed(simulation, cleanup)
            lone = start_session(owner, problem, run_request, Start.entrance(problem))
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


@pytest.mark.parametrize("backend", GROUPS)
def test_a_group_s_receipts_and_tickets_follow_the_session_lifecycle(backend):
    available(backend)
    with ExitStack() as cleanup:
        group, _ = group_of([member(index, backend) for index in range(2)], cleanup)
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
        # Every member's ticket names one batch and one generation: each names its
        # member too, and another member's is refused.
        (other,) = batch.members[1]
        assert isinstance(other, ObservationReady)
        assert (plane.ticket.member, other.ticket.member) == (0, 1)
        with pytest.raises(SessionError, match="member 0's, not member 1's"):
            group.export(1, plane.ticket, IDENTITY)
        with pytest.raises(SessionError, match="member 1's, not member 0's"):
            group.noise_phase(0, other.ticket)
        group.acknowledge(batch.receipt)
        group.close()
        assert group.closed
        assert group.member_status(0).state is SessionState.CLOSED
        group.close()


@pytest.mark.parametrize("backend", GROUPS)
def test_a_group_starts_its_members_fresh_and_one_start_each(backend):
    available(backend)
    simulations = [member(index, backend) for index in range(2)]
    problems = [one.translate() for one in simulations]
    with ExitStack() as cleanup:
        owners = [constructed(one, cleanup) for one in simulations]
        requests = [one._run_request() for one in simulations]
        with pytest.raises(SessionError, match="one problem, request and start"):
            start_group(owners, problems, requests, [Start.entrance(problems[0])])
        resumed = Start.entrance(problems[1])
        resumed.plane_index = 1
        with pytest.raises(SessionError, match="at their entrance"):
            start_group(
                owners, problems, requests, [Start.entrance(problems[0]), resumed]
            )
        # The launch plane is owed: a start that has observed it is not an entrance.
        observed = Start.entrance(problems[1])
        observed.plane_pending = False
        with pytest.raises(SessionError, match="its launch plane owed"):
            start_group(
                owners, problems, requests, [Start.entrance(problems[0]), observed]
            )


@pytest.mark.parametrize("backend", GROUPS)
def test_a_schedule_past_the_last_plane_completes_without_a_step(backend):
    # The completion takes no step, whatever the schedule holds beyond the last plane,
    # so a stop requested after the last plane finds no step to stop at: the members
    # complete there, as each would run alone.
    available(backend)
    schedule = FixedSchedule.uniform(0.0, 2 * PLANES[-1], STEP, save_planes=PLANES)
    simulations = [member(index, backend, schedule=schedule) for index in range(2)]
    token = StopToken()
    with ExitStack() as cleanup:
        group, _ = group_of(simulations, cleanup, stop=token)
        for _ in range(len(PLANES) + 1):
            window = group.advance()
            group.acknowledge(window.receipt)
        token.set()
        completion = group.advance()
        group.acknowledge(completion.receipt)
        assert group.finished
    assert completion.outcome is Outcome.COMPLETED
    for events in completion.members:
        (terminal,) = events
        assert isinstance(terminal, Terminal)
        assert terminal.outcome is Outcome.COMPLETED
        assert terminal.z == pytest.approx(PLANES[-1])
        assert terminal.history is not None
        assert terminal.history.n_steps == 0


@pytest.mark.parametrize("backend", GROUPS)
def test_a_boundary_stop_ends_its_member_alone_and_the_others_go_on(backend):
    # A check strict enough to refuse any launch stops the first member at once; the
    # second, unwatched, runs the whole schedule (SEP 21). The reference's group
    # attaches the monitors before the launch plane, as the batched scan always did,
    # so the stop ends the launch window; a native member commits its launch plane
    # first and its monitor then judges the launch (SEP 47 section 5.3's start), so
    # the stop ends the next window, still at the launch.
    available(backend)
    simulations = [member(index, backend) for index in range(2)]
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
        assert (ended, stopped.z) == (1 if backend == "native" else 0, 0.0)
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


@pytest.mark.parametrize("backend", LOCKSTEP)
def test_a_non_finite_norm_poisons_every_member_that_stepped(backend, monkeypatch):
    available(backend)
    # Each Adams member's march takes the replayed step before the group judges the
    # norms; a refused norm leaves every member's state past its last recorded step,
    # perhaps not finite: none can be resumed there (SEP 47 section 6.7).
    simulations = [member(index, backend, FIXED_ORDER) for index in range(3)]
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


@pytest.mark.parametrize("backend", LOCKSTEP)
def test_a_trial_failure_partway_reports_each_member_s_own_position(
    backend, monkeypatch
):
    available(backend)
    # Members take a replayed step in turn, so a failure in the second leaves the first
    # stepped but unrecorded -- its march past the step, in flight, poisoned -- and
    # the second and third untouched at the step's start: no common cursor or
    # rollback is invented (SEP 47 section 6.7).
    simulations = [member(index, backend, FIXED_ORDER) for index in range(3)]
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


@pytest.mark.parametrize("backend", LOCKSTEP)
def test_an_in_flight_failure_poisons_its_member_alone(backend, monkeypatch):
    available(backend)
    # An in-place step abandoned partway leaves no valid state for its member, which
    # cannot be checkpointed; the member after it, not yet stepped, can be.
    simulations = [member(index, backend, FIXED_ORDER) for index in range(2)]
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


def assert_drained_after_failure(group, ended):
    """A failed window's terminals deliver its two fixed steps, once, at its plane."""
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


@pytest.mark.parametrize("backend", LOCKSTEP)
def test_a_window_s_failure_delivers_the_attempts_it_drained(backend, monkeypatch):
    available(backend)
    # The statistics fail after each member's attempts of the window were drained:
    # each terminal delivers them, and the status counts no attempt it did not.
    with ExitStack() as cleanup:
        group, owners = group_of(
            [member(index, backend) for index in range(2)], cleanup
        )
        launch = group.advance()
        group.acknowledge(launch.receipt)

        def broken(self, spec):
            msg = "manufactured statistics failure"
            raise RuntimeError(msg)

        monkeypatch.setattr(type(owners[0]), "stats", broken)
        ended = group.advance()
        assert_drained_after_failure(group, ended)
        group.acknowledge(ended.receipt)


@pytest.mark.parametrize("backend", GROUPS)
def test_a_group_wide_stop_ends_every_member_at_a_common_endpoint(backend):
    available(backend)
    token = StopToken()
    with ExitStack() as cleanup:
        group, _ = group_of(
            [member(index, backend) for index in range(2)], cleanup, stop=token
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
        assert isinstance(terminal.error, StopRequested)


def runner_of(group: GroupSession) -> Any:
    """Return the lockstep runner a group drives, whose class a test patches."""
    return cast("Any", group)._runner


@dataclass
class FinalWarning(BoundaryMonitor):
    """A boundary monitor whose final check warns, as a removed fraction over budget."""

    def finish(self) -> None:
        """Warn as the end-of-run check does when the collar removed too much."""
        warnings.warn(
            BoundaryWarning("a manufactured final warning", parameter="boundary_check"),
            stacklevel=2,
        )


def final_warning(simulation: Simulation) -> FinalWarning:
    """Return a fresh :class:`FinalWarning` for a member's grid."""
    built = BoundaryMonitor.build(simulation.grid, None, BoundaryCheck())
    return FinalWarning(
        built.check,
        built.weights,
        built.kind,
        built.columns,
        built.split,
        built.interval,
    )


@pytest.mark.parametrize("backend", GROUPS)
def test_a_group_closes_with_its_scope(backend):
    available(backend)
    simulations = [member(index, backend) for index in range(2)]
    with ExitStack() as cleanup:
        group, _ = group_of(simulations, cleanup)
        with group as scoped:
            assert scoped is group
            scoped.acknowledge(scoped.advance().receipt)
        assert group.closed
        assert group.member_status(1).state is SessionState.CLOSED


@pytest.mark.parametrize("backend", GROUPS)
def test_a_spatial_group_exports_its_field_measure_and_each_member_s_noise(backend):
    # A spatial route's field measure is already the stored quantity: a true-power
    # recipe exports it unscaled, and each member carries its own noise realisation.
    available(backend)
    simulations = [cartesian(backend, noise=True) for _ in range(2)]
    with ExitStack() as cleanup:
        group, _ = group_of(simulations, cleanup)
        launch = group.advance()
        for index, events in enumerate(launch.members):
            (event,) = events
            assert isinstance(event, ObservationReady)
            field = np.array(group.export(index, event.ticket, IDENTITY))
            measured = group.export(index, event.ticket, CONSUMER)
            assert measured.tobytes() == field.tobytes()
            phase = group.noise_phase(index, event.ticket)
            assert phase is not None
            assert phase.shape == field.shape
            assert np.all(np.isfinite(phase))
        group.acknowledge(launch.receipt)


@pytest.mark.parametrize("backend", GROUPS)
def test_a_noise_free_member_has_no_noise_phase(backend):
    available(backend)
    with ExitStack() as cleanup:
        group, _ = group_of([member(index, backend) for index in range(2)], cleanup)
        launch = group.advance()
        (event,) = launch.members[1]
        assert isinstance(event, ObservationReady)
        assert group.noise_phase(1, event.ticket) is None
        group.acknowledge(launch.receipt)


@pytest.mark.parametrize("backend", LOCKSTEP)
def test_a_launch_the_runner_cannot_watch_fails_every_member_at_the_entrance(
    backend, monkeypatch
):
    available(backend)
    with ExitStack() as cleanup:
        group, _ = group_of([member(index, backend) for index in range(2)], cleanup)

        def refused(self, monitors):
            msg = "manufactured monitor attachment failure"
            raise RuntimeError(msg)

        monkeypatch.setattr(type(runner_of(group)), "attach_monitors", refused)
        ended = group.advance()
        group.acknowledge(ended.receipt)
        assert group.finished
    assert ended.outcome is Outcome.FAILED
    assert isinstance(ended.error, RuntimeError)
    for events in ended.members:
        (terminal,) = events
        assert isinstance(terminal, Terminal)
        assert terminal.error is ended.error
        assert terminal.z == 0.0
        assert terminal.eligibility is Eligibility.ELIGIBLE
        assert terminal.ticket is not None


@pytest.mark.parametrize("backend", LOCKSTEP)
def test_a_window_s_statistics_failure_delivers_the_attempts_it_drained(
    backend, monkeypatch
):
    # The group's statistics fail at the window's end, after each member's attempts
    # of the window were drained: each terminal delivers them, once.
    available(backend)
    with ExitStack() as cleanup:
        group, _ = group_of([member(index, backend) for index in range(2)], cleanup)
        launch = group.advance()
        group.acknowledge(launch.receipt)

        def broken(self, spec):
            msg = "manufactured group statistics failure"
            raise RuntimeError(msg)

        monkeypatch.setattr(type(runner_of(group)), "stats", broken)
        ended = group.advance()
        group.acknowledge(ended.receipt)
    assert_drained_after_failure(group, ended)


@pytest.mark.parametrize("backend", LOCKSTEP)
def test_a_member_unreadable_after_a_failure_still_reports_its_terminal(
    backend, monkeypatch
):
    # After a group-wide failure each member is drained alone; one whose state cannot
    # be read delivers no attempts, and the others' terminals are unaffected.
    available(backend)
    with ExitStack() as cleanup:
        group, owners = group_of(
            [member(index, backend) for index in range(2)], cleanup
        )
        launch = group.advance()
        group.acknowledge(launch.receipt)

        def failed(self, target):
            msg = "manufactured window failure"
            raise RuntimeError(msg)

        def unreadable(*, field=True):
            msg = "manufactured unreadable member"
            raise RuntimeError(msg)

        monkeypatch.setattr(type(runner_of(group)), "propagate_to", failed)
        monkeypatch.setattr(owners[1], "observe", unreadable)
        ended = group.advance()
        group.acknowledge(ended.receipt)
    assert ended.outcome is Outcome.FAILED
    terminals = [events[0] for events in ended.members]
    assert all(isinstance(terminal, Terminal) for terminal in terminals)
    for terminal in terminals:
        assert isinstance(terminal, Terminal)
        assert terminal.error is ended.error
        assert terminal.z == 0.0
        assert terminal.history is not None
        assert terminal.history.n_steps == 0


@pytest.mark.parametrize("backend", GROUPS)
def test_a_group_whose_members_all_stop_at_their_launch_completes(backend):
    # Every member ends at its own boundary at the launch: nothing is left to advance.
    # A native member's monitor judges its launch after the launch plane, one window
    # on (SEP 47 section 5.3).
    available(backend)
    simulations = [member(index, backend) for index in range(2)]
    strict = BoundaryCheck(budget=1e-300, stop_factor=1.0)
    with ExitStack() as cleanup:
        group, _ = group_of(
            simulations,
            cleanup,
            monitors=[
                BoundaryMonitor.build(one.grid, None, strict) for one in simulations
            ],
        )
        batches = drive(group)
        assert len(batches) == (2 if backend == "native" else 1)
        batch = batches[-1]
    assert batch.outcome is Outcome.COMPLETED
    assert batch.error is None
    for events in batch.members:
        (stopped,) = events
        assert stopped.z == 0.0
        assert isinstance(stopped, Terminal)
        assert stopped.outcome is Outcome.FAILED
        assert isinstance(stopped.error, BoundaryError)
        assert stopped.eligibility is Eligibility.ELIGIBLE


@pytest.mark.parametrize("backend", GROUPS)
def test_a_group_wide_stop_reports_nothing_more_for_a_member_already_ended(backend):
    available(backend)
    simulations = [member(index, backend) for index in range(2)]
    strict = BoundaryCheck(budget=1e-300, stop_factor=1.0)
    token = StopToken()
    with ExitStack() as cleanup:
        group, _ = group_of(
            simulations,
            cleanup,
            monitors=[BoundaryMonitor.build(simulations[0].grid, None, strict), None],
            stop=token,
        )
        launch = group.advance()
        if backend == "native":
            # The launch plane first, then the monitor's launch judgement.
            group.acknowledge(launch.receipt)
            launch = group.advance()
        (refused,) = launch.members[0]
        assert isinstance(refused, Terminal)
        assert isinstance(refused.error, BoundaryError)
        group.acknowledge(launch.receipt)
        token.set()
        stopped = group.advance()
        group.acknowledge(stopped.receipt)
        assert group.finished
    assert stopped.outcome is Outcome.STOPPED
    assert stopped.members[0] == ()
    (terminal,) = stopped.members[1]
    assert isinstance(terminal, Terminal)
    assert terminal.outcome is Outcome.STOPPED
    assert group.member_status(0).outcome is Outcome.FAILED
    assert group.member_status(1).outcome is Outcome.STOPPED


@pytest.mark.parametrize("backend", LOCKSTEP)
def test_a_failed_final_check_ends_the_group_failed(backend, monkeypatch):
    available(backend)
    with ExitStack() as cleanup:
        group, _ = group_of([member(index, backend) for index in range(2)], cleanup)

        def broken(self):
            msg = "manufactured final check failure"
            raise RuntimeError(msg)

        monkeypatch.setattr(type(runner_of(group)), "finish_monitors", broken)
        batches = drive(group)
    ended = batches[-1]
    assert ended.outcome is Outcome.FAILED
    assert isinstance(ended.error, RuntimeError)
    for events in ended.members:
        (terminal,) = events
        assert isinstance(terminal, Terminal)
        assert terminal.outcome is Outcome.FAILED
        assert terminal.z == pytest.approx(PLANES[-1])
        assert terminal.eligibility is Eligibility.ELIGIBLE


@pytest.mark.parametrize("backend", LOCKSTEP)
def test_a_final_warning_raised_as_an_error_fails_its_member_alone(backend):
    # A run of the member alone would fail at its final check; in a group it fails
    # that member, and the other completes (SEP 21). The tight window's own warnings
    # are not what is tested.
    available(backend)
    simulations = [member(index, backend) for index in range(2)]
    with ExitStack() as cleanup, warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        warnings.filterwarnings(
            "error", ".*a manufactured final warning", BoundaryWarning
        )
        group, _ = group_of(
            simulations, cleanup, monitors=[final_warning(simulations[0]), None]
        )
        batches = drive(group)
    last = batches[-1]
    assert last.outcome is Outcome.COMPLETED
    (failed,) = last.members[0]
    assert isinstance(failed, Terminal)
    assert failed.outcome is Outcome.FAILED
    assert isinstance(failed.error, BoundaryWarning)
    assert failed.eligibility is Eligibility.ELIGIBLE
    assert failed.z == pytest.approx(PLANES[-1])
    (completed,) = last.members[1]
    assert isinstance(completed, Terminal)
    assert completed.outcome is Outcome.COMPLETED


@pytest.mark.parametrize("backend", LOCKSTEP)
def test_a_final_warning_raised_as_an_error_fails_the_member_s_lone_run(backend):
    # The lone counterpart of the test above: the member's own session fails at its
    # final check, at the run's end, with its state eligible for a checkpoint.
    available(backend)
    simulation = member(0, backend)
    problem = simulation.translate()
    with ExitStack() as cleanup, warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        warnings.filterwarnings(
            "error", ".*a manufactured final warning", BoundaryWarning
        )
        lone = start_session(
            constructed(simulation, cleanup),
            problem,
            simulation._run_request().for_run(monitor=final_warning(simulation)),
            Start.entrance(problem),
            owns=False,
        )
        cleanup.callback(lone.close)
        batch = None
        while not lone.finished:
            batch = lone.advance()
            lone.acknowledge(batch.receipt)
    assert batch is not None
    (failed,) = batch.events
    assert isinstance(failed, Terminal)
    assert failed.outcome is Outcome.FAILED
    assert isinstance(failed.error, BoundaryWarning)
    assert failed.eligibility is Eligibility.ELIGIBLE
    assert failed.z == pytest.approx(PLANES[-1])
