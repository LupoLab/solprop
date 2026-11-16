# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference's marches own their state and ask their session narrowly (SEP 52).

Each march keeps its position, proposal, cursor and accepted norms in its own
:class:`~solprop.backends.host.marches.MarchState`, hands state-sized work to its
engine and asks the session only through
:class:`~solprop.backends.host.marches.MarchHost`. Here the engine and the session are
recorders: what a march calls, in which order, and what it hands on, without any field
arithmetic. The decisions themselves are the trace corpora's
(``test_rk_traces.py``, ``test_adams_traces.py``).
"""

import math
from typing import Any, cast

import pytest

from solprop.backends.engines import AcceptRecord
from solprop.backends.host.low_storage import HostLowStorageMarch
from solprop.backends.host.marches import MarchState, in_flight_refusal
from solprop.backends.host.runge_kutta import RungeKuttaMarch
from solprop.backends.records import RungeKuttaPolicy
from solprop.exceptions import BackendError, ConvergenceError
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, PIController


class Host:
    """A session that records what a march asks of it."""

    collar_quiet = False

    def __init__(self, *, natural_every: int = 0) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.records: list[AcceptRecord] = []
        self.natural_every = natural_every
        self.stepped_count = 0

    def poll_stop(self) -> None:
        self.calls.append(("poll",))

    def cap(self, proposal: float, z: float) -> float:
        self.calls.append(("cap", z))
        return proposal

    def record(self, end: float, dz: float, norm: float, accepted: bool) -> None:
        self.calls.append(("record", end, accepted))

    def absorb(self, record: AcceptRecord) -> None:
        self.calls.append(("absorb",))
        self.records.append(record)

    def stepped(self) -> None:
        self.calls.append(("stepped",))
        self.stepped_count += 1

    def natural_stop(self, target: float) -> bool:
        return bool(self.natural_every) and self.stepped_count % self.natural_every == 0


class Engine:
    """A Runge-Kutta engine that scores trials from a list and records its calls."""

    def __init__(self, norms: list[float], *, diverge_at: int | None = None) -> None:
        self.norms = list(norms)
        self.calls: list[tuple[Any, ...]] = []
        self.diverge_at = diverge_at
        self.trials = 0
        self.flight = False

    @property
    def in_flight(self) -> bool:
        return self.flight

    def trial(self, z: float, dz: float) -> float:
        self.trials += 1
        self.calls.append(("trial", z, dz))
        self.flight = True
        if self.trials == self.diverge_at:
            return math.inf
        return self.norms.pop(0) if self.norms else 0.5

    def accept(self, dz: float) -> AcceptRecord:
        self.calls.append(("accept", dz))
        self.flight = False
        return AcceptRecord(removed=float(len(self.calls)), stiffness=0.25)

    def extend(self, z: float, dz: float) -> None:
        raise NotImplementedError

    def reading(self) -> Any:
        raise NotImplementedError


def _policy() -> RungeKuttaPolicy:
    return RungeKuttaPolicy(
        controller=PIController(rtol=1e-6, atol=0.0),
        dz_min=0.0,
        dz_max=1.0,
        auto_init=False,
        stretch=1.0,
        z_end=1.0,
        save_planes=(1.0,),
        interpolate=False,
        hard_kinks=(),
        breakpoints=(0.0, 1.0),
        landing_ulps=8,
        start_growth=100.0,
        quiet_interval=None,
    )


def test_a_march_state_starts_afresh_at_a_load():
    state = MarchState(z=0.3, proposal=0.1, cursor=4)
    state.norms.extend([1.0, 2.0])
    state.load(0.5, 0.2, 7)
    assert (state.z, state.proposal, state.cursor) == (0.5, 0.2, 7)
    assert list(state.norms) == []


def test_the_runge_kutta_march_keeps_its_state_and_asks_in_order():
    engine, host = Engine([2.0, 0.5]), Host()
    march = RungeKuttaMarch(engine, _policy(), host, schedule=None, proposal=0.25)
    march.load(0.0, 0.25, 0)
    # A load starts the march's own state afresh and asks nothing of the engine: each
    # run builds its engine afresh (SEP 52).
    assert engine.calls == []
    march.march_to(0.25)
    assert march.state.z == 0.25
    assert len(march.state.norms) >= 1
    # A rejected trial is recorded and never accepted or absorbed; the accepted one
    # is accepted, absorbed and accounted after the position moved.
    kinds = [call[0] for call in host.calls]
    assert kinds[:4] == ["poll", "cap", "record", "poll"]
    assert host.calls[2] == ("record", 0.25, False)
    assert kinds[4:8] == ["cap", "record", "absorb", "stepped"]
    assert [call[0] for call in engine.calls[:3]] == ["trial", "trial", "accept"]
    # What the accept reported is what the session absorbs.
    assert host.records[0].removed == 3.0


def test_a_replayed_runge_kutta_step_that_diverges_commits_nothing():
    engine, host = Engine([], diverge_at=2), Host()
    schedule = FixedSchedule.uniform(0.0, 1.0, 0.25)
    march = RungeKuttaMarch(engine, _policy(), host, schedule=schedule, proposal=0.25)
    march.load(0.0, 0.25, 0)
    with pytest.raises(ConvergenceError, match="non-finite error norm"):
        march.advance(1.0)
    assert march.state.cursor == 1
    assert march.state.z == 0.25
    assert [call[0] for call in engine.calls].count("accept") == 1


def test_the_low_storage_march_reports_a_step_left_in_flight():
    engine, host = Engine([], diverge_at=2), Host()
    schedule = FixedSchedule.uniform(0.0, 1.0, 0.25)
    march = HostLowStorageMarch(
        engine, LowStorageRungeKutta(), host, schedule=schedule, proposal=0.25
    )
    march.load(0.0, 0.25, 0)
    with pytest.raises(ConvergenceError, match="non-finite state"):
        march.advance(1.0)
    # The first step committed with its evaluations; the second is in flight.
    assert march.state.z == 0.25
    assert march.source_evaluations == LowStorageRungeKutta().evaluations
    assert march.in_flight == (0.25, 0.25)
    assert march.refusal() == in_flight_refusal("numpy_ref", 0.25, 0.25)
    with pytest.raises(BackendError, match="abandoned in flight"):
        march.replay_step(0.25, 0.5, 0.0)
    # A load clears it, and the count is the loaded run's.
    march.load(0.0, 0.25, 0)
    assert march.in_flight is None
    assert march.source_evaluations == 0


def test_the_low_storage_march_hands_the_stiffness_from_the_accept(caplog):
    engine, host = Engine([]), Host()
    schedule = FixedSchedule.uniform(0.0, 0.5, 0.25)
    method = LowStorageRungeKutta()
    march = HostLowStorageMarch(engine, method, host, schedule=schedule, proposal=0.25)
    march.load(0.0, 0.25, 0)
    march.advance(0.5)
    assert march.state.z == 0.5
    assert len(host.records) == 2
    assert all(record.stiffness == 0.25 for record in host.records)
    assert "stiffness indicator" not in caplog.text


def test_an_adams_floor_after_a_rejection_keeps_a_failed_release_as_a_note():
    from solprop.backends.host.adams import HostAdamsMarch, _Attempt  # noqa: PLC0415

    ended = []

    class Failing:
        def begin_attempt(self, cost: int) -> None:
            pass

        def end_attempt(self, outcome: Any) -> None:
            ended.append(outcome)
            raise BackendError("release failed")

    march = cast("Any", object.__new__(HostAdamsMarch))
    march._floor = ConvergenceError("the floor")
    attempt = _Attempt(Failing())
    attempt.begin(2)
    with pytest.raises(ConvergenceError, match="the floor") as caught:
        march._end(attempt, accepted=False)
    assert caught.value.__notes__ == [
        "releasing the source reservation failed: release failed"
    ]
    assert not attempt.open
    assert len(ended) == 1
    # Accepted, the end's failure is the error, before any floor.
    march._floor = ConvergenceError("the floor")
    attempt = _Attempt(Failing())
    attempt.begin(1)
    with pytest.raises(BackendError, match="release failed"):
        march._end(attempt, accepted=True)
