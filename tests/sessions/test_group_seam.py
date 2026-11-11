# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A backend registers with one row and runs its own group through the seam (SEP 47).

``toy`` is a test-only backend: the reference's arithmetic under another registry
name, with a prepared group of its own and its own lockstep runner. It is registered
with one :class:`~solprop.backends.registry.BackendEntry`, nothing else in solprop
names it, and :func:`solprop.backends.registry.select`,
:func:`~solprop.backends.registry.select_batch` and
:func:`solprop.backends.sessions.start_group` reach it through that row and its
``prepare_group`` alone (SEP 47 sections 6.1, 6.7).
"""

from collections.abc import Sequence
from contextlib import ExitStack
from typing import Any, ClassVar

import numpy as np
import pytest

from bounds import assert_within
from sessions.test_group import FIELD_BOUND, IDENTITY, PLANES, drive, member
from solprop.backends import registry
from solprop.backends.events import ObservationReady, Outcome
from solprop.backends.host.driver import HostDriver
from solprop.backends.host.group import HostGroup
from solprop.backends.host.group_session import HostGroupSession, HostPreparedGroup
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.protocol import Backend, BatchBackend
from solprop.backends.registry import BackendEntry, select, select_batch
from solprop.backends.session import GroupSession, PreparedGroup, Start
from solprop.backends.sessions import start_group, start_session
from solprop.driver import Simulation
from solprop.exceptions import BackendError
from solprop.log import StopFlag
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest


class Toy(NumpyRefBackend):
    """The reference's arithmetic under another name, preparing its own group."""

    name = "toy"

    def prepare_group(
        self,
        members: Sequence[Backend],
        problems: Sequence[PropagationProblem],
        requests: Sequence[RunRequest],
    ) -> PreparedGroup:
        """Prepare the toy's own group."""
        return ToyGroup(members, problems, requests)


class ToyGroup(HostPreparedGroup):
    """The toy's prepared group: the group over its lockstep runner, counted."""

    started: ClassVar[list[int]] = []

    def start(
        self, starts: Sequence[Start], *, stop: StopFlag | None = None
    ) -> HostGroupSession:
        """Record the start and open the group."""
        ToyGroup.started.append(len(starts))
        return super().start(starts, stop=stop)


def lockstep(owners: Sequence[Backend]) -> BatchBackend:
    """Return the toy's lockstep runner: each member's own trial, in turn."""
    return HostGroup([member for member in owners if isinstance(member, HostDriver)])


TOY = BackendEntry(
    requires=(),
    extras=(),
    factory=f"{__name__}:Toy",
    batch=f"{__name__}:lockstep",
)
"""The toy's one registry row."""


@pytest.fixture
def toy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the toy for one test."""
    monkeypatch.setitem(registry.BACKENDS, "toy", TOY)
    monkeypatch.setattr(ToyGroup, "started", [])


def constructed(name: str, simulation: Simulation, cleanup: ExitStack) -> Any:
    """Return a member's backend of the named row, constructed and holding no state."""
    owner = select(
        name,
        simulation.translate(),
        simulation._run_request(),
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    cleanup.callback(owner.close)
    return owner


def fields_of(name: str, simulations: Sequence[Simulation]) -> list[list[np.ndarray]]:
    """Run the members as one group of the named row; return each member's planes."""
    problems = [one.translate() for one in simulations]
    fields: list[list[np.ndarray]] = [[] for _ in simulations]
    with ExitStack() as cleanup:
        owners = [constructed(name, one, cleanup) for one in simulations]
        group: GroupSession = start_group(
            owners,
            problems,
            [one._run_request() for one in simulations],
            [Start.entrance(problem) for problem in problems],
        )
        cleanup.callback(group.close)
        outcome = None
        while not group.finished:
            batch = group.advance()
            outcome = batch.outcome
            for index, events in enumerate(batch.members):
                for event in events:
                    if isinstance(event, ObservationReady):
                        fields[index].append(
                            np.array(group.export(index, event.ticket, IDENTITY))
                        )
            group.acknowledge(batch.receipt)
        assert outcome is Outcome.COMPLETED
        for index in range(len(simulations)):
            status = group.member_status(index)
            assert status.outcome is Outcome.COMPLETED
            assert status.z == pytest.approx(PLANES[-1])
    return fields


def alone(simulation: Simulation) -> list[np.ndarray]:
    """Run one member alone on the reference; return its planes."""
    problem = simulation.translate()
    planes = []
    with ExitStack() as cleanup:
        owner = constructed("numpy_ref", simulation, cleanup)
        lone = start_session(
            owner, problem, simulation._run_request(), Start.entrance(problem)
        )
        cleanup.callback(lone.close)
        while not lone.finished:
            batch = lone.advance()
            for event in batch.events:
                if isinstance(event, ObservationReady):
                    planes.append(np.array(lone.export(event.ticket, IDENTITY)))
            lone.acknowledge(batch.receipt)
    return planes


@pytest.mark.usefixtures("toy")
def test_a_registered_backend_runs_its_own_group_through_the_seam():
    simulations = [member(index, "numpy_ref") for index in range(2)]
    with ExitStack() as cleanup:
        owner = constructed("toy", simulations[0], cleanup)
        assert type(owner) is Toy
    toy = fields_of("toy", simulations)
    # The toy prepared and started its own group, once, over its own runner.
    assert ToyGroup.started == [2]
    reference = [alone(one) for one in simulations]
    for index, (planes, expected) in enumerate(zip(toy, reference, strict=True)):
        assert len(planes) == len(expected) == len(PLANES) + 1
        for one, other in zip(planes, expected, strict=True):
            assert_within(one, other, FIELD_BOUND, what=f"member {index}")


@pytest.mark.usefixtures("toy")
def test_the_lockstep_runner_is_the_registered_row_s(monkeypatch):
    simulations = [member(index, "numpy_ref") for index in range(2)]
    problems = [one.translate() for one in simulations]
    with ExitStack() as cleanup:
        owners = [constructed("toy", one, cleanup) for one in simulations]
        group = start_group(
            owners,
            problems,
            [one._run_request() for one in simulations],
            [Start.entrance(problem) for problem in problems],
        )
        cleanup.callback(group.close)
        assert type(getattr(group, "_runner")) is HostGroup  # noqa: B009 - private
        drive(group)
    monkeypatch.setitem(registry.BACKENDS, "toy", BackendEntry((), (), TOY.factory))
    with ExitStack() as cleanup:
        owners = [constructed("toy", one, cleanup) for one in simulations]
        for owner, problem in zip(owners, problems, strict=True):
            owner.set_initial(np.asarray(Start.entrance(problem).take_field()), None)
        with pytest.raises(BackendError, match="Batch has no runner for backend 'toy'"):
            select_batch(owners)


@pytest.mark.usefixtures("toy")
def test_a_backend_that_runs_no_group_refuses_it_at_the_door(monkeypatch):
    def refused(self, members, problems, requests):
        raise BackendError("the toy runs no group")

    monkeypatch.setattr(Toy, "prepare_group", refused)
    simulations = [member(index, "numpy_ref") for index in range(2)]
    problems = [one.translate() for one in simulations]
    with ExitStack() as cleanup:
        owners = [constructed("toy", one, cleanup) for one in simulations]
        with pytest.raises(BackendError, match="the toy runs no group"):
            start_group(
                owners,
                problems,
                [one._run_request() for one in simulations],
                [Start.entrance(problem) for problem in problems],
            )
    assert ToyGroup.started == []


@pytest.mark.usefixtures("toy")
def test_a_row_without_a_device_is_refused_a_required_device_by_name(monkeypatch):
    monkeypatch.setenv(registry.REQUIRE_DEVICE_SETTING, "any")
    with pytest.raises(BackendError, match="this run is on toy"):
        registry._check_required_device("toy")
    assert not registry._device_build("toy")
    assert registry.statistics_refusal("toy", ["beam_spectrum"]) is None
