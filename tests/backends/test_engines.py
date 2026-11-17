# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The engine protocols, frozen (SEP 52 section 5.2).

``solprop.backends.engines`` mirrors the native marches' engines one-to-one:
``RungeKuttaEngine`` is ``rk::Engine`` and ``AdamsEngine`` is ``march::Engine``. A
change to a member or a signature here changes what every engine implements, and is made
with the C++ interface in the same commit.
"""

import dataclasses
import inspect
import math

import pytest

from solprop.backends import engines
from solprop.backends.engines import (
    AcceptRecord,
    AdamsEngine,
    AttemptOutcome,
    BoundaryReading,
    HistoryEntry,
    RungeKuttaEngine,
)

RUNGE_KUTTA_ENGINE = {
    "trial": "(self, z: float, dz: float) -> float",
    "accept": "(self, dz: float) -> solprop.backends.engines.AcceptRecord",
    "extend": "(self, z: float, dz: float) -> None",
    "reading": "(self) -> solprop.backends.engines.BoundaryReading",
    "in_flight": "(self) -> bool",
}
"""``rk::Engine`` (``native/include/solprop_native/march/rk_march.hpp``), member by
member."""

ADAMS_ENGINE = {
    "start": "(self, z: float) -> None",
    "trial": (
        "(self, z: float, dz: float, stencil: solprop.backends.adams_core.Stencil)"
        " -> jaxtyping.Float[ndarray, 'n_orders']"
    ),
    "accept": (
        "(self, dz: float, free: int, stencil: solprop.backends.adams_core.Stencil,"
        " kept: collections.abc.Sequence[solprop.backends.engines.HistoryEntry])"
        " -> solprop.backends.engines.AcceptRecord"
    ),
    "chord_phase": "(self, z1: float, z2: float) -> float",
    "step_carrier": "(self, order: int) -> jaxtyping.Float[ndarray, 'n_rotations']",
    "begin_attempt": "(self, cost: int) -> None",
    "end_attempt": "(self, outcome: solprop.backends.engines.AttemptOutcome) -> None",
    "reading": "(self) -> solprop.backends.engines.BoundaryReading",
}
"""``march::Engine`` (``native/include/solprop_native/march/adams_march.hpp``)."""

RECORDS = {
    AcceptRecord: ("stiffness", "removed", "found"),
    BoundaryReading: ("total", "regions", "occupancy"),
    HistoryEntry: ("z", "identity", "row"),
}
"""The records the protocols pass, field by field."""

OUTCOMES = {"REJECTED": 0, "ACCEPTED": 1, "ABANDONED": 2}
"""``HostOutcome`` (``native/include/solprop_native/runtime/host_source.hpp``), by
value."""


@pytest.mark.parametrize(
    ("protocol", "surface"),
    [(RungeKuttaEngine, RUNGE_KUTTA_ENGINE), (AdamsEngine, ADAMS_ENGINE)],
    ids=["runge_kutta", "adams"],
)
def test_the_engine_protocols_are_frozen(protocol, surface):
    members = {name for name in vars(protocol) if not name.startswith("_")}
    assert members == set(surface)
    for name, signature in surface.items():
        member = vars(protocol)[name]
        if isinstance(member, property):
            assert member.fget is not None
            member = member.fget
        assert str(inspect.signature(member)) == signature, name


def test_only_in_flight_is_a_property():
    # The low-storage step's interval (SEP 52 section 5.2) is read, never called.
    assert isinstance(vars(RungeKuttaEngine)["in_flight"], property)
    assert not any(
        isinstance(member, property) for member in vars(AdamsEngine).values()
    )


@pytest.mark.parametrize(("record", "fields"), RECORDS.items())
def test_the_records_the_engines_pass_are_frozen(record, fields):
    assert tuple(field.name for field in dataclasses.fields(record)) == fields
    assert record.__dataclass_params__.frozen  # pyright: ignore[reportAttributeAccessIssue]


def test_the_attempt_outcomes_are_the_native_values():
    assert {outcome.name: outcome.value for outcome in AttemptOutcome} == OUTCOMES


def test_an_accept_record_knows_nothing_by_default():
    # The stiffness diagnostic is NaN where the engine does not form it, and a step
    # with no split collar reports no removal and no found occupancies.
    record = AcceptRecord()
    assert math.isnan(record.stiffness)
    assert (record.removed, record.found) == (0.0, None)


def test_the_module_declares_the_protocols_and_their_records():
    assert set(engines.__all__) == {
        "AcceptRecord",
        "AdamsEngine",
        "AttemptOutcome",
        "BoundaryReading",
        "HistoryEntry",
        "RungeKuttaEngine",
    }
