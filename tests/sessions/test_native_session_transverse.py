# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native session's own transverse edge against the driver's, end to end.

SEP 47 section 6.4: the native session owns a free-space run's transverse edge (SEP 40
section 5.3) as it owns the temporal monitor, and Python only formats what it reports.
Here an edged run goes through the native session and through the transitional
:class:`~solprop.backends.driver_session.DriverSession` over the native backend, whose
solprop edge judges the same native readings of the same strict marches: the warnings
must be the same, at the same positions, the stops the same, the attempts and planes
the same, and the records the same. Both sides make the same scalar decisions on the
same readings, so they agree exactly, but for the three reductions that are each
backend's reading (``transverse.hpp``): the safety ramp's removal and the launch's
share in its band, which the native edge sums in order and solprop pairwise. Those
fields of the record are held to the bound of
``tests/backends/test_transverse_owner.py``; the recorded-readings harness there holds
the two edges to each other apart from any run.

A warning is a service barrier (:class:`~solprop.backends.events.ServiceRequired`),
each of a point's warnings its own; a collar's current is taken at every committed
step with no return to Python.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from backends.test_transverse_backends import radial_collar
from sessions.executors import trace
from sessions.test_native_session_monitor import IDENTITY, Served, _identity, _serve
from solprop.backends.driver_session import DriverSession
from solprop.backends.events import ObservationReady, Outcome, ServiceRequired
from solprop.backends.registry import select
from solprop.backends.session import Start
from solprop.driver import Simulation
from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.monitor import BoundaryCheck
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, Multistep, RungeKutta
from solprop.translate.run_request import describe_run
from test_transverse_reads import REREVIEW, packet_run

native_session = pytest.importorskip("solprop_native.session")
_counters = pytest.importorskip("solprop_native._counters")

U = 2.0**-53
"""The unit roundoff of binary64."""

READINGS = ("ramp_removed", "raw_ramp_removed", "declared_share_in_ramp")
"""The edge record's fields formed from a reduction that is each backend's reading."""

MARCHES = {"rk": RungeKutta, "adams": Multistep}


def _prepared(simulation: Simulation) -> tuple[Any, Any]:
    """Return a simulation's problem, without statistics, and its monitored request."""
    problem = dataclasses.replace(simulation.translate(), stats_spec=StatsSpec())
    return problem, describe_run(problem, monitor=simulation._boundary_monitor())


def _start(
    simulation: Simulation,
    kind: str,
    start: Start | None = None,
    prepared: tuple[Any, Any] | None = None,
) -> Any:
    """Start a session of `kind` on a simulation's fresh edged run."""
    problem, request = _prepared(simulation) if prepared is None else prepared
    assert request.boundary.monitor is not None
    assert request.boundary.monitor.transverse is not None
    if start is None:
        start = Start(field=problem.initial.field0[0], z=float(problem.initial.z))
    context = {
        "grid": simulation.grid,
        "modes": simulation.modes,
        "profiles": simulation.profiles,
    }
    if kind == "driver":
        return DriverSession(
            select("native", problem, **context), problem, request, start
        )
    return native_session.NativeSession(problem, request, start, **context)


def _both(simulation_of: Any, **serve: Any) -> tuple[Served, Served]:
    served = []
    for kind in ("driver", "native"):
        session = _start(simulation_of(), kind)
        try:
            served.append(_serve(session, **serve))
        finally:
            session.close()
    return served[0], served[1]


def _edge_bound(record: dict[str, Any], columns: int) -> float:
    """Return the bound on a reading field's difference: the ramp's sums (above)."""
    reads = int(record["reads"])
    scale = abs(float(record["raw_ramp_removed"])) + abs(
        float(record["declared_share_in_ramp"])
    )
    order = columns - 1 + math.ceil(math.log2(max(columns, 2)))
    return (order + 2 * reads + 12) * U * scale


def _compare(driver: Served, native: Served, columns: int) -> None:
    """Require the same trace, warnings, outcome and records."""
    assert native.warnings == driver.warnings
    assert trace(native.run) == trace(driver.run)
    assert native.record is not None and driver.record is not None
    expected, got = dict(driver.record), dict(native.record)
    assert list(got) == list(expected)
    edge, edge_expected = got.pop("transverse"), expected.pop("transverse")
    assert got == expected
    assert list(edge) == list(edge_expected)
    bound = _edge_bound(edge_expected, columns)
    for name, value in edge_expected.items():
        if name in READINGS:
            assert edge[name] == pytest.approx(value, rel=0.0, abs=bound), name
        else:
            assert edge[name] == value, name


def _columns(simulation: Simulation) -> int:
    """Return the run's transverse columns, the ramp's sums' length."""
    monitor = simulation._boundary_monitor()
    assert monitor is not None and monitor.transverse is not None
    return int(np.asarray(monitor.transverse.space.spectral_weights).size)


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_the_packet_stops_where_the_driver_stops_it(march):
    def simulation() -> Simulation:
        return packet_run(REREVIEW, integrator=MARCHES[march](), backend="native")

    driver, native = _both(simulation)
    _compare(driver, native, _columns(simulation()))
    terminal = native.run.terminal
    assert terminal.outcome is Outcome.FAILED
    assert isinstance(terminal.error, BoundaryError)
    assert "reached the transverse edge" in str(terminal.error)
    assert str(terminal.error) == str(driver.run.terminal.error)
    assert terminal.z == driver.run.terminal.z
    assert native.record is not None
    assert native.record["transverse"]["transverse_status"] == "stopped"


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_a_relaxed_packet_is_read_on_the_driver_s_spacing(march):
    def simulation() -> Simulation:
        return packet_run(
            REREVIEW,
            integrator=MARCHES[march](),
            check=BoundaryCheck(transverse_budget=10.0),
            planes=[REREVIEW.length / 2, REREVIEW.length],
            backend="native",
        )

    driver, native = _both(simulation)
    _compare(driver, native, _columns(simulation()))
    assert native.run.terminal.outcome is Outcome.COMPLETED
    assert native.record is not None
    edge = native.record["transverse"]
    assert edge["reads"] >= 16
    # The backstop grows to L / 4 while the reach holds (BACKSTOP_READS).
    assert edge["largest_gap"] <= REREVIEW.length / 4 * (1 + 1e-9)


def test_a_replay_is_coarse_and_read_at_its_own_steps():
    # A fixed schedule's long steps: the launch warns that the run is coarse, a
    # barrier before the first step, and the reads fall where the steps land.
    length = REREVIEW.length
    edges = np.r_[0.0, 1e-4 * (2.0 ** np.arange(1, 10) - 1), 0.70475, 1.0]

    def simulation() -> Simulation:
        return packet_run(
            REREVIEW,
            integrator=Multistep(),
            schedule=FixedSchedule(np.diff(edges) * length),
            check=BoundaryCheck(transverse_budget=10.0),
            backend="native",
        )

    driver, native = _both(simulation)
    _compare(driver, native, _columns(simulation()))
    assert native.record is not None
    assert native.record["transverse"]["transverse_status"] == "coarse"
    first = native.warning_events[0]
    assert "longest step" in first.message
    assert first.z == 0.0


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_a_requested_collar_s_current_is_taken_natively_at_every_step(march):
    driver, native = _both(lambda: radial_collar("native", MARCHES[march]()))
    _compare(driver, native, 48)
    assert native.record is not None
    edge = native.record["transverse"]
    assert edge["spatial_removed"] > 0.0
    assert edge["current"] != 0.0 or edge["current_baseline"] >= 0.0


def test_an_edged_native_session_returns_to_python_only_at_its_events():
    # SEP 47 section 7.3: the edge's reads and a collar's current at every step are
    # the session's own; unbounded by attempts or time, the run returns at its planes
    # and its end alone, with no reading taken through Python.
    simulation = radial_collar("native", RungeKutta())
    session = _start(simulation, "native")
    counters = _counters.enable()
    batches = []
    try:
        while not session.finished:
            batch = session.advance(work_limit=2**40, deadline=1e9)
            batches.append(batch)
            for event in batch.events:
                assert not isinstance(event, ServiceRequired), event
            session.acknowledge(batch.receipt)
        status = session.native.status()
        record = session.monitor_record()
    finally:
        _counters.disable()
        session.close()
    observations = [
        event
        for batch in batches
        for event in batch.events
        if isinstance(event, ObservationReady)
    ]
    assert len(batches) == len(observations) + 1 == status.batches
    accepted = sum(int(event.history.accepted.sum()) for event in observations)
    assert accepted > len(observations)
    found = counters.record()
    assert found["transverse_reads"] == 0
    assert found["native_returns"] == {}
    assert found["field_crossings"] == {}
    assert record is not None
    assert record["transverse"]["reads"] > 0


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_a_raised_warning_ends_the_run_at_its_endpoint(march):
    def simulation() -> Simulation:
        return packet_run(REREVIEW, integrator=MARCHES[march](), backend="native")

    driver, native = _both(simulation, error=True)
    terminal = driver.run.terminal
    assert terminal.outcome is Outcome.FAILED
    assert isinstance(terminal.error, BoundaryWarning)
    assert native.raised is not None
    z, warning = native.raised
    assert terminal.error is not None
    assert (z, _identity(warning)) == (terminal.z, _identity(terminal.error))
    assert trace(native.run) == trace(driver.run)[:-1]


def test_a_resumed_edge_continues_the_driver_s_record():
    # The first execution runs to its first plane; both sessions then resume there
    # from its record, the native one reconstructing the edge's natively.
    def simulation() -> Simulation:
        return packet_run(
            REREVIEW,
            integrator=RungeKutta(),
            check=BoundaryCheck(transverse_budget=10.0),
            planes=[REREVIEW.length / 2, REREVIEW.length],
            backend="native",
        )

    first = _start(simulation(), "driver")
    state = None
    record: dict[str, Any] = {}
    try:
        while not first.finished:
            batch = first.advance()
            planes = [
                event
                for event in batch.events
                if isinstance(event, ObservationReady) and event.plane_index == 0
            ]
            if planes:
                state = (
                    planes[0].z,
                    np.array(first.export(planes[0].ticket, IDENTITY)),
                )
                record = dict(first.monitor_record() or {})
                break
            first.acknowledge(batch.receipt)
    finally:
        first.close()
    assert state is not None
    assert record["transverse"]["reads"] > 0
    z, field = state
    served = []
    for kind in ("driver", "native"):
        start = Start(
            field=field,
            z=z,
            plane_index=1,
            plane_pending=False,
            monitor={"record": record, "plane": True, "unwatched": []},
        )
        session = _start(simulation(), kind, start)
        try:
            served.append(_serve(session))
        finally:
            session.close()
    driver, native = served
    _compare(driver, native, _columns(simulation()))
    assert native.record is not None
    edge = native.record["transverse"]
    assert edge["reads"] > record["transverse"]["reads"]
    assert edge["launch_energy"] == record["transverse"]["launch_energy"]
