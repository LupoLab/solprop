# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native session's own transverse edge against the reference's, end to end.

SEP 47 section 6.4: the native session owns a free-space run's transverse edge (SEP 40
section 5.3) as it owns the temporal monitor, and Python only formats what it reports.
Here an edged run goes through the native session and through the NumPy reference's
session, whose solprop edge judges the same problem. The two backends round
differently and so take different steps (SEP 25), and the edge reads on its own
spacing from their states: the two are held to the same decisions -- the same
warnings in order and in kind, the same outcome and failure, the same planes, their
statistics and positions to :data:`PLANE_RTOL`, and the same record layout -- and the
native session, on one thread, to its own run bit for bit. Both sides make the same
scalar decisions on the same readings, which the recorded-readings harness
(``tests/backends/test_transverse_owner.py``) holds apart from any run, with the three
reductions that are each backend's reading (``transverse.hpp``) to their bound there.

A warning is a service barrier (:class:`~solprop.backends.events.ServiceRequired`),
each of a point's warnings its own; a collar's current is taken at every committed
step with no return to Python.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from backends.test_transverse_backends import radial_collar
from sessions.executors import (
    assert_close_trace,
    device_build,
    masked,
    one_thread,
    record_layout,
    trace,
)
from sessions.test_native_session_monitor import IDENTITY, Served, _serve
from solprop.backends.events import ObservationReady, Outcome, ServiceRequired
from solprop.backends.host.session import HostSession
from solprop.backends.registry import select
from solprop.backends.session import Start
from solprop.driver import Simulation
from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.monitor import BoundaryCheck
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, Multistep, RungeKutta
from test_transverse_reads import REREVIEW, packet_run

native_session = pytest.importorskip("solprop_native.session")

_counters = pytest.importorskip("solprop_native._counters")

MARCHES = {"rk": RungeKutta, "adams": Multistep}


def _prepared(simulation: Simulation) -> tuple[Any, Any]:
    """Return a simulation's problem, without statistics, and its monitored request."""
    return simulation.translate(), simulation._run_request().for_run(
        stats=StatsSpec(), monitor=simulation._boundary_monitor()
    )


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
        start = Start(field=problem.initial.field0, z=float(problem.initial.z))
    context = {
        "grid": simulation.grid,
        "modes": simulation.modes,
        "profiles": simulation.profiles,
    }
    if kind == "reference":
        return HostSession(
            select("numpy_ref", problem, request, **context), problem, request, start
        )
    return native_session.NativeSession(problem, request, start, **context)


def _both(simulation_of: Any, **serve: Any) -> tuple[Served, Served]:
    """Serve the reference's session and the native session twice; return one of each.

    On one thread the native session's second run repeats its first bit for bit.
    """
    served = []
    for kind in ("reference", "native", "native"):
        session = _start(simulation_of(), kind)
        try:
            served.append(_serve(session, **serve))
        finally:
            session.close()
    if one_thread():
        first, second = served[1], served[2]
        assert first.warnings == second.warnings
        assert trace(first.run) == trace(second.run)
        assert first.record == second.record
    return served[0], served[1]


PLANE_RTOL = 1e-6
"""How closely the native planes, their statistics and a stopping endpoint follow the
reference's, each backend taking its own steps under the controller's tolerance."""


def _samples(simulation: Simulation) -> int:
    """Return the monitor's time samples, the split floor's sums' length."""
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    return int(monitor.weights.shape[1])


def _compare(
    reference: Served, native: Served, columns: int, *, samples: int | None = None
) -> None:
    """Require the reference's decisions, outcome, planes and record layout.

    `columns` and `samples` are the edge's and the monitor's sum lengths, which bound
    the readings each backend reduces in its own order when both take the same steps
    (``tests/backends/test_transverse_owner.py``); against the reference, which takes
    its own, the readings are not compared.
    """
    del columns, samples
    assert [(one[0], masked(str(one[1]))) for one in native.warnings] == [
        (one[0], masked(str(one[1]))) for one in reference.warnings
    ]
    assert_close_trace(native.run, reference.run, rtol=PLANE_RTOL)
    assert native.record is not None and reference.record is not None
    assert record_layout(native.record) == record_layout(reference.record)
    edge, expected = native.record["transverse"], reference.record["transverse"]
    assert (edge["reads"] > 0) is (expected["reads"] > 0)
    for name in ("noise_allowance", "launch_energy"):
        assert edge[name] == pytest.approx(expected[name], rel=PLANE_RTOL), name


def _columns(simulation: Simulation) -> int:
    """Return the run's transverse columns, the ramp's sums' length."""
    monitor = simulation._boundary_monitor()
    assert monitor is not None and monitor.transverse is not None
    return int(np.asarray(monitor.transverse.space.spectral_weights).size)


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_the_packet_stops_where_the_reference_stops_it(march):
    def simulation() -> Simulation:
        return packet_run(REREVIEW, integrator=MARCHES[march](), backend="native")

    reference, native = _both(simulation)
    _compare(reference, native, _columns(simulation()))
    terminal = native.run.terminal
    assert terminal.outcome is Outcome.FAILED
    assert isinstance(terminal.error, BoundaryError)
    assert "reached the transverse edge" in str(terminal.error)
    assert masked(str(terminal.error)) == masked(str(reference.run.terminal.error))
    assert native.record is not None
    assert native.record["transverse"]["transverse_status"] == "stopped"


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_a_relaxed_packet_is_read_on_the_reference_s_spacing(march):
    def simulation() -> Simulation:
        return packet_run(
            REREVIEW,
            integrator=MARCHES[march](),
            check=BoundaryCheck(transverse_budget=10.0),
            planes=[REREVIEW.length / 2, REREVIEW.length],
            backend="native",
        )

    reference, native = _both(simulation)
    _compare(reference, native, _columns(simulation()))
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

    reference, native = _both(simulation)
    _compare(reference, native, _columns(simulation()))
    assert native.record is not None
    assert native.record["transverse"]["transverse_status"] == "coarse"
    first = native.warning_events[0]
    assert "longest step" in first.message
    assert first.z == 0.0


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_a_requested_collar_s_current_is_taken_natively_at_every_step(march):
    reference, native = _both(lambda: radial_collar("native", MARCHES[march]()))
    _compare(reference, native, 48)
    assert native.record is not None
    edge = native.record["transverse"]
    assert edge["spatial_removed"] > 0.0
    assert edge["current"] != 0.0 or edge["current_baseline"] >= 0.0


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_the_input_noise_beside_a_collar_is_judged_as_the_reference_judges_it(march):
    """The input draw's readings -- its allowance, its columns in the reach and the
    ramp, a collar's strip floor -- are formed on the host once, as solprop's edge
    forms them at its launch, and carried by the native launch: the run makes the
    reference's decisions, with the same allowance (:func:`_compare`).
    """

    def simulation() -> Simulation:
        return radial_collar("native", MARCHES[march](), noise="input")

    problem, request = _prepared(simulation())
    assert problem.initial.noise_model == "input"
    assert native_session.refusal(problem, request) is None
    reference, native = _both(simulation)
    samples = _samples(simulation()) if device_build() else None
    _compare(reference, native, 48, samples=samples)
    assert native.record is not None
    edge = native.record["transverse"]
    assert edge["noise_allowance"] > 0.0
    assert edge["strip_floor"] > 0.0
    assert edge["noise_share_in_ramp"] > 0.0


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_a_noisy_packet_is_read_as_the_reference_reads_it(march):
    """A free-space packet launched with the input noise, relaxed and read to the end:
    the run makes the reference's decisions, with the same allowance.
    """

    def simulation() -> Simulation:
        return packet_run(
            REREVIEW,
            integrator=MARCHES[march](),
            check=BoundaryCheck(transverse_budget=10.0),
            planes=[REREVIEW.length / 2, REREVIEW.length],
            backend="native",
            noise="input",
        )

    reference, native = _both(simulation)
    _compare(reference, native, _columns(simulation()), samples=_samples(simulation()))
    assert native.run.terminal.outcome is reference.run.terminal.outcome
    assert native.record is not None
    assert native.record["transverse"]["noise_allowance"] > 0.0
    assert native.record["floor_removed_fraction"] > 0.0


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
    assert found["native_returns"] == {
        "session_observation": len(observations),
        "session_terminal": 1,
    }
    assert found["field_crossings"] == {}
    assert record is not None
    assert record["transverse"]["reads"] > 0


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_a_raised_warning_ends_the_run_at_its_endpoint(march):
    def simulation() -> Simulation:
        return packet_run(REREVIEW, integrator=MARCHES[march](), backend="native")

    reference, native = _both(simulation, error=True)
    terminal = reference.run.terminal
    assert terminal.outcome is Outcome.FAILED
    assert isinstance(terminal.error, BoundaryWarning)
    assert native.raised is not None
    z, warning = native.raised
    assert terminal.error is not None
    assert type(warning) is type(terminal.error)
    assert masked(str(warning)) == masked(str(terminal.error))
    assert z == pytest.approx(terminal.z, rel=PLANE_RTOL)
    assert [(one.kind, one.plane_index) for one in native.run.observations] == [
        (one.kind, one.plane_index) for one in reference.run.observations
    ]


@pytest.mark.parametrize(
    ("noise", "continued"),
    [(None, True), ("input", True), ("input", False)],
    ids=["quiet", "noisy", "noisy-restart"],
)
def test_a_resumed_edge_continues_the_reference_s_record(noise, continued):
    # The first execution runs to its first plane; both sessions then resume there
    # from its record, the native one reconstructing the edge's natively. A noisy
    # run's resumed launch carries the draw's readings, which its reads judge above;
    # restarted there without the record, its launch is judged afresh, on the loaded
    # state's columns less the draw it carries.
    def simulation() -> Simulation:
        return packet_run(
            REREVIEW,
            integrator=RungeKutta(),
            check=BoundaryCheck(transverse_budget=10.0),
            planes=[REREVIEW.length / 2, REREVIEW.length],
            backend="native",
            noise=noise,
        )

    first = _start(simulation(), "reference")
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
    crossings = []
    for kind in ("reference", "native"):
        start = Start(
            field=field,
            z=z,
            plane_index=1,
            plane_pending=False,
            monitor={"record": record, "plane": True, "unwatched": []}
            if continued
            else None,
        )
        counters = _counters.enable()
        try:
            session = _start(simulation(), kind, start)
            try:
                served.append(_serve(session))
            finally:
                session.close()
            crossings.append(counters.record()["field_crossings"])
        finally:
            _counters.disable()
    reference, native = served
    # A restart's launch is judged on the loaded state less the draw, which crosses
    # once; a continued record's launch is not judged, and nothing crosses for it.
    if continued:
        assert crossings[1] == {}
    else:
        assert sum(crossings[1].values()) == 1, crossings[1]
    samples = None if noise is None else _samples(simulation())
    _compare(reference, native, _columns(simulation()), samples=samples)
    assert native.record is not None
    edge = native.record["transverse"]
    if continued:
        assert edge["reads"] > record["transverse"]["reads"]
        assert edge["launch_energy"] == record["transverse"]["launch_energy"]
    else:
        assert edge["launch_energy"] != record["transverse"]["launch_energy"]
    assert (edge["noise_allowance"] > 0.0) is (noise is not None)
