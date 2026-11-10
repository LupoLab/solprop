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
share in its band, which the native edge sums in order and solprop pairwise, and with
the input noise the ramp's removal of the noise, summed likewise. Those
fields of the record are held to the bound of
``tests/backends/test_transverse_owner.py``; the recorded-readings harness there holds
the two edges to each other apart from any run.

A warning is a service barrier (:class:`~solprop.backends.events.ServiceRequired`),
each of a point's warnings its own; a collar's current is taken at every committed
step with no return to Python.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from backends.test_transverse_backends import radial_collar
from sessions.executors import device_build, trace
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
from test_transverse_reads import REREVIEW, packet_run

native_session = pytest.importorskip("solprop_native.session")

_counters = pytest.importorskip("solprop_native._counters")

U = 2.0**-53
"""The unit roundoff of binary64."""

READINGS = (
    "ramp_removed",
    "raw_ramp_removed",
    "noise_ramp_removed",
    "declared_share_in_ramp",
)
"""The edge record's fields formed from a reduction that is each backend's reading."""

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
    if kind == "driver":
        return DriverSession(
            select("native", problem, request, **context), problem, request, start
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
    scale = (
        abs(float(record["raw_ramp_removed"]))
        + abs(float(record["noise_ramp_removed"]))
        + abs(float(record["declared_share_in_ramp"]))
    )
    order = columns - 1 + math.ceil(math.log2(max(columns, 2)))
    return (order + 2 * reads + 12) * U * scale


SPLIT_FLOOR = ("removed_fraction", "removed_signed", "floor_removed_fraction")
"""The temporal record's fields that hold the input floor's share of a split step's
removal, ``sum_t (1 - exp(-alpha(t) h))``, which the native monitor sums in order and
solprop pairwise (``watch.hpp``)."""


def _floor_delta(n_t: int, steps: int) -> float:
    """Return the relative bound on a split floor share after `steps` accepted steps.

    ``tests/backends/test_monitor_owner.py``'s: each step's sum over the ``n_t``
    samples, in order against pairwise, and its accumulation over the steps.
    """
    return (n_t + math.ceil(math.log2(n_t)) + 2 * steps + 4) * U


def _samples(simulation: Simulation) -> int:
    """Return the monitor's time samples, the split floor's sums' length."""
    monitor = simulation._boundary_monitor()
    assert monitor is not None
    return int(monitor.weights.shape[1])


def _steps_at(run: Any) -> dict[float, int]:
    """Return the accepted steps a run took up to each of its observations."""
    steps = 0
    found: dict[float, int] = {}
    for event, history, _ in run.assembled():
        steps += int(np.count_nonzero(history.accepted))
        if isinstance(event, ObservationReady):
            found[float(event.z)] = steps
    return found


def _compare(
    driver: Served, native: Served, columns: int, *, samples: int | None = None
) -> None:
    """Require the same trace, warnings, outcome and records.

    With the monitor's time `samples`, a noisy split run's floor fields
    (:data:`SPLIT_FLOOR`) are held to their bound at the run's steps and scale, and
    each plane's removal to its bound at that plane's steps on the scale of the
    floor share its removal subtracts, at most ``min(floor_removed, noise_floor)``;
    the rest of the temporal record exactly.
    """
    assert native.warnings == driver.warnings
    assert trace(native.run) == trace(driver.run)
    assert native.record is not None and driver.record is not None
    expected, got = dict(driver.record), dict(native.record)
    assert list(got) == list(expected)
    edge, edge_expected = got.pop("transverse"), expected.pop("transverse")
    if samples is not None:
        floor = abs(float(expected["floor_removed_fraction"]))
        bound = _floor_delta(samples, int(expected["steps"])) * floor
        for name in SPLIT_FLOOR:
            assert got.pop(name) == pytest.approx(
                expected.pop(name), rel=0.0, abs=bound
            ), name
        # The removal subtracts at most the floor itself from the raw removal, which
        # both sides record exactly.
        share = min(floor, abs(float(expected["noise_floor"])))
        steps = _steps_at(driver.run)
        # A continued record counts the earlier execution's steps, and holds its
        # planes, before any of this run's.
        earlier = int(expected["steps"]) - max(steps.values(), default=0)
        planes, planes_expected = got.pop("planes"), expected.pop("planes")
        assert list(planes) == list(planes_expected)
        for name, values in planes_expected.items():
            if name != "removed_fraction":
                assert planes[name] == values, name
                continue
            for z, value, found in zip(
                planes_expected["z"], values, planes[name], strict=True
            ):
                taken = earlier + steps.get(float(z), 0)
                at = _floor_delta(samples, taken) * share
                assert found == pytest.approx(value, rel=0.0, abs=at), z
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


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_the_input_noise_beside_a_collar_is_judged_as_the_driver_judges_it(march):
    """The input draw's readings -- its allowance, its columns in the reach and the
    ramp, a collar's strip floor -- are formed on the host once, as solprop's edge
    forms them at its launch, and carried by the native launch: the run is the
    driver's, its record too but for the ramp's sums (above). On a device build the
    split floor's share (:data:`SPLIT_FLOOR`) is held to its bound too: the native
    monitor sums it in order and solprop pairwise, which a device's readings round
    apart in the last bit where a CPU build's happen to agree.
    """

    def simulation() -> Simulation:
        return radial_collar("native", MARCHES[march](), noise="input")

    problem, request = _prepared(simulation())
    assert problem.initial.noise_model == "input"
    assert native_session.refusal(problem, request) is None
    driver, native = _both(simulation)
    samples = _samples(simulation()) if device_build() else None
    _compare(driver, native, 48, samples=samples)
    assert native.record is not None
    edge = native.record["transverse"]
    assert edge["noise_allowance"] > 0.0
    assert edge["strip_floor"] > 0.0
    assert edge["noise_share_in_ramp"] > 0.0


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_a_noisy_packet_is_read_as_the_driver_reads_it(march):
    """A free-space packet launched with the input noise, relaxed and read to the end:
    the run is the driver's, its edge record too but for the ramp's sums, and its
    temporal record but for the split floor's share (:data:`SPLIT_FLOOR`), each
    within its bound.
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

    driver, native = _both(simulation)
    _compare(driver, native, _columns(simulation()), samples=_samples(simulation()))
    assert native.run.terminal.outcome is driver.run.terminal.outcome
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

    driver, native = _both(simulation, error=True)
    terminal = driver.run.terminal
    assert terminal.outcome is Outcome.FAILED
    assert isinstance(terminal.error, BoundaryWarning)
    assert native.raised is not None
    z, warning = native.raised
    assert terminal.error is not None
    assert (z, _identity(warning)) == (terminal.z, _identity(terminal.error))
    assert trace(native.run) == trace(driver.run)[:-1]


@pytest.mark.parametrize(
    ("noise", "continued"),
    [(None, True), ("input", True), ("input", False)],
    ids=["quiet", "noisy", "noisy-restart"],
)
def test_a_resumed_edge_continues_the_driver_s_record(noise, continued):
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
    crossings = []
    for kind in ("driver", "native"):
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
    driver, native = served
    # A restart's launch is judged on the loaded state less the draw, which crosses
    # once on either side; a continued record's launch is not judged, and nothing
    # crosses for it.
    assert crossings[1] == crossings[0]
    if continued:
        assert crossings[1] == {}
    samples = None if noise is None else _samples(simulation())
    _compare(driver, native, _columns(simulation()), samples=samples)
    assert native.record is not None
    edge = native.record["transverse"]
    if continued:
        assert edge["reads"] > record["transverse"]["reads"]
        assert edge["launch_energy"] == record["transverse"]["launch_energy"]
    else:
        assert edge["launch_energy"] != record["transverse"]["launch_energy"]
    assert (edge["noise_allowance"] > 0.0) is (noise is not None)
