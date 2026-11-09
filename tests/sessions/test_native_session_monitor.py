# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native session's own boundary monitor against the driver's, end to end.

SEP 47 section 6.4: the native session owns the temporal boundary monitor, and Python
only formats what it reports. Here a monitored run goes through the native session and
through the transitional :class:`~solprop.backends.driver_session.DriverSession` over
the native backend, whose solprop monitor watches the same strict marches over the same
kernels: the warnings must be the same, at the same positions, the stops the same, the
attempts and planes the same, and the records the same. Both sides take the same
readings of the same states and make the same scalar decisions on them, so on one
build they agree exactly (section 4.3); the recorded-readings harness
(``tests/backends/test_monitor_owner.py``) holds the two monitors to each other apart
from any run.

A warning the native session reports is a service barrier
(:class:`~solprop.backends.events.ServiceRequired`): the caller emits it, and its
filters may raise it, which ends the run at that committed endpoint (section 5.5);
the driver emits it inline, its filters raising it into a failure there.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import pytest

from sessions.executors import Run, trace
from solprop.backends.driver_session import DriverSession
from solprop.backends.events import (
    EventBatch,
    ObservationReady,
    Outcome,
    ServiceKind,
    ServiceRequired,
    Terminal,
    emit_warning,
)
from solprop.backends.export import ExportRecipe
from solprop.backends.registry import select
from solprop.backends.session import Start
from solprop.constants import C
from solprop.driver import Simulation
from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.grid import EnvelopeGrid
from solprop.monitor import BoundaryCheck
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec
from solprop.stepper.multistep import Multistep
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.run_request import describe_run

native_session = pytest.importorskip("solprop_native.session")

BETA2 = 2e-26
"""s^2/m, as ``tests/test_monitor.py``'s walking pulse."""

TRANGE = 1e-12
"""Its window, s."""

IDENTITY = ExportRecipe(power=False, optics=False)


def walker(  # noqa: PLR0913 - one choice per keyword
    drift: float,
    *,
    check: BoundaryCheck | None = None,
    integrator: Multistep | RungeKutta | None = None,
    scheme: str | None = None,
    n_save: int = 3,
    length: float = 1.0,
    trange: float = TRANGE,
) -> Simulation:
    """Return ``tests/test_monitor.py``'s linear pulse, walking `drift` seconds."""
    grid = EnvelopeGrid.from_limits(length, 1e-6, (0.8e-6, 1.25e-6), trange)
    omega0 = 2 * np.pi * C / 1e-6
    offset = drift / (BETA2 * length)
    extra: dict[str, Any] = {}
    if integrator is not None:
        extra["integrator"] = integrator
    if scheme is not None:
        extra["absorber_scheme"] = scheme
    simulation = Simulation.from_gnlse(
        1e-30, length, [0.0, 0.0, BETA2], wavelength=1e-6, grid=grid,
        pulses=[GaussianPulse(
            wavelength=2 * np.pi * C / (omega0 + offset), fwhm=200e-15, peak_power=1.0,
        )],
        noise=None, raman=False, shock=False, n_save=n_save, backend="native",
        **extra,
    )  # fmt: skip
    simulation.settings = dataclasses.replace(
        simulation.settings,
        boundary_check=BoundaryCheck() if check is None else check,
    )
    return simulation


def spreading() -> Simulation:
    """Return ``tests/test_split_crossing.py``'s pulse that outgrows its schedule."""
    grid = EnvelopeGrid.from_limits(2.0, 1e-6, (0.8e-6, 1.25e-6), 2e-12)
    return Simulation.from_gnlse(
        0.03, 2.0, [0.0, 0.0, BETA2], wavelength=1e-6, grid=grid,
        pulses=[GaussianPulse(wavelength=1e-6, fwhm=500e-15, peak_power=1e3)],
        noise=None, raman=False, shock=False, backend="native",
        integrator=RungeKutta(), absorber_scheme="split_step",
        schedule=FixedSchedule.uniform(0.0, 2.0, 0.05, save_planes=[2.0]),
        save_planes=[2.0],
    )  # fmt: skip


@dataclasses.dataclass
class Served:
    """What serving a session to its end gave."""

    run: Run
    warnings: list[tuple[object, ...]]
    raised: tuple[float, BaseException] | None
    record: dict[str, Any] | None
    warning_events: list[ServiceRequired]


def _prepared(simulation: Simulation, period: int = 2) -> tuple[Any, Any]:
    """Return a simulation's problem and its monitored run request.

    Statistics the native session takes natively, with periodic rows every `period`
    accepted steps after each step's check (SEP 47 section 5.3), as the problem's own:
    the driver takes its rows from the problem.
    """
    problem = dataclasses.replace(
        simulation.translate(),
        stats_spec=StatsSpec(
            names=("energy", "peak_power"), periodic=("energy",), period=period
        ),
    )
    return problem, describe_run(problem, monitor=simulation._boundary_monitor())


def _start(
    simulation: Simulation,
    kind: str,
    start: Start | None = None,
    *,
    period: int = 2,
    prepared: tuple[Any, Any] | None = None,
) -> Any:
    """Start a session of `kind` on a simulation's fresh monitored run."""
    problem, request = _prepared(simulation, period) if prepared is None else prepared
    if start is None:
        start = Start(field=problem.initial.field0[0], z=float(problem.initial.z))
    context = {
        "grid": simulation.grid,
        "modes": simulation.modes,
        "profiles": simulation.profiles,
    }
    if kind == "driver":
        driver = select("native", problem, **context)
        return DriverSession(driver, problem, request, start)
    return native_session.NativeSession(problem, request, start, **context)


def _serve(session: Any, *, error: bool = False) -> Served:
    """Advance a session to its end, emitting its warnings as a service loop does.

    With `error`, the warnings are raised: a native warning barrier then ends the run
    at its committed endpoint, its field exported first as a failure checkpoint is.
    """
    batches: list[EventBatch] = []
    raised = None
    events: list[ServiceRequired] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error" if error else "always", BoundaryWarning)
        while not session.finished:
            batch = session.advance()
            batches.append(batch)
            stop = False
            for event in batch.events:
                if isinstance(event, ServiceRequired) and event.kind is (
                    ServiceKind.WARNING
                ):
                    events.append(event)
                    assert event.ticket is not None and event.ticket.z == event.z
                    field = session.export(event.ticket, IDENTITY)
                    try:
                        emit_warning(event)
                    except BoundaryWarning as warning:
                        assert (
                            field.shape == session.export(event.ticket, IDENTITY).shape
                        )
                        raised = (event.z, warning)
                        stop = True
            if stop:
                break
            session.acknowledge(batch.receipt)
    status = session.status()
    record = session.monitor_record()
    messages = [_identity(one.message) for one in caught]
    return Served(
        Run(batches, [], status), messages, raised, dict(record or {}), events
    )


def _identity(warning: BaseException | str) -> tuple[object, ...]:
    """Return a warning's class, text, arguments and attributes, to compare them."""
    assert isinstance(warning, Warning), "a warning was issued without its instance"
    return (type(warning), str(warning), warning.args, sorted(vars(warning).items()))


def _compare(driver: Served, native: Served) -> None:
    """Require the same trace, warnings, outcome and record."""
    # The same warnings, class, text and attributes (``parameter``) alike.
    assert native.warnings == driver.warnings
    assert all(one[3] == [("parameter", "boundary_check")] for one in native.warnings)
    assert trace(native.run) == trace(driver.run)
    assert native.record == driver.record
    for event in native.warning_events:
        # A warning is reported at the committed endpoint its check judged.
        if "at z = " in event.message:
            where = float(event.message.split("at z = ")[1].split(" m")[0])
            assert where == pytest.approx(event.z, rel=1e-6)


def _both(simulation_of, *, period: int = 2, **serve: Any) -> tuple[Served, Served]:
    served = []
    for kind in ("driver", "native"):
        session = _start(simulation_of(), kind, period=period)
        try:
            served.append(_serve(session, **serve))
        finally:
            session.close()
    return served[0], served[1]


MARCHES = {"rk": RungeKutta, "adams": Multistep}


@pytest.mark.parametrize("march", sorted(MARCHES))
@pytest.mark.parametrize("scheme", ["nonlinear", "split_step"])
def test_a_walking_pulse_warns_and_stops_where_the_driver_does(march, scheme):
    driver, native = _both(
        lambda: walker(1.2 * TRANGE, integrator=MARCHES[march](), scheme=scheme)
    )
    _compare(driver, native)
    terminal = native.run.terminal
    assert terminal.outcome is Outcome.FAILED
    assert isinstance(terminal.error, BoundaryError)
    assert str(terminal.error) == str(driver.run.terminal.error)
    assert terminal.z == driver.run.terminal.z
    assert native.warning_events, "the walk did not warn before it stopped"


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_a_raised_warning_ends_the_run_at_its_endpoint(march):
    driver, native = _both(
        lambda: walker(0.47e-12, integrator=MARCHES[march]()), error=True
    )
    terminal = driver.run.terminal
    assert terminal.outcome is Outcome.FAILED
    assert isinstance(terminal.error, BoundaryWarning)
    assert native.raised is not None
    z, warning = native.raised
    assert terminal.error is not None
    assert (z, _identity(warning)) == (terminal.z, _identity(terminal.error))
    # The observations before it are the driver's.
    assert trace(native.run) == trace(driver.run)[:-1]


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_a_clear_run_keeps_the_driver_s_record(march):
    driver, native = _both(lambda: walker(0.1e-12, integrator=MARCHES[march]()))
    _compare(driver, native)
    assert native.run.terminal.outcome is Outcome.COMPLETED
    assert native.record is not None
    assert native.record["steps"] + native.record["skipped"] > 0
    assert len(native.record["planes"]["z"]) == len(native.run.observations)


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_a_quiet_collar_rests_and_returns_as_the_driver_s_does(march):
    # A pulse that starts clear and walks well into the collar, as in
    # ``tests/test_monitor.py``: the collar rests at first, its interval capping the
    # steps, and absorbs once light arrives.
    check = BoundaryCheck(budget=1.0, stop_factor=1.0, quiet_collar=True)
    driver, native = _both(
        lambda: walker(
            1.8 * TRANGE, trange=3 * TRANGE, check=check, integrator=MARCHES[march]()
        )
    )
    _compare(driver, native)
    assert native.record is not None
    assert native.record["collar_left_out_checks"] > 0
    assert native.record["removed_fraction"] > 0.1


def test_the_final_removal_warns_before_completion():
    # A walk that the relaxed guards let leave the window: the collar removes it, and
    # the final check warns before the run completes.
    check = BoundaryCheck(budget=1e-3, stop_factor=1e3)
    driver, native = _both(lambda: walker(0.9e-12, check=check))
    _compare(driver, native)
    assert native.run.terminal.outcome is Outcome.COMPLETED
    assert any("temporal collar removed" in str(one[1]) for one in native.warnings)
    last = native.warning_events[-1]
    assert "temporal collar removed" in last.message
    # The barrier precedes the terminal event and follows the last plane.
    events = native.run.events
    position = events.index(last)
    assert isinstance(events[position + 1], Terminal)
    assert isinstance(events[position - 1], ObservationReady)


def test_a_raised_final_warning_is_a_failure_with_its_field():
    check = BoundaryCheck(budget=1e-3, stop_factor=1e3)
    driver, native = _both(lambda: walker(0.9e-12, check=check), error=True)
    terminal = driver.run.terminal
    assert terminal.outcome is Outcome.FAILED
    assert native.raised is not None
    assert terminal.error is not None
    assert (native.raised[0], _identity(native.raised[1])) == (
        terminal.z,
        _identity(terminal.error),
    )


def test_a_replay_the_light_outgrows_is_refused_where_the_driver_refuses():
    driver, native = _both(spreading)
    _compare(driver, native)
    terminal = native.run.terminal
    assert terminal.outcome is Outcome.FAILED
    assert isinstance(terminal.error, BoundaryError)
    assert "crossing limit fell to" in str(terminal.error)
    assert str(terminal.error) == str(driver.run.terminal.error)


@pytest.mark.parametrize("closing", [True, False])
def test_a_resumed_monitor_continues_the_driver_s_record(closing):
    # The first execution runs to its second plane; both sessions then resume there
    # from its record, the native one reconstructing it natively.
    simulation = walker(0.3e-12, n_save=4)
    first = _start(simulation, "driver")
    try:
        state = None
        record: dict[str, Any] = {}
        while not first.finished:
            batch = first.advance()
            planes = [
                event
                for event in batch.events
                if isinstance(event, ObservationReady) and event.plane_index == 1
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
    if not closing:
        record.pop("planes")
    z, field = state
    served = []
    for kind in ("driver", "native"):
        start = Start(
            field=field,
            z=z,
            plane_index=2,
            plane_pending=False,
            monitor={"record": record, "plane": True, "unwatched": [(0.0, 0.0)]},
        )
        session = _start(walker(0.3e-12, n_save=4), kind, start)
        try:
            served.append(_serve(session))
        finally:
            session.close()
    driver, native = served
    _compare(driver, native)
    assert native.record is not None
    assert native.record["resumed_at"] == [z]
    assert native.record["steps"] > record["steps"]


@pytest.mark.parametrize("march", sorted(MARCHES))
def test_a_warned_step_s_row_follows_its_barrier_indexed_once(march):
    # A row every accepted step: the warned step's row is taken after its barrier, in
    # the next chunk, and still indexes that step's attempt in the assembled history.
    driver, native = _both(
        lambda: walker(1.2 * TRANGE, integrator=MARCHES[march]()), period=1
    )
    _compare(driver, native)
    assert native.warning_events
    for event in native.run.observations[1:]:
        rows = event.step_stats
        assert rows.n_rows == int(event.history.accepted.sum())
        np.testing.assert_array_equal(event.history.z[rows.step], rows.z)


def _through_serve(simulation: Simulation, kind: str, *, error: bool = False) -> Any:
    """Run a monitored session through the service loop; return its output.

    The production selection of the native session is not wired yet: the loop is
    handed the session directly, with the context ``run_problem`` would give it.
    """
    from solprop.driver.run import field_retention  # noqa: PLC0415
    from solprop.driver.service import ServiceContext, serve  # noqa: PLC0415
    from solprop.driver.simulation import metadata_for  # noqa: PLC0415
    from solprop.log import resolve_reporter  # noqa: PLC0415
    from solprop.output import MemoryOutput  # noqa: PLC0415

    prepared = _prepared(simulation)
    problem, request = prepared
    session = _start(simulation, kind, prepared=prepared)
    output = MemoryOutput()
    context = ServiceContext(
        problem=problem,
        request=request,
        metadata=metadata_for(problem, simulation.grid, modes=simulation.modes),
        reporter=resolve_reporter("none"),
        retention=field_retention(problem, "all", ()),
        resume_history=None,
        snapshots=None,
        tdse_binding=None,
        tdse_archive=None,
        recipe=None,
        start_plane=0,
    )
    caught: list[warnings.WarningMessage] = []
    raised: BaseException | None = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error" if error else "always", BoundaryWarning)
            try:
                serve(session, output, context)
            except (BoundaryError, BoundaryWarning) as failure:
                raised = failure
    finally:
        session.close()
    return output, [_identity(one.message) for one in caught], raised


def _saved(output: Any) -> dict[str, Any]:
    """Return what a run's output holds: planes, fields, statistics and records."""
    outcome = output.outcome
    record = dict(outcome.execution["boundary_monitor"])
    return {
        "z": np.asarray(output["z"]).tolist(),
        "fields": np.asarray(output.fields("fft")).tobytes(),
        "energy": np.asarray(output["stats/energy"]).tobytes(),
        "status": outcome.status,
        "monitor": record,
    }


@pytest.mark.parametrize(
    ("drift", "error", "failure"),
    [
        (1.2 * TRANGE, False, BoundaryError),
        (0.47 * TRANGE, True, BoundaryWarning),
        (0.1 * TRANGE, False, type(None)),
    ],
    ids=["stopped", "raised", "clear"],
)
def test_a_monitored_native_session_runs_through_the_service_loop(
    drift, error, failure
):
    # The loop serves the native session as it serves the driver's: the same planes
    # and fields, the same warnings issued by the loop from the barriers, the same
    # failure and the same monitor record in the run's execution record.
    driver = _through_serve(walker(drift), "driver", error=error)
    native = _through_serve(walker(drift), "native", error=error)
    assert native[1] == driver[1]
    assert isinstance(native[2], failure)
    assert (type(native[2]), str(native[2])) == (type(driver[2]), str(driver[2]))
    saved = _saved(native[0])
    assert saved == _saved(driver[0])
    assert saved["status"] == ("complete" if failure is type(None) else "failed")
