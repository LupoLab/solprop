# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native session's method history, call ledger and failures outside a march.

SEP 47 sections 5.4 and 5.5: the native session owns the Adams march's state, so it
exports the method history (:class:`~solprop.output.schema.MarchSnapshot`) at every
accepted endpoint a ticket names, installs a resumed run's history and call ledgers
from its start, and states the ledger in its execution record. Each is held to the
transitional :class:`~solprop.backends.driver_session.DriverSession` over the same
native backend, which runs the same strict march: on one thread, bitwise.

A failure outside the session -- a signal arriving during a native call, a service
failure between batches -- still leaves the committed state exportable for its
checkpoint, with every attempt made delivered once.
"""

import dataclasses
import signal
import warnings
from typing import Any

import numpy as np
import pytest

from sessions.executors import IDENTITY, PLANES, Started, drive, fibre, window
from sessions.test_native_session import _case, _driver_session, _one_thread
from solprop._interrupts import handled
from solprop.backends.driver_session import DriverSession
from solprop.backends.events import ObservationReady, Outcome
from solprop.backends.registry import select
from solprop.backends.session import SessionError, Start
from solprop.backends.sessions import start_session
from solprop.exceptions import BoundaryWarning, ResponseBudgetError, RunTerminated
from solprop.output.schema import MarchSnapshot
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.run_request import describe_run

native_session = pytest.importorskip("solprop_native.session")
native_backend = pytest.importorskip("solprop_native.backend")

LENGTH = PLANES[-1]

ADAMS = {
    "adams": {"integrator": Multistep()},
    "adams-interaction": {"integrator": Multistep(formulation="interaction")},
    "adams-replay": {
        "integrator": Multistep(),
        "schedule": FixedSchedule(np.full(40, LENGTH / 40)),
    },
}
"""Both formulations of the native Adams march, adaptive and replayed."""

needs_default_sigterm = pytest.mark.skipif(
    signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL,
    reason="SIGTERM already has a handler in this process",
)


def _native(problem, request, start, **controls) -> Any:
    """Return a native session over a fresh native backend.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.
    request : solprop.translate.run_request.RunRequest
        Its run request.
    start : solprop.backends.session.Start
        Where it starts.
    **controls
        ``stop`` and ``listener``.

    Returns
    -------
    solprop_native.session.NativeSession
        The session.
    """
    return native_session.NativeSession(
        problem, request, start, grid=window(), modes=[fibre().mode], **controls
    )


def _entrance(problem) -> Start:
    """Return a problem's fresh start.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.

    Returns
    -------
    solprop.backends.session.Start
        The entrance.
    """
    return Start(field=problem.initial.field0[0], z=float(problem.initial.z))


def _snapshots(started: Started) -> list[tuple[MarchSnapshot | None, int | None]]:
    """Drive a session to its end; return each plane's snapshot and its size.

    Parameters
    ----------
    started : sessions.executors.Started
        The session.

    Returns
    -------
    list
        ``(snapshot, nbytes)`` per observation, the launch's included.
    """
    taken: list[tuple[MarchSnapshot | None, int | None]] = []

    def take(session, batch) -> None:
        for event in batch.events:
            if isinstance(event, ObservationReady):
                taken.append(
                    (
                        session.march_snapshot(event.ticket),
                        session.march_snapshot_nbytes(),
                    )
                )

    drive(started, between=take)
    started.session.close()
    return taken


def _assert_same_snapshot(got: MarchSnapshot, expected: MarchSnapshot) -> None:
    """Assert two method histories are bitwise one.

    Parameters
    ----------
    got : solprop.output.schema.MarchSnapshot
        The snapshot under test.
    expected : solprop.output.schema.MarchSnapshot
        The one it must reproduce.
    """
    for name in (
        "z",
        "proposal",
        "method",
        "order",
        "rejections",
        "last_rejected",
        "identities",
        "collar_was_quiet",
        "history_key",
    ):
        assert getattr(got, name) == getattr(expected, name), name
    for name in ("state", "positions", "slots"):
        one, other = getattr(got, name), getattr(expected, name)
        assert one.shape == other.shape and one.dtype == other.dtype, name
        assert one.tobytes() == other.tobytes(), name
    assert got.rows.keys() == expected.rows.keys()
    for name, row in got.rows.items():
        assert row.shape == expected.rows[name].shape, name
        assert row.tobytes() == expected.rows[name].tobytes(), name


@pytest.mark.parametrize("case", sorted(ADAMS))
def test_the_session_exports_the_drivers_method_history(case):
    """SEP 47 section 5.4: at every plane the native session's snapshot is the one the
    backend's own march gives over the driver session, bitwise on one thread: the
    state, the history's positions, identities and rows, and the march's scalars;
    its size is known before it is copied.
    """
    if not _one_thread():
        pytest.skip("bitwise agreement is claimed on one thread (SOLPROP_THREADS=1)")
    problem = _case(**ADAMS[case])
    request = describe_run(problem)
    expected = _snapshots(_driver_session(problem, request))
    got = _snapshots(Started(_native(problem, request, _entrance(problem))))
    assert len(got) == len(expected) > len(PLANES)
    # The launch holds no history; every later plane does.
    assert got[0] == (None, None) == expected[0]
    for (snapshot, nbytes), (reference, size) in zip(
        got[1:], expected[1:], strict=True
    ):
        assert snapshot is not None and reference is not None
        _assert_same_snapshot(snapshot, reference)
        assert nbytes == size == snapshot.nbytes


def test_a_snapshot_is_the_state_its_ticket_names():
    """SEP 47 section 5.4: the snapshot's state is the committed field the ticket
    exports, its newest sample at the ticket's position; a stale ticket is refused,
    and a Runge-Kutta march keeps none.
    """
    problem = _case(integrator=Multistep(), step_planes=0)
    request = describe_run(problem)
    session = _native(problem, request, _entrance(problem))
    try:
        session.acknowledge(session.advance().receipt)
        batch = session.advance()
        (event,) = batch.events
        assert isinstance(event, ObservationReady)
        snapshot = session.march_snapshot(event.ticket)
        assert snapshot is not None
        assert snapshot.z == event.z == snapshot.positions[-1]
        field = np.asarray(session.export(event.ticket, IDENTITY))
        assert snapshot.state.tobytes() == field[np.newaxis].tobytes()
        assert session.march_snapshot_nbytes() == snapshot.nbytes
        session.acknowledge(batch.receipt)
        with pytest.raises(SessionError, match="expired"):
            session.march_snapshot(event.ticket)
    finally:
        session.close()
    problem = _case(step_planes=0)
    session = _native(problem, describe_run(problem), _entrance(problem))
    try:
        session.acknowledge(session.advance().receipt)
        batch = session.advance()
        (event,) = batch.events
        assert session.march_snapshot(event.ticket) is None
        assert session.march_snapshot_nbytes() is None
    finally:
        session.close()


def _resumed(problem, snapshot: MarchSnapshot, ledger: dict[str, object]) -> Start:
    """Return a resume at a snapshot's plane, carrying its history and ledger.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.
    snapshot : solprop.output.schema.MarchSnapshot
        The history, at a save plane.
    ledger : dict
        The call ledger's record there.

    Returns
    -------
    solprop.backends.session.Start
        The start.
    """
    index = int(np.searchsorted(problem.stepper.save_planes, snapshot.z, "right"))
    return Start(
        field=np.array(snapshot.state[0]),
        z=snapshot.z,
        plane_index=index,
        plane_pending=False,
        resume=True,
        march=snapshot,
        ledgers={"logical": ledger},
    )


def _at_second_plane(started: Started) -> tuple[MarchSnapshot, dict[str, object]]:
    """Run a session to its second plane; return its history and ledger there.

    Parameters
    ----------
    started : sessions.executors.Started
        The session, closed on return.

    Returns
    -------
    tuple
        The snapshot and the call ledger's record.
    """
    session = started.session
    planes = 0
    try:
        while True:
            batch = session.advance()
            for event in batch.events:
                if isinstance(event, ObservationReady) and event.plane_index >= 0:
                    planes += 1
                    if planes == 2:
                        snapshot = session.march_snapshot(event.ticket)
                        assert snapshot is not None
                        record = session.execution_record()["response_calls"]
                        assert isinstance(record, dict)
                        return snapshot, record
            session.acknowledge(batch.receipt)
    finally:
        session.close()


@pytest.mark.parametrize("case", sorted(ADAMS))
def test_a_resumed_history_continues_as_the_drivers_does(case):
    """SEP 47 section 5.5: a start carrying the Adams history and the call ledger
    continues the march as the backend's own march does from the same start: the
    same trace and fields, bitwise on one thread, and the same ledger at the end,
    cumulative from the restored one; the history was installed, not refused.
    """
    if not _one_thread():
        pytest.skip("bitwise agreement is claimed on one thread (SOLPROP_THREADS=1)")
    problem = _case(step_planes=0, **ADAMS[case])
    request = describe_run(problem)
    snapshot, ledger = _at_second_plane(
        Started(_native(problem, request, _entrance(problem)))
    )
    assert ledger["accepted_steps"]
    driver = select("native", problem, grid=window(), modes=[fibre().mode])
    baseline_session = DriverSession(
        driver, problem, request, _resumed(problem, snapshot, ledger)
    )
    assert baseline_session.march_refusal is None
    baseline = drive(Started(baseline_session), export=IDENTITY)
    expected = driver.metadata()["response_calls"]
    baseline_session.close()
    session = _native(problem, request, _resumed(problem, snapshot, ledger))
    assert session.march_refusal is None
    run = drive(Started(session), export=IDENTITY)
    got = session.execution_record()["response_calls"]
    session.close()
    from sessions.executors import assert_same_run  # noqa: PLC0415 - beside its use

    assert_same_run(run, baseline)
    assert got == expected
    assert got["accepted_steps"] > ledger["accepted_steps"]


def test_the_ledger_record_is_the_drivers():
    """The execution record's call ledger is the native march's own: its counters and
    every accepted step's order, as the driver session's backend records them.
    """
    if not _one_thread():
        pytest.skip("bitwise agreement is claimed on one thread (SOLPROP_THREADS=1)")
    problem = _case(integrator=Multistep())
    request = describe_run(problem)
    started = _driver_session(problem, request)
    drive(started)
    driver = started.session.driver  # pyright: ignore[reportAttributeAccessIssue]
    expected = driver.metadata()["response_calls"]
    started.session.close()
    session = _native(problem, request, _entrance(problem))
    drive(Started(session))
    got = session.execution_record()["response_calls"]
    session.close()
    assert got == expected
    assert sum(count for _, count in got["orders"]) == got["accepted_steps"] > 0


def test_the_budget_stops_the_session_as_it_stops_the_drivers_march():
    """The source-call budget ends the native session where it ends the backend's
    march: the same error, message, position, field and calls.
    """
    problem = _case(
        integrator=Multistep(max_calls=40),
        step_planes=0,
        dz_init=LENGTH / 400,
        dz_max=LENGTH / 400,
    )
    request = describe_run(problem)
    baseline = drive(_driver_session(problem, request))
    run = drive(Started(_native(problem, request, _entrance(problem))))
    got, expected = run.terminal.error, baseline.terminal.error
    assert isinstance(got, ResponseBudgetError)
    assert isinstance(expected, ResponseBudgetError)
    assert str(got) == str(expected)
    assert (got.z, got.calls) == (expected.z, expected.calls)
    field, reference = np.asarray(got.field), np.asarray(expected.field)
    assert field.shape == reference.shape == problem.shape
    if _one_thread():
        assert field.tobytes() == reference.tobytes()


def test_a_failure_between_batches_still_checkpoints_the_committed_state():
    """SEP 47 section 5.5: a failure outside the session after an acknowledgment --
    a service failure, a signal between batches -- issues a failed terminal batch at
    the committed state, whose field and history are exported for the checkpoint;
    nothing advances again.
    """
    problem = _case(integrator=Multistep(), step_planes=0)
    session = _native(problem, describe_run(problem), _entrance(problem))
    try:
        session.acknowledge(session.advance().receipt)
        batch = session.advance()
        (plane,) = batch.events
        assert isinstance(plane, ObservationReady)
        session.acknowledge(batch.receipt)
        ticket = session.pending_ticket()
        assert ticket is not None
        assert ticket.z == plane.z and ticket.generation == plane.ticket.generation
        assert session.pending_ticket() == ticket
        field = np.asarray(session.export(ticket, IDENTITY))
        assert np.all(np.isfinite(field))
        snapshot = session.march_snapshot(ticket)
        assert snapshot is not None and snapshot.z == plane.z
        undelivered, rows = session.undelivered(ticket)
        assert undelivered.n_steps == 0 and rows.n_rows == 0
        assert session.status().outcome is Outcome.FAILED
        session.acknowledge(ticket.receipt)
        assert session.finished
        assert session.pending_ticket() is None
        with pytest.raises(SessionError):
            session.advance()
    finally:
        session.close()


def _armed(monkeypatch, stage: str, after: int, category: str) -> None:
    """Arm a native march fault when a native session configures its march.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Restores the march's configuration.
    stage : str
        ``"startup"``, ``"trial"`` or ``"accept"``.
    after : int
        That stage's runs before it fires.
    category : str
        The failure, or ``"sigint"`` or ``"sigterm"`` for a signal raised in the call.
    """
    march = native_backend._NativeAdamsMarch
    configure = march._configure

    def arming(self) -> None:
        configure(self)
        self._backend._native_state.adams_march_fault(stage, after, category)

    monkeypatch.setattr(march, "_configure", arming)


@needs_default_sigterm
def test_a_signal_during_an_advance_is_raised_with_its_batch_pending(monkeypatch):
    """SEP 47 section 5.5: a signal arriving while the native call holds no Python
    lock runs its handler as the advance returns; the run's handler raises it once the
    batch is pending and its chunk held, so the checkpoint exports that batch's state
    and includes every attempt the advance made, once, with the history there.
    """
    # Steps short enough that the advance's bound, not a plane, returns it.
    problem = _case(
        integrator=Multistep(),
        step_planes=0,
        dz_init=LENGTH / 400,
        dz_max=LENGTH / 400,
    )
    _armed(monkeypatch, "trial", 5, "sigterm")
    session = _native(problem, describe_run(problem), _entrance(problem))
    try:
        session.acknowledge(session.advance().receipt)
        with handled(), pytest.raises(RunTerminated):
            session.advance(work_limit=50, deadline=1e9)
        status = session.status()
        # The call went on to its bound after the signal arrived.
        assert status.attempts == 50
        ticket = session.pending_ticket()
        assert ticket is not None and ticket.z == status.z > 0.0
        history, _ = session.undelivered(ticket)
        assert history.n_steps == 50
        assert history.z[history.accepted][-1] == ticket.z
        assert session.undelivered(ticket)[0].n_steps == 0
        snapshot = session.march_snapshot(ticket)
        assert snapshot is not None and snapshot.z == ticket.z
        field = np.asarray(session.export(ticket, IDENTITY))
        assert snapshot.state.tobytes() == field[np.newaxis].tobytes()
    finally:
        session.close()


def test_a_backend_that_steps_on_the_host_is_served_by_the_driver(monkeypatch):
    """A backend whose own march would step on the host (here the private switch) is
    served by the transitional session over that march, decided before the start's
    field is taken; otherwise production runs on the native session.
    """
    problem = _case(integrator=Multistep(), step_planes=0)
    request = describe_run(problem)
    for switch, served in (("native", native_session.NativeSession), ("host", None)):
        monkeypatch.setattr(native_backend, "_ADAMS_MARCH", switch)
        backend = select("native", problem, grid=window(), modes=[fibre().mode])
        start = _entrance(problem)
        session = start_session(backend, problem, request, start)
        try:
            if served is None:
                assert isinstance(session, DriverSession)
                assert native_session.backend_refusal(backend) is not None
            else:
                assert isinstance(session, served)
                assert native_session.backend_refusal(backend) is None
        finally:
            session.close()


def _noisy(noise: str, *, check: bool) -> Any:
    """Return a short soliton with a noise model, its boundary monitor on or off."""
    from solprop.driver import Simulation  # noqa: PLC0415 - beside its one use
    from solprop.monitor import BoundaryCheck  # noqa: PLC0415 - beside its one use
    from solprop.pulses import SechPulse  # noqa: PLC0415 - beside its one use

    grid = window()
    simulation = Simulation.from_gnlse(
        0.11, LENGTH, [0.0, 0.0, -1.183e-26], wavelength=835e-9, grid=grid,
        pulses=[SechPulse(wavelength=835e-9, width=50e-15, peak_power=1e3)],
        noise=noise, seed=7, raman=False, shock=False, n_save=len(PLANES) + 1,
        backend="native", integrator=Multistep(),
    )  # fmt: skip
    simulation.settings = dataclasses.replace(
        simulation.settings, boundary_check=BoundaryCheck() if check else None
    )
    return simulation


@pytest.mark.parametrize("check", [True, False], ids=["monitored", "unmonitored"])
@pytest.mark.parametrize("noise", ["input", "modified"])
def test_a_noisy_run_is_the_drivers(noise, check):
    """The input and modified noise models run on the native session as on the
    backend's own march: the same noise phase at every plane, and on one thread the
    same trace, fields and boundary-monitor record, the input draw's vacuum floor
    read by the native monitor as solprop's monitor reads it.
    """
    simulation = _noisy(noise, check=check)
    problem = simulation.translate()
    assert problem.initial.noise_model == noise
    request = describe_run(problem, monitor=simulation._boundary_monitor())
    assert native_session.refusal(problem, request) is None
    context = {"grid": simulation.grid, "modes": simulation.modes}
    phases: dict[str, list[np.ndarray]] = {"driver": [], "native": []}
    runs = {}
    records = {}
    for kind in ("driver", "native"):
        start = Start(field=problem.initial.field0[0], z=float(problem.initial.z))
        if kind == "driver":
            session: Any = DriverSession(
                select("native", problem, **context), problem, request, start
            )
        else:
            session = native_session.NativeSession(problem, request, start, **context)

        def phase(one, batch, found=phases[kind]) -> None:
            for event in batch.events:
                if isinstance(event, ObservationReady):
                    found.append(np.array(one.noise_phase(event.ticket)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BoundaryWarning)
            runs[kind] = drive(Started(session), export=IDENTITY, between=phase)
        records[kind] = session.monitor_record()
        session.close()
    assert len(phases["native"]) == len(phases["driver"]) > len(PLANES)
    for got, expected in zip(phases["native"], phases["driver"], strict=True):
        assert got.shape == problem.shape[1:]
        assert got.tobytes() == expected.tobytes()
    assert (records["native"] is None) is not check
    if _one_thread():
        from sessions.executors import assert_same_run  # noqa: PLC0415

        assert_same_run(runs["native"], runs["driver"])
        assert records["native"] == records["driver"]
