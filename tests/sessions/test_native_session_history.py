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

from sessions.executors import (
    IDENTITY,
    PLANES,
    Started,
    drive,
    fibre,
    skip_device_refused,
    window,
)
from sessions.test_native_session import _case, _driver_session, _one_thread
from solprop._interrupts import handled
from solprop.backends.driver_session import DriverSession
from solprop.backends.events import ObservationReady, Outcome
from solprop.backends.registry import select
from solprop.backends.session import SessionError, Start
from solprop.backends.sessions import start_session
from solprop.exceptions import (
    BackendError,
    BoundaryWarning,
    ResponseBudgetError,
    RunTerminated,
)
from solprop.output.schema import MarchSnapshot
from solprop.stepper.low_storage import LowStorageRungeKutta
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

SNAPSHOTTED = {**ADAMS, "adams-dense": {"integrator": Multistep(), "interpolate": True}}
"""The Adams cases, and dense output, whose interpolated planes hold no history."""

needs_default_sigterm = pytest.mark.skipif(
    signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL,
    reason="SIGTERM already has a handler in this process",
)


def _native(problem, request, start, **controls) -> Any:
    """Return a native session over a fresh native backend.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
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
    skip_device_refused(problem, request)
    return native_session.NativeSession(
        problem, request, start, grid=window(), modes=[fibre().mode], **controls
    )


def _entrance(problem) -> Start:
    """Return a problem's fresh start.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
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


@pytest.mark.parametrize("case", sorted(SNAPSHOTTED))
def test_the_session_exports_the_drivers_method_history(case):
    """SEP 47 section 5.4: at every plane the native session's snapshot is the one the
    backend's own march gives over the driver session, bitwise on one thread: the
    state, the history's positions, identities and rows, and the march's scalars;
    its size is known before it is copied.
    """
    if not _one_thread():
        pytest.skip("bitwise agreement is claimed on one thread (SOLPROP_THREADS=1)")
    problem = _case(**SNAPSHOTTED[case])
    request = describe_run(problem)
    expected = _snapshots(_driver_session(problem, request))
    got = _snapshots(Started(_native(problem, request, _entrance(problem))))
    assert len(got) == len(expected) > len(PLANES)
    # The launch holds no history; every later plane does.
    assert got[0] == (None, None) == expected[0]
    for (snapshot, nbytes), (reference, size) in zip(
        got[1:], expected[1:], strict=True
    ):
        if case.endswith("dense") and reference is None:
            # A dense plane: the march stands ahead of it and keeps no history there.
            assert (snapshot, nbytes, size) == (None, None, None)
            continue
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
    problem : solprop.translate.problem.PropagationProblem
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
        # Nothing is pending to bind a ticket to (the protocol's answer), until the
        # failure's own terminal batch is issued.
        assert session.pending_ticket() is None
        ticket = session.failure_ticket()
        assert ticket is not None
        assert ticket.z == plane.z and ticket.generation == plane.ticket.generation
        assert session.pending_ticket() == ticket == session.failure_ticket()
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
        assert session.failure_ticket() is None
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


def _noisy(noise: Any, *, check: bool, march: str = "adams") -> Any:
    """Return a short soliton with a noise model, its boundary monitor on or off, on
    the adaptive Adams march, densely observed, or replayed by the low-storage march.
    """
    from solprop.driver import Simulation  # noqa: PLC0415 - beside its one use
    from solprop.monitor import BoundaryCheck  # noqa: PLC0415 - beside its one use
    from solprop.pulses import SechPulse  # noqa: PLC0415 - beside its one use
    from solprop.stepper import LowStorageRungeKutta  # noqa: PLC0415

    grid = window()
    method: dict[str, Any] = {"integrator": Multistep()}
    if march == "low-storage":
        method = {
            "integrator": LowStorageRungeKutta(),
            "schedule": FixedSchedule.uniform(
                0.0, LENGTH, LENGTH / 80, save_planes=list(PLANES)
            ),
        }
    simulation = Simulation.from_gnlse(
        0.11, LENGTH, [0.0, 0.0, -1.183e-26], wavelength=835e-9, grid=grid,
        pulses=[SechPulse(wavelength=835e-9, width=50e-15, peak_power=1e3)],
        noise=noise, seed=7, raman=False, shock=False, n_save=len(PLANES) + 1,
        backend="native", **method,
    )  # fmt: skip
    simulation.settings = dataclasses.replace(
        simulation.settings,
        boundary_check=BoundaryCheck() if check else None,
        interpolate_planes=march == "dense",
    )
    return simulation


@pytest.mark.parametrize("march", ["adams", "dense", "low-storage"])
@pytest.mark.parametrize("check", [True, False], ids=["monitored", "unmonitored"])
@pytest.mark.parametrize("noise", ["input", "modified"])
def test_a_noisy_run_is_the_drivers(noise, check, march):
    """The input and modified noise models run on the native session as on the
    backend's own march, adaptive Adams, densely observed or low-storage: the same
    noise phase at every plane, and on one thread the same trace, fields and
    boundary-monitor record, the input draw's vacuum floor read by the native monitor
    as solprop's monitor reads it.
    """
    if march == "dense" and noise == "modified":
        pytest.skip("the translation refuses dense output beside the modified noise")
    simulation = _noisy(noise, check=check, march=march)
    problem = simulation.translate()
    assert problem.initial.noise_model == noise
    request = describe_run(problem, monitor=simulation._boundary_monitor())
    skip_device_refused(problem, request)
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


def test_a_noisy_run_on_a_z_table_is_refused_once_the_backend_is_built():
    """The noise phase at a plane reads a z table's propagator phase, which the leased
    state holds: the native session refuses the built backend by name, directly as
    through the door, before the start's field is taken and leaving the backend open
    to the transitional session, which serves the run.
    """
    from driver.test_specified_paths import (  # noqa: PLC0415 - beside its one use
        GRID,
        PULSE,
        linear_walkoff,
    )
    from driver.test_specified_paths import (  # noqa: PLC0415
        LENGTH as WALKOFF_LENGTH,
    )
    from solprop.driver import Simulation  # noqa: PLC0415 - beside its one use
    from solprop.grid import EnvelopeGrid  # noqa: PLC0415 - beside its one use

    grid = EnvelopeGrid.from_limits(WALKOFF_LENGTH, 835e-9, (600e-9, 1300e-9), 2e-12)
    simulation = Simulation.from_gnlse(
        0.1, WALKOFF_LENGTH, fibre=linear_walkoff(grid), shock=False, n_save=4,
        noise="input", seed=3, backend="native", integrator=Multistep(),
        **GRID, **PULSE,
    )  # fmt: skip
    problem = simulation.translate()
    request = describe_run(problem)
    skip_device_refused(problem, request)
    assert native_session.refusal(problem, request) is None
    context = {"grid": simulation.grid, "modes": simulation.modes}
    backend = select("native", problem, **context)
    try:
        start = Start(field=problem.initial.field0[0], z=float(problem.initial.z))
        with pytest.raises(BackendError, match="'input' noise model on a z table"):
            native_session.NativeSession(problem, request, start, backend=backend)
        assert start.field is not None
        session = start_session(backend, problem, request, start, owns=False)
        try:
            assert isinstance(session, DriverSession)
            session.acknowledge(session.advance().receipt)
        finally:
            session.close()
    finally:
        backend.close()


class _TranslationError(RuntimeError):
    """An error translating a native batch, standing in for any such bug."""


def _untranslatable(monkeypatch) -> None:
    """Make the next batch the session translates raise :class:`_TranslationError`."""
    event = native_session.NativeSession._event
    calls: list[int] = []

    def failing(self, *args, **kwargs):
        if len(calls) == 1:
            calls.append(1)
            raise _TranslationError("the batch could not be translated")
        calls.append(1)
        return event(self, *args, **kwargs)

    monkeypatch.setattr(native_session.NativeSession, "_event", failing)


def test_an_error_translating_a_batch_leaves_its_chunk_to_the_checkpoint(monkeypatch):
    """SEP 47 section 5.3: the batch is held before it is translated, so an error in
    the translation leaves its attempts to the failure's checkpoint, delivered once.
    """
    problem = _case(integrator=Multistep(), step_planes=0)
    _untranslatable(monkeypatch)
    session = _native(problem, describe_run(problem), _entrance(problem))
    try:
        session.acknowledge(session.advance().receipt)
        with pytest.raises(_TranslationError):
            session.advance()
        status = session.status()
        assert status.attempts > 0
        ticket = session.failure_ticket()
        assert ticket is not None and ticket.z == status.z
        history, _ = session.undelivered(ticket)
        assert history.n_steps == status.attempts
        assert session.undelivered(ticket)[0].n_steps == 0
    finally:
        session.close()


@needs_default_sigterm
def test_a_signal_beside_an_error_translating_its_batch_is_raised_with_its_cause(
    monkeypatch,
):
    """A signal that arrived during the native call is not dropped beside an error in
    the batch's translation: it is raised, the error its cause.
    """
    problem = _case(
        integrator=Multistep(),
        step_planes=0,
        dz_init=LENGTH / 400,
        dz_max=LENGTH / 400,
    )
    _armed(monkeypatch, "trial", 5, "sigterm")
    _untranslatable(monkeypatch)
    session = _native(problem, describe_run(problem), _entrance(problem))
    try:
        session.acknowledge(session.advance().receipt)
        with handled(), pytest.raises(RunTerminated) as raised:
            session.advance(work_limit=50, deadline=1e9)
        assert isinstance(raised.value.__cause__, _TranslationError)
        ticket = session.failure_ticket()
        assert ticket is not None
        assert session.undelivered(ticket)[0].n_steps == 50
    finally:
        session.close()


RESTARTED = {
    "low-storage": {
        "integrator": LowStorageRungeKutta(),
        "schedule": FixedSchedule(np.full(40, LENGTH / 40)),
    },
    "adams-dense": {"integrator": Multistep(), "interpolate": True},
}
"""Runs the native session now serves that keep no history a resume carries."""


@pytest.mark.parametrize("case", sorted(RESTARTED))
def test_production_selects_the_session_and_a_field_restart_continues_as_the_driver(
    case,
):
    """Production selects the native session for low-storage and dense runs, and a
    field restart from a saved plane (no history: the low-storage march keeps none,
    and a dense plane's march stands ahead of it) continues as the backend's own march
    continues it, bitwise on one thread.
    """
    problem = _case(step_planes=0, **RESTARTED[case])
    request = describe_run(problem)
    skip_device_refused(problem, request)
    backend = select("native", problem, grid=window(), modes=[fibre().mode])
    session = start_session(backend, problem, request, _entrance(problem))
    assert isinstance(session, native_session.NativeSession)
    field = None
    z, index = 0.0, 0
    try:
        planes = 0
        while field is None:
            batch = session.advance()
            for event in batch.events:
                if isinstance(event, ObservationReady) and event.plane_index >= 0:
                    planes += 1
                    if planes == 2:
                        field = np.array(session.export(event.ticket, IDENTITY))
                        z, index = event.z, event.plane_index + 1
                        assert session.march_snapshot(event.ticket) is None
            session.acknowledge(batch.receipt)
    finally:
        session.close()

    def resumed() -> Start:
        return Start(
            field=field.copy(), z=z, plane_index=index, plane_pending=False, resume=True
        )

    driver = select("native", problem, grid=window(), modes=[fibre().mode])
    baseline = drive(
        Started(DriverSession(driver, problem, request, resumed())), export=IDENTITY
    )
    run = drive(Started(_native(problem, request, resumed())), export=IDENTITY)
    assert len(run.observations) == len(baseline.observations) == len(PLANES) - 2
    if _one_thread():
        from sessions.executors import assert_same_run  # noqa: PLC0415

        assert_same_run(run, baseline)
