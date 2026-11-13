# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native session's method history, call ledger and failures outside a march.

SEP 47 sections 5.4 and 5.5: the native session owns the Adams march's state, so it
exports the method history (:class:`~solprop.output.records.MarchSnapshot`) at every
accepted endpoint a ticket names, installs a resumed run's history and call ledgers
from its start, and states the ledger in its execution record. Each is held to the
NumPy reference session on the same problem within the budgets of SEP 25
(:func:`sessions.test_native_session.assert_meets_reference`), and, on one thread, to
the native session's own run: repeated, or uninterrupted where a resume continues it,
bit for bit.

A failure outside the session -- a signal arriving during a native call, a service
failure between batches -- still leaves the committed state exportable for its
checkpoint, with every attempt made delivered once.
"""

import dataclasses
import signal
import warnings
from typing import Any, Literal

import numpy as np
import pytest

from sessions.executors import (
    IDENTITY,
    PLANES,
    Started,
    assert_same_run,
    drive,
    fibre,
    masked,
    record_layout,
    skip_device_refused,
    window,
)
from sessions.test_native_session import (
    PLANE_BOUND,
    _case,
    _one_thread,
    _reference_session,
    assert_meets_reference,
)
from solprop._interrupts import handled
from solprop.backends.events import ObservationReady, Outcome
from solprop.backends.host.session import HostSession
from solprop.backends.registry import select
from solprop.backends.session import SessionError, Start
from solprop.backends.sessions import start_session
from solprop.exceptions import (
    BackendError,
    BoundaryWarning,
    ResponseBudgetError,
    RunTerminated,
)
from solprop.output.records import MarchSnapshot
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule

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
    skip_device_refused(problem)
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
    return Start(field=problem.initial.field0, z=float(problem.initial.z))


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
    got : solprop.output.records.MarchSnapshot
        The snapshot under test.
    expected : solprop.output.records.MarchSnapshot
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


def _assert_close_snapshot(got: MarchSnapshot, expected: MarchSnapshot) -> None:
    """Assert a native method history is the reference's within the run's budgets.

    The march's discrete state -- its method, order, rejections and history layout --
    exactly; its positions and proposal to the accepted-step budget, and its state
    and rows to :data:`~sessions.test_native_session.PLANE_BOUND` of their peaks.

    Parameters
    ----------
    got : solprop.output.records.MarchSnapshot
        The native snapshot.
    expected : solprop.output.records.MarchSnapshot
        The reference's.
    """
    for name in ("method", "order", "rejections", "last_rejected", "collar_was_quiet"):
        assert getattr(got, name) == getattr(expected, name), name
    assert got.z == pytest.approx(expected.z, rel=1e-6)
    assert got.proposal == pytest.approx(expected.proposal, rel=1e-6)
    np.testing.assert_allclose(got.positions, expected.positions, rtol=1e-6)
    for one, other in [(got.state, expected.state)] + [
        (row, expected.rows[name]) for name, row in got.rows.items()
    ]:
        assert one.shape == other.shape and one.dtype == other.dtype
        scale = float(np.max(np.abs(other), initial=0.0))
        np.testing.assert_allclose(one, other, rtol=0, atol=PLANE_BOUND * scale)
    assert got.rows.keys() == expected.rows.keys()


@pytest.mark.parametrize("case", sorted(SNAPSHOTTED))
def test_the_session_exports_its_method_history(case):
    """SEP 47 section 5.4: at every plane the native session exports its method
    history, its size known before it is copied; on one thread a repeated run's are
    bitwise its own -- the state, the history's positions, identities and rows, and
    the march's scalars -- and they meet the reference session's
    (:func:`_assert_close_snapshot`).
    """
    problem, run_request = _case(**SNAPSHOTTED[case])
    request = run_request
    expected = _snapshots(_reference_session(problem, request))
    got = _snapshots(Started(_native(problem, request, _entrance(problem))))
    assert len(got) == len(expected) > len(PLANES)
    # The launch holds no history; every later plane does.
    assert got[0] == (None, None) == expected[0]
    for (snapshot, nbytes), (reference, _) in zip(got[1:], expected[1:], strict=True):
        if case.endswith("dense") and reference is None:
            # A dense plane: the march stands ahead of it and keeps no history there.
            assert (snapshot, nbytes) == (None, None)
            continue
        assert snapshot is not None and reference is not None
        _assert_close_snapshot(snapshot, reference)
        assert nbytes == snapshot.nbytes
    if not _one_thread():
        pytest.skip("bitwise repetition is claimed on one thread (SOLPROP_THREADS=1)")
    again = _snapshots(Started(_native(problem, request, _entrance(problem))))
    for (snapshot, _), (repeated, _) in zip(got[1:], again[1:], strict=True):
        if snapshot is None:
            assert repeated is None
            continue
        assert repeated is not None
        _assert_same_snapshot(repeated, snapshot)


def test_a_snapshot_is_the_state_its_ticket_names():
    """SEP 47 section 5.4: the snapshot's state is the committed field the ticket
    exports, its newest sample at the ticket's position; a stale ticket is refused,
    and a Runge-Kutta march keeps none.
    """
    problem, run_request = _case(integrator=Multistep(), step_planes=0)
    request = run_request
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
        assert snapshot.state.tobytes() == field.tobytes()
        assert session.march_snapshot_nbytes() == snapshot.nbytes
        session.acknowledge(batch.receipt)
        with pytest.raises(SessionError, match="expired"):
            session.march_snapshot(event.ticket)
    finally:
        session.close()
    problem, run_request = _case(step_planes=0)
    session = _native(problem, run_request, _entrance(problem))
    try:
        session.acknowledge(session.advance().receipt)
        batch = session.advance()
        (event,) = batch.events
        assert session.march_snapshot(event.ticket) is None
        assert session.march_snapshot_nbytes() is None
    finally:
        session.close()


def _resumed(
    problem, run_request, snapshot: MarchSnapshot, ledger: dict[str, object]
) -> Start:
    """Return a resume at a snapshot's plane, carrying its history and ledger.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    run_request : solprop.translate.run_request.RunRequest
        Its run's request, whose planes the snapshot's position is among.
    snapshot : solprop.output.records.MarchSnapshot
        The history, at a save plane.
    ledger : dict
        The call ledger's record there.

    Returns
    -------
    solprop.backends.session.Start
        The start.
    """
    index = int(np.searchsorted(run_request.stepper.save_planes, snapshot.z, "right"))
    return Start(
        field=np.array(snapshot.state),
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
def test_a_resumed_history_continues_the_uninterrupted_run(case):
    """SEP 47 section 5.5: a start carrying the Adams history and the call ledger
    continues the march as the run that was not interrupted goes on: on one thread
    the same accepted steps, planes and fields after the resume, bitwise, and the
    same ledger at the end, cumulative from the restored one; the history was
    installed, not refused.
    """
    if not _one_thread():
        pytest.skip("bitwise agreement is claimed on one thread (SOLPROP_THREADS=1)")
    problem, run_request = _case(step_planes=0, **ADAMS[case])
    request = run_request
    snapshot, ledger = _at_second_plane(
        Started(_native(problem, request, _entrance(problem)))
    )
    assert ledger["accepted_steps"]
    whole = _native(problem, request, _entrance(problem))
    baseline = drive(Started(whole), export=IDENTITY)
    expected = whole.execution_record()["response_calls"]
    whole.close()
    session = _native(
        problem, request, _resumed(problem, run_request, snapshot, ledger)
    )
    assert session.march_refusal is None
    run = drive(Started(session), export=IDENTITY)
    got = session.execution_record()["response_calls"]
    session.close()
    later = [one for one in baseline.observations if one.z > snapshot.z]
    assert [(one.kind, one.z) for one in run.observations] == [
        (one.kind, one.z) for one in later
    ]
    ends = baseline.accepted_ends()
    assert run.accepted_ends().tobytes() == ends[ends > snapshot.z].tobytes()
    for field, other in zip(run.fields, baseline.fields[-len(later) :], strict=True):
        assert field.tobytes() == other.tobytes()
    assert got == expected
    assert got["accepted_steps"] > ledger["accepted_steps"]


def test_the_ledger_record_is_the_reference_s():
    """The execution record's call ledger is the native march's own: its counters and
    every accepted step's order, as the reference session's march records them on the
    same problem, which takes the same steps.
    """
    problem, run_request = _case(integrator=Multistep())
    request = run_request
    started = _reference_session(problem, request)
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


def test_the_budget_stops_the_session_as_it_stops_the_reference_s_march():
    """The source-call budget ends the native session where it ends the reference's
    march: the same error, message, position and calls, and the field within the
    plane budget; on one thread a repeated run's field bitwise.
    """
    problem, run_request = _case(
        integrator=Multistep(max_calls=40),
        step_planes=0,
        dz_init=LENGTH / 400,
        dz_max=LENGTH / 400,
    )
    request = run_request
    baseline = drive(_reference_session(problem, request), raises=False)
    run = drive(Started(_native(problem, request, _entrance(problem))), raises=False)
    got, expected = run.terminal.error, baseline.terminal.error
    assert isinstance(got, ResponseBudgetError)
    assert isinstance(expected, ResponseBudgetError)
    assert masked(str(got)) == masked(str(expected))
    assert got.z == pytest.approx(expected.z, rel=1e-12)
    assert got.calls == expected.calls
    field, reference = np.asarray(got.field), np.asarray(expected.field)
    assert field.shape == reference.shape == problem.shape
    scale = float(np.max(np.abs(reference)))
    np.testing.assert_allclose(field, reference, rtol=0, atol=PLANE_BOUND * scale)
    if _one_thread():
        again = drive(
            Started(_native(problem, request, _entrance(problem))), raises=False
        )
        repeated = again.terminal.error
        assert isinstance(repeated, ResponseBudgetError)
        assert np.asarray(repeated.field).tobytes() == field.tobytes()


def test_a_failure_between_batches_still_checkpoints_the_committed_state():
    """SEP 47 section 5.5: a failure outside the session after an acknowledgment --
    a service failure, a signal between batches -- issues a failed terminal batch at
    the committed state, whose field and history are exported for the checkpoint;
    nothing advances again.
    """
    problem, run_request = _case(integrator=Multistep(), step_planes=0)
    session = _native(problem, run_request, _entrance(problem))
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
    problem, run_request = _case(
        integrator=Multistep(),
        step_planes=0,
        dz_init=LENGTH / 400,
        dz_max=LENGTH / 400,
    )
    _armed(monkeypatch, "trial", 5, "sigterm")
    session = _native(problem, run_request, _entrance(problem))
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
        assert snapshot.state.tobytes() == field.tobytes()
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
def test_a_noisy_run_meets_the_reference(noise, check, march):
    """The input and modified noise models run on the native session, adaptive Adams,
    densely observed or low-storage: the reference session's noise phase at every
    plane to the plane budget, its run within the budgets of
    :func:`~sessions.test_native_session.assert_meets_reference`, the same
    boundary-monitor record layout, and on one thread a repeated run's trace, fields,
    phases and record bit for bit.
    """
    if march == "dense" and noise == "modified":
        pytest.skip("the translation refuses dense output beside the modified noise")
    simulation = _noisy(noise, check=check, march=march)
    problem = simulation.translate()
    run_request = simulation._run_request()
    assert problem.initial.noise_model == noise
    request = run_request.for_run(monitor=simulation._boundary_monitor())
    skip_device_refused(problem)
    assert native_session.refusal(problem, request) is None
    context = {"grid": simulation.grid, "modes": simulation.modes}
    kinds = ("reference", "native", "again")
    phases: dict[str, list[np.ndarray]] = {kind: [] for kind in kinds}
    runs = {}
    records = {}
    for kind in kinds:
        start = Start(field=problem.initial.field0, z=float(problem.initial.z))
        if kind == "reference":
            session: Any = HostSession(
                select("numpy_ref", problem, run_request, **context),
                problem,
                request,
                start,
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
    _assert_phases_meet(phases["native"], phases["reference"], problem.shape)
    assert (records["native"] is None) is not check
    assert record_layout(records["native"]) == record_layout(records["reference"])
    assert_meets_reference(runs["native"], runs["reference"])
    if not _one_thread():
        pytest.skip(
            "the runs and records are repeated bit for bit on one thread "
            "(SOLPROP_THREADS=1), where the reductions' order is fixed"
        )
    assert_same_run(runs["again"], runs["native"])
    assert records["again"] == records["native"]
    for got, expected in zip(phases["again"], phases["native"], strict=True):
        assert got.tobytes() == expected.tobytes()


def _assert_phases_meet(
    phases: list[np.ndarray], reference: list[np.ndarray], shape: tuple[int, ...]
) -> None:
    """Assert a native run's noise phases are the reference's, to the plane budget.

    Parameters
    ----------
    phases : list of numpy.ndarray
        The native session's phase at each plane.
    reference : list of numpy.ndarray
        The reference session's.
    shape : tuple of int
        The field's shape, each phase's.
    """
    assert len(phases) == len(reference) > 1
    for got, expected in zip(phases, reference, strict=True):
        assert got.shape == expected.shape == shape
        np.testing.assert_allclose(got, expected, rtol=0, atol=PLANE_BOUND)


def _walkoff(
    noise: Literal["input", "modified"], integrator: Any, seed: int = 3, **settings: Any
) -> Any:
    """Return a noisy linear walk-off run, whose propagator phase is a z table.

    `settings` replace any of its settings (``Simulation.from_gnlse``).
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
    return Simulation.from_gnlse(
        0.1, WALKOFF_LENGTH, fibre=linear_walkoff(grid), shock=False, n_save=4,
        noise=noise, seed=seed, backend="native", integrator=integrator,
        **(GRID | PULSE | settings),
    )  # fmt: skip


@pytest.mark.parametrize(
    ("noise", "march"),
    [
        ("input", "adams"),
        ("input", "runge-kutta"),
        ("modified", "adams"),
        pytest.param("modified", "runge-kutta", marks=pytest.mark.medium),
    ],
    ids=lambda value: value,
)
def test_a_noisy_run_on_a_z_table_meets_the_reference(noise, march):
    """The noise phase at a plane reads a z table's propagator phase, which the leased
    state holds: the native session reads its slot at the plane through the session,
    under its lock, as the backend reads it from its state, and the door serves the
    run natively. The reference's phase at every plane and its run, within their
    budgets, and on one thread a repeated run bit for bit.
    """
    from solprop.stepper import RungeKutta  # noqa: PLC0415 - beside its one use

    integrator = Multistep() if march == "adams" else RungeKutta()
    simulation = _walkoff(noise, integrator)
    problem = simulation.translate()
    assert problem.linear.phi_representation == "table"
    assert problem.initial.noise_model == noise
    request = simulation._run_request()
    skip_device_refused(problem)
    assert native_session.refusal(problem, request) is None
    context = {"grid": simulation.grid, "modes": simulation.modes}
    kinds = ("reference", "native", "again")
    phases: dict[str, list[np.ndarray]] = {kind: [] for kind in kinds}
    runs = {}
    for kind in kinds:
        name = "numpy_ref" if kind == "reference" else "native"
        backend = select(name, problem, request, **context)
        start = Start(field=problem.initial.field0, z=float(problem.initial.z))
        session: Any = start_session(backend, problem, request, start)
        expected = HostSession if kind == "reference" else native_session.NativeSession
        assert isinstance(session, expected)

        def phase(one, batch, found=phases[kind]) -> None:
            for event in batch.events:
                if isinstance(event, ObservationReady):
                    found.append(np.array(one.noise_phase(event.ticket)))

        try:
            runs[kind] = drive(Started(session), export=IDENTITY, between=phase)
        finally:
            session.close()
            backend.close()
    assert len(phases["native"]) == 4
    _assert_phases_meet(phases["native"], phases["reference"], problem.shape)
    assert np.any(phases["native"][-1] != phases["native"][0])
    assert_meets_reference(runs["native"], runs["reference"])
    if not _one_thread():
        pytest.skip(
            "the runs are repeated bit for bit on one thread (SOLPROP_THREADS=1), "
            "where the reductions' order is fixed"
        )
    assert_same_run(runs["again"], runs["native"])
    for got, expected in zip(phases["again"], phases["native"], strict=True):
        assert got.tobytes() == expected.tobytes()


def test_a_z_table_s_noise_phase_needs_a_pending_ticket():
    """The slot is read under the session's lock for the pending batch alone: an
    acknowledged ticket has expired, as for an export. A position outside the table's
    span is the backend's error, as the backend's own read raises it.
    """
    simulation = _walkoff("modified", Multistep())
    problem = simulation.translate()
    request = simulation._run_request()
    skip_device_refused(problem)
    context = {"grid": simulation.grid, "modes": simulation.modes}
    start = Start(field=problem.initial.field0, z=float(problem.initial.z))
    session = native_session.NativeSession(problem, request, start, **context)
    try:
        batch = session.advance()
        ticket = next(
            event.ticket
            for event in batch.events
            if isinstance(event, ObservationReady)
        )
        assert session.noise_phase(ticket) is not None
        outside = dataclasses.replace(ticket, z=1e3)
        with pytest.raises(BackendError):
            session.noise_phase(outside)
        assert session.noise_phase(ticket) is not None
        session.acknowledge(batch.receipt)
        with pytest.raises(SessionError, match="expired"):
            session.noise_phase(ticket)
    finally:
        session.close()


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
    problem, run_request = _case(integrator=Multistep(), step_planes=0)
    _untranslatable(monkeypatch)
    session = _native(problem, run_request, _entrance(problem))
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
    problem, run_request = _case(
        integrator=Multistep(),
        step_planes=0,
        dz_init=LENGTH / 400,
        dz_max=LENGTH / 400,
    )
    _armed(monkeypatch, "trial", 5, "sigterm")
    _untranslatable(monkeypatch)
    session = _native(problem, run_request, _entrance(problem))
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
def test_production_selects_the_session_and_a_field_restart_meets_the_reference(
    case,
):
    """Production selects the native session for low-storage and dense runs, and a
    field restart from a saved plane (no history: the low-storage march keeps none,
    and a dense plane's march stands ahead of it) continues as the reference's session
    continues it from the same field, within the budgets of
    :func:`~sessions.test_native_session.assert_meets_reference`, and repeats itself
    bitwise on one thread.
    """
    problem, run_request = _case(step_planes=0, **RESTARTED[case])
    request = run_request
    skip_device_refused(problem)
    backend = select(
        "native", problem, run_request, grid=window(), modes=[fibre().mode]
    )
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

    driver = select(
        "numpy_ref", problem, run_request, grid=window(), modes=[fibre().mode]
    )
    baseline = drive(
        Started(HostSession(driver, problem, request, resumed())), export=IDENTITY
    )
    run = drive(Started(_native(problem, request, resumed())), export=IDENTITY)
    assert len(run.observations) == len(baseline.observations) == len(PLANES) - 2
    assert_meets_reference(run, baseline)
    if _one_thread():
        again = drive(Started(_native(problem, request, resumed())), export=IDENTITY)
        assert_same_run(again, run)
