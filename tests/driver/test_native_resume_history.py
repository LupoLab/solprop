# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native backend's resident Adams history crosses a resume (SEP 29 section 6.6).

The engine keeps its history's rows in the native state; it copies them to the host for
a snapshot and back for a resume. The copies are lossless, so a restored march holds
exactly what was stored: its rows, positions, slots, identities, order and controller
state, which these tests compare exactly. Whether the restored history is then *used*
is witnessed separately -- the engine's startup sample, which a restarted history takes,
is never evaluated -- since a restarted history is also correct to the accuracy
contract and an accuracy check alone cannot tell the two apart (SEP 29, review R17).
The propagation after the restore is held to an accuracy envelope, never to the
uninterrupted run's bits or work counts (SEP 25).
"""

import dataclasses
import logging
from typing import Any

import numpy as np
import pytest

from driver.test_resume import LENGTH, N_SAVE, run_until
from driver.test_resume_history import (
    AFRESH,
    RTOL,
    assert_accurate_run,
    coupled,
    soliton,
    tight_run,
    varying,
)
from driver.test_specified_paths import GRID, PULSE, linear_walkoff
from driver.test_specified_paths import LENGTH as MOVING_LENGTH
from march_bounds import TIGHTER
from sessions.executors import IDENTITY, Started, drive, one_thread
from sessions.runs import Snapshots, bare, to_planes
from sessions.runs import run as run_session
from solprop.backends.events import ObservationReady
from solprop.backends.registry import available, select
from solprop.backends.session import Start
from solprop.backends.sessions import start_session
from solprop.driver import Simulation
from solprop.grid import EnvelopeGrid
from solprop.output import open_file
from solprop.output.hdf5 import resume
from solprop.output.schema import MarchSnapshot
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, Multistep, RungeKutta

pytestmark = [
    pytest.mark.skipif("native" not in available(), reason="needs solprop-native"),
    # The fixtures here launch noise-free into time windows kept tight for speed, which
    # hold more than a millionth of the energy in their guards or collar: the boundary
    # monitor warns (SEP 21), and the window's edge is not what this module tests.
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
]


def moving(backend, rtol=RTOL):
    """Return a moving-dispersion fibre: the z table, under a compensated march."""

    def build():
        grid = EnvelopeGrid.from_limits(MOVING_LENGTH, 835e-9, (600e-9, 1300e-9), 2e-12)
        return Simulation.from_gnlse(
            0.1,
            MOVING_LENGTH,
            fibre=linear_walkoff(grid),
            shock=False,
            n_save=4,
            noise=None,
            backend=backend,
            integrator=Multistep(),
            rtol=rtol,
            **GRID,
            **PULSE,
        )

    return build


CASES = {
    "plain": lambda build, backend, rtol=RTOL: soliton(build, backend, rtol=rtol),
    "interaction": lambda build, backend, rtol=RTOL: soliton(
        build, backend, integrator=Multistep(formulation="interaction"), rtol=rtol
    ),
    "compensated": lambda _, backend, rtol=RTOL: varying(backend, rtol),
    "coupled": lambda _, backend, rtol=RTOL: coupled(backend, rtol),
    "z_table": lambda _, backend, rtol=RTOL: moving(backend, rtol),
}
"""Each history the resident engine keeps: the exponential and interaction ones, a
varying operator's compensated one, a frozen-residual coupled flow's, and the
compensated one of an operator on the z table, whose rate the native derivative table
supplies."""

ROWS = {
    "plain": {"store"},
    "interaction": {"store"},
    "compensated": {"store", "states", "rates"},
    "coupled": {"store", "states", "rates", "applied"},
    "z_table": {"store", "states", "rates"},
}
"""The rows each history exports."""


def _on_device() -> bool:
    """Whether the installed native build executes on an accelerator."""
    if "native" not in available():
        return False
    import solprop_native  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    return solprop_native.build_info().device_backend != "none"


CASE_PARAMS = [
    pytest.param(
        name,
        marks=pytest.mark.skipif(
            name == "z_table" and _on_device(),
            reason="device builds refuse a z table by name",
        ),
    )
    for name in sorted(CASES)
]
"""The cases, a z table's skipped on a device build, which refuses it."""

QUIET = (
    # The coupled case's short window and marginal wall, as its parity cases run them.
    pytest.mark.filterwarnings("ignore:the temporal absorber attenuates a packet"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.GrazingIncidenceWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
)


def quiet(test):
    """Apply :data:`QUIET` to `test`."""
    for mark in QUIET:
        test = mark(test)
    return test


def opened(simulation, backend, planes=None):
    """Return the translated problem, a request and `backend` built for it.

    The request observes no statistics (:func:`sessions.runs.bare`); `planes`
    replaces its save planes, a run that ends at the last of them.
    """
    problem = simulation.translate()
    run_request = simulation._run_request()
    if planes is not None:
        run_request = to_planes(run_request, planes)
    built: Any = select(
        backend,
        problem,
        run_request,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    return problem, bare(run_request), built


def resumed_at(stored: MarchSnapshot, request, *, pending: bool = True) -> Start:
    """Return a resume at a snapshot's plane that carries its history.

    With `pending`, the plane at the snapshot is owed, so the session observes it
    before any step and the history it installed can be read there.
    """
    index = int(np.searchsorted(request.observation.save_planes, stored.z, "right"))
    return Start(
        field=np.array(stored.state),
        z=stored.z,
        plane_index=index,
        plane_pending=pending,
        resume=True,
        march=stored,
    )


def started(*args: Any, **keywords: Any) -> Any:
    """Start a run's session (:func:`~solprop.backends.sessions.start_session`).

    Typed loosely: the tests read the session's ``march_refusal``, which the native
    and the reference sessions both state beyond the protocol.
    """
    return start_session(*args, **keywords)


def pending_snapshot(session) -> MarchSnapshot | None:
    """Advance a session to its first observation; return the history it holds there.

    The batch stays pending; the caller closes the session.
    """
    while True:
        batch = session.advance()
        for event in batch.events:
            if isinstance(event, ObservationReady):
                return session.march_snapshot(event.ticket)
        session.acknowledge(batch.receipt)


def startups(session) -> int:
    """Return the startup samples a session's march took, which only a (re)started
    history takes; a resume that carries no ledger counts from zero."""
    record = session.execution_record()["response_calls"]
    assert isinstance(record, dict)
    return int(record["startup"])


def assert_same_snapshot(found: MarchSnapshot | None, stored: MarchSnapshot) -> None:
    """Assert two snapshots hold the same lossless state, exactly."""
    assert found is not None
    assert found.z == stored.z
    assert np.array_equal(found.state, stored.state)
    assert found.proposal == stored.proposal
    assert found.method == stored.method
    assert found.order == stored.order
    assert found.rejections == stored.rejections
    assert found.last_rejected == stored.last_rejected
    assert found.collar_was_quiet == stored.collar_was_quiet
    assert np.array_equal(found.positions, stored.positions)
    assert found.identities == stored.identities
    assert np.array_equal(found.slots, stored.slots)
    assert set(found.rows) == set(stored.rows)
    for name, row in stored.rows.items():
        assert found.rows[name].dtype == row.dtype
        assert np.array_equal(found.rows[name], row), name


@quiet
@pytest.mark.parametrize("case", CASE_PARAMS)
def test_the_resident_history_is_restored_exactly_and_continued(build_simulation, case):
    simulation = CASES[case](build_simulation, "native")()
    problem, request, first = opened(simulation, "native")
    engine = first._resident
    assert type(engine).__name__ == "_ResidentAdams"
    # The engine's own layout keeps a leading axis of one after the row axis.
    shape = (engine.n_slots, 1, *problem.shape)
    whole = Snapshots()
    baseline = run_session(first, problem, request, owns=True, between=whole)
    stored = whole.at_plane(0)
    # Once the history has formed, the snapshot holds it.
    assert stored is not None
    assert len(stored.identities) >= 2
    assert set(stored.rows) == ROWS[case]
    for row in stored.rows.values():
        assert row.shape == shape and row.dtype == np.complex128
    assert np.any(stored.rows["store"][stored.slots] != 0)
    _, _, second = opened(simulation, "native")
    session = started(second, problem, request, resumed_at(stored, request))
    taken = Snapshots()
    try:
        assert session.march_refusal is None
        run = drive(Started(session), export=IDENTITY, between=taken)
        # The restored history was continued: no startup sample was evaluated.
        assert startups(session) == 0
    finally:
        session.close()
    # Installed losslessly: the history at the resumed plane, before any step, is
    # the stored one exactly.
    restored = taken.taken[0][1]
    assert taken.taken[0][0].z == stored.z
    assert_same_snapshot(restored, stored)
    assert all(np.all(np.isfinite(field)) for field in run.fields)
    if not one_thread():
        return
    # On one thread the continued march is the uninterrupted one: each later plane's
    # field and history to the bit.
    later = [
        (event, field, snapshot)
        for (event, field), (_, snapshot) in zip(
            zip(baseline.observations, baseline.fields, strict=True),
            whole.taken,
            strict=True,
        )
        if event.z > stored.z
    ]
    resumed = list(zip(run.observations, run.fields, strict=True))[1:]
    assert [event.z for event, _ in resumed] == [event.z for event, _, _ in later]
    for (_, field), (_, expected, snapshot), (_, found) in zip(
        resumed, later, taken.taken[1:], strict=True
    ):
        assert field.tobytes() == expected.tobytes()
        assert snapshot is not None
        assert_same_snapshot(found, snapshot)


@quiet
@pytest.mark.parametrize("case", ["compensated", "coupled"])
def test_a_resume_without_the_snapshot_restarts_the_history(build_simulation, case):
    # The witness above is a real one: the same run without the snapshot does
    # evaluate the startup sample, once, on its way to the first plane.
    simulation = CASES[case](build_simulation, "native")()
    first = float(simulation._run_request().stepper.save_planes[0])
    problem, request, backend = opened(simulation, "native", planes=[first])
    session = started(backend, problem, request, Start.entrance(problem))
    try:
        drive(Started(session))
        assert startups(session) == 1
    finally:
        session.close()


@quiet
@pytest.mark.medium
@pytest.mark.parametrize("case", CASE_PARAMS)
def test_a_resumed_native_run_is_the_uninterrupted_one(
    build_simulation, tmp_path, case, caplog
):
    build = CASES[case](build_simulation, "native")
    whole, part = tmp_path / "whole.h5", tmp_path / "part.h5"
    build().run(whole, progress="none")
    reference = tight_run(
        CASES[case](build_simulation, "native", rtol=RTOL / TIGHTER),
        tmp_path / "tight.h5",
    )
    run_until(build(), part, after=2)
    state = resume(part)
    assert state.march is not None
    assert set(state.march.rows) == ROWS[case]
    # The resume decides in the simulation; the run's session installs the history.
    with caplog.at_level(logging.INFO, logger="solprop.driver"):
        build().run(part, resume=True, progress="none")
    assert AFRESH not in caplog.text
    assert "continuing the Adams history" in caplog.text
    assert_accurate_run(whole, part, reference)


@pytest.mark.medium
def test_a_native_run_stopped_between_planes_continues_from_its_checkpoint(
    build_simulation, tmp_path, monkeypatch, caplog
):
    # The native march's failure is injected natively at a trial past the first
    # plane (`adams_march_fault`): the run stops at its last accepted step, strictly
    # between two planes, and keeps the march's state beside the terminal checkpoint.
    from driver.test_native_march_lifecycle import (  # noqa: PLC0415 - needs native
        with_fault,
    )

    build = soliton(build_simulation, "native")
    whole, part = tmp_path / "whole.h5", tmp_path / "part.h5"
    build().run(whole, progress="none")
    reference = tight_run(
        soliton(build_simulation, "native", rtol=RTOL / TIGHTER),
        tmp_path / "tight.h5",
    )
    planes = [0.0, *build().settings.save_planes.tolist()]
    with monkeypatch.context() as patch:
        with_fault(patch, "trial", 120, "backend")
        with pytest.raises(RuntimeError, match="an injected native failure"):
            build().run(part, progress="none")
    state = resume(part)
    assert state.checkpoint is not None
    assert state.march is not None and state.march.z == state.checkpoint.plane.z
    stop = state.checkpoint.plane.z
    assert stop > planes[1] and not any(np.isclose(stop, one) for one in planes)
    with caplog.at_level(logging.INFO, logger="solprop.driver"):
        build().run(part, resume=True, progress="none")
    assert AFRESH not in caplog.text
    assert_accurate_run(whole, part, reference)


def assert_frozen_residual_history_refused(build_simulation, taken, resumed, message):
    """Assert the coupled case's snapshot from `taken` is refused on `resumed`."""
    stored = snapshot_at_first_plane(CASES["coupled"](build_simulation, taken)())
    simulation = CASES["coupled"](build_simulation, resumed)()
    problem, request, backend = opened(simulation, resumed)
    session = started(backend, problem, request, resumed_at(stored, request))
    try:
        reason = session.march_refusal
        assert reason is not None
        assert message in reason
        # Nothing was installed: the march still has no history to export.
        assert pending_snapshot(session) is None
    finally:
        session.close()


@quiet
@pytest.mark.parametrize(
    ("taken", "resumed", "message"),
    [
        ("native", "numpy_ref", "native engine's applied frozen-residual"),
        ("numpy_ref", "native", "host engine's frozen-residual weights"),
    ],
)
def test_a_frozen_residual_history_does_not_cross_engines(
    build_simulation, taken, resumed, message
):
    # The native engine keeps each sample's generator applied to its state, the host
    # engine the sample's weights: neither can continue the other's history.
    assert_frozen_residual_history_refused(build_simulation, taken, resumed, message)


def assert_same_rows_cross(build_simulation, taken, resumed, case):
    """Assert a `case` snapshot from `taken` is restored whole on `resumed`."""
    stored = snapshot_at_first_plane(CASES[case](build_simulation, taken)())
    simulation = CASES[case](build_simulation, resumed)()
    problem, request, backend = opened(simulation, resumed)
    session = started(backend, problem, request, resumed_at(stored, request))
    try:
        assert session.march_refusal is None
        assert_same_snapshot(pending_snapshot(session), stored)
    finally:
        session.close()


@quiet
@pytest.mark.parametrize("case", ["plain", "compensated"])
@pytest.mark.parametrize(
    ("taken", "resumed"), [("native", "numpy_ref"), ("numpy_ref", "native")]
)
def test_a_history_of_the_same_rows_crosses_backends(
    build_simulation, taken, resumed, case
):
    # Where the rows mean the same -- the samples, a compensated history's states and
    # rates -- a snapshot continues on the other backend.
    assert_same_rows_cross(build_simulation, taken, resumed, case)


def snapshot_at_first_plane(simulation):
    """Return the march snapshot of `simulation`'s backend at its first save plane.

    Its session runs to that plane alone, which it lands on as the whole run does.
    """
    first = float(simulation._run_request().stepper.save_planes[0])
    problem, request, backend = opened(
        simulation, simulation.settings.backend, planes=[first]
    )
    taken = Snapshots()
    run_session(backend, problem, request, owns=True, between=taken)
    stored = taken.at_plane(0)
    assert stored is not None
    return stored


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda rows: rows.pop("rates"), "lacks ['rates']"),
        (lambda rows: rows.update(applied=rows["store"]), "also holds ['applied']"),
        (lambda rows: rows.update(weights_0=np.ones(3)), "frozen-residual weights"),
        (
            lambda rows: rows.update(store=rows["store"][:-1]),
            "do not fit the native backend's complex store",
        ),
        (
            lambda rows: rows.update(states=rows["states"].real.copy()),
            "do not fit the native backend's complex store",
        ),
    ],
)
def test_an_incompatible_native_history_is_refused(build_simulation, damage, message):
    simulation = CASES["compensated"](build_simulation, "native")()
    stored = snapshot_at_first_plane(simulation)
    rows = dict(stored.rows)
    damage(rows)
    problem, request, backend = opened(simulation, "native")
    damaged = dataclasses.replace(stored, rows=rows)
    session = started(
        backend, problem, request, resumed_at(damaged, request), owns=False
    )
    try:
        reason = session.march_refusal
        assert reason is not None
        assert "native backend" in reason
        assert message in reason
        # Refused before any row was copied: the history is still empty.
        assert pending_snapshot(session) is None
    finally:
        session.close()
    # The intact snapshot is still taken up afterwards, by the same backend.
    session = started(backend, problem, request, resumed_at(stored, request))
    try:
        assert session.march_refusal is None
        assert_same_snapshot(pending_snapshot(session), stored)
    finally:
        session.close()


@quiet
def test_a_snapshot_from_elsewhere_or_another_march_is_not_continued(build_simulation):
    # A native Adams snapshot offered at another position starts the history afresh;
    # offered to a march that keeps no history, it is not continued, and that march
    # has no snapshot of its own to give.
    stored = snapshot_at_first_plane(soliton(build_simulation, "native")())
    problem, request, backend = opened(soliton(build_simulation, "native")(), "native")
    entrance = Start(
        field=problem.initial.field0, z=float(problem.initial.z), march=stored
    )
    session = started(backend, problem, request, entrance)
    try:
        assert "the run resumes at z = 0.0 m" in str(session.march_refusal)
        assert pending_snapshot(session) is None
    finally:
        session.close()
    planes = np.linspace(0.0, LENGTH, N_SAVE)[1:].tolist()
    schedule = FixedSchedule.uniform(0.0, LENGTH, LENGTH / 32, save_planes=planes)
    marches = {
        "Runge-Kutta": {"integrator": RungeKutta()},
        "low-storage": {
            "integrator": LowStorageRungeKutta(),
            "schedule": schedule,
            "dz_max": schedule.max_step,
            "dz_init": schedule.max_step,
            # Without the collar, whose stable step a replay must respect.
            "absorber": False,
        },
    }
    for march, keywords in marches.items():
        other = soliton(build_simulation, "native", **keywords)()
        problem, request, backend = opened(other, "native")
        session = started(backend, problem, request, resumed_at(stored, request))
        try:
            assert session.march_refusal == (
                f"the run steps with the {march} march, which keeps no history"
            )
            assert pending_snapshot(session) is None
        finally:
            session.close()


@pytest.mark.medium
def test_a_native_resume_under_modified_noise_continues_the_same_march(
    tmp_path, monkeypatch
):
    # The modified noise model's realisation lives in the native state beside the
    # field; installing the snapshot's state must reload it too, or the restored march
    # evaluates its sources without the noise (a departure of about 1e-8 here). On a
    # prescribed schedule the resumed run is the uninterrupted discrete march, so it is
    # held to the matched bound the host engines meet, not to a restart's envelope.
    from solprop_native.session import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        NativeSession,
    )

    from driver.test_envelope_plasma import build  # noqa: PLC0415 - shared case
    from march_bounds import simulation_bound  # noqa: PLC0415

    options = {"backend": "native", "noise": "modified", "seed": 821}
    whole = build(**options).run(progress="none")
    path = tmp_path / "envelope.h5"
    run_until(build(**options), path, after=1)
    stored = resume(path).march
    assert stored is not None
    # The snapshot installs losslessly beside the modified noise: a session resumed at
    # its plane, the plane owed, holds exactly the stored history there.
    simulation = build(**options)
    problem, request, backend = opened(simulation, "native")
    session = started(backend, problem, request, resumed_at(stored, request))
    try:
        assert session.march_refusal is None
        assert_same_snapshot(pending_snapshot(session), stored)
    finally:
        session.close()
    # The production resume's session installs it too.
    refusals = []
    original = NativeSession.__init__

    def watched(self, *args, **keywords):
        original(self, *args, **keywords)
        refusals.append(self.march_refusal)

    monkeypatch.setattr(NativeSession, "__init__", watched)
    build(**options).run(path, resume=True, progress="none")
    assert refusals == [None]
    with open_file(path) as file:
        expected = whole.fields()
        scale = float(np.max(np.abs(expected)))
        departure = float(np.max(np.abs(file.fields() - expected))) / scale
    # The power conversion's two roundings (2.7e-16) and the two runs' own rounding
    # budget (SEP 25), as `test_envelope_hdf5_resume_and_reconstruction` declares it.
    assert departure <= 2.7e-16 + simulation_bound(build(**options))
