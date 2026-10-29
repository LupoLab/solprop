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

from driver.test_resume import CrashError, run_until
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
from solprop.backends.registry import available, select
from solprop.driver import Simulation
from solprop.exceptions import SolpropWarning
from solprop.grid import EnvelopeGrid
from solprop.output import open_file
from solprop.output.hdf5 import resume
from solprop.output.schema import MarchSnapshot
from solprop.stepper import Multistep
from solprop.stepper.multistep import ComparisonMultistep

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


def opened(simulation, backend):
    """Return the translated problem and `backend` built for it, at the entrance."""
    problem = simulation.translate()
    # A step driver that is also a closable backend: no one static type names both.
    built: Any = select(
        backend,
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    built.set_initial(problem.initial.field0, None, z=problem.initial.z)
    return problem, built


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


def counted_starts(backend):
    """Count the startup samples, which only a (re)started history takes.

    The march charges each to its ledger before the engine evaluates it, so the count
    holds on the native march, which calls its engine natively, as on the host one; a
    restored history does not carry the ledger, so the count starts here.
    """
    before = backend._adams.ledger.startup
    return lambda: backend._adams.ledger.startup - before


@quiet
@pytest.mark.parametrize("case", CASE_PARAMS)
def test_the_resident_history_is_restored_exactly_and_continued(build_simulation, case):
    simulation = CASES[case](build_simulation, "native")()
    problem, first = opened(simulation, "native")
    planes = [float(z) for z in problem.stepper.save_planes]
    try:
        engine = first._adams._engine
        assert type(engine).__name__ == "_ResidentAdams"
        shape = (engine.n_slots, *problem.shape)
        first.propagate_to(planes[0])
        stored = first.march_snapshot()
    finally:
        first.close()
    # Once the history has formed, the snapshot holds it.
    assert stored is not None
    assert len(stored.identities) >= 2
    assert set(stored.rows) == ROWS[case]
    for row in stored.rows.values():
        assert row.shape == shape and row.dtype == np.complex128
    assert np.any(stored.rows["store"][stored.slots] != 0)
    _, second = opened(simulation, "native")
    try:
        second.set_initial(problem.initial.field0, None, z=stored.z)
        starts = counted_starts(second)
        assert second.restore_march(stored) is None
        assert_same_snapshot(second.march_snapshot(), stored)
        second.propagate_to(planes[1])
        # The restored history was continued: no startup sample was evaluated.
        assert starts() == 0
        assert np.all(np.isfinite(second._adams_state()))
    finally:
        second.close()


@quiet
# The host engine runs solprop's march, with the warning the admission tests assert.
@pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning")
@pytest.mark.parametrize("case", CASE_PARAMS)
def test_the_host_engine_on_native_hooks_round_trips_too(
    build_simulation, monkeypatch, case
):
    # Where the resident engine does not apply (a block of more than sixteen channels,
    # the untransported witness), the native backend runs solprop's host engine over
    # its hooks; that history crosses a resume in the host engine's own rows.
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        NativeBackend,
    )

    monkeypatch.setattr(NativeBackend, "_adams_engine", lambda *_: None)
    simulation = CASES[case](build_simulation, "native")()
    problem, first = opened(simulation, "native")
    try:
        assert type(first._adams._engine).__name__ == "HostHistory"
        first.propagate_to(float(problem.stepper.save_planes[0]))
        stored = first.march_snapshot()
    finally:
        first.close()
    assert stored is not None
    weights = {name for name in stored.rows if name.startswith("weights_")}
    assert set(stored.rows) - weights == ROWS[case] - {"applied"}
    assert bool(weights) == (case == "coupled")
    _, second = opened(simulation, "native")
    try:
        second.set_initial(problem.initial.field0, None, z=stored.z)
        starts = counted_starts(second)
        assert second.restore_march(stored) is None
        assert_same_snapshot(second.march_snapshot(), stored)
        second.propagate_to(float(problem.stepper.save_planes[1]))
        assert starts() == 0
    finally:
        second.close()


@quiet
@pytest.mark.parametrize("case", ["compensated", "coupled"])
def test_a_resume_without_the_snapshot_restarts_the_history(build_simulation, case):
    # The witness above is a real one: the same backend without the snapshot does
    # evaluate the startup sample.
    simulation = CASES[case](build_simulation, "native")()
    problem, backend = opened(simulation, "native")
    try:
        starts = counted_starts(backend)
        backend.propagate_to(float(problem.stepper.save_planes[0]))
        assert starts() == 1
    finally:
        backend.close()


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
    with caplog.at_level(logging.INFO, logger="solprop.driver.simulation"):
        build().run(part, resume=True, progress="none")
    assert AFRESH not in caplog.text
    assert "continuing the Adams history" in caplog.text
    assert_accurate_run(whole, part, reference)


@pytest.mark.medium
def test_a_native_run_stopped_between_planes_continues_from_its_checkpoint(
    build_simulation, tmp_path, monkeypatch, caplog
):
    # The resident engine evaluates its sources itself, so the failure is injected at
    # its trial, which only the host march calls from Python: the run stops at its last
    # accepted step, strictly between two planes past the first, and keeps the march's
    # state beside the terminal checkpoint. The native march's failures are injected
    # natively (tests/backends/test_native_march_lifecycle.py).
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        backend,
    )
    from solprop_native.backend import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        _ResidentAdams,
    )

    monkeypatch.setattr(backend, "_ADAMS_MARCH", "host")

    build = soliton(build_simulation, "native")
    whole, part = tmp_path / "whole.h5", tmp_path / "part.h5"
    build().run(whole, progress="none")
    reference = tight_run(
        soliton(build_simulation, "native", rtol=RTOL / TIGHTER),
        tmp_path / "tight.h5",
    )
    planes = [0.0, *build().settings.save_planes.tolist()]
    original = _ResidentAdams.exponential_trial

    def failing(self, z, *args, **keywords):
        if z > planes[1] and not any(np.isclose(z, one) for one in planes):
            raise CrashError("the source failed")
        return original(self, z, *args, **keywords)

    with monkeypatch.context() as patch:
        patch.setattr(_ResidentAdams, "exponential_trial", failing)
        with pytest.raises(CrashError):
            build().run(part, progress="none")
    state = resume(part)
    assert state.checkpoint is not None
    assert state.march is not None and state.march.z == state.checkpoint.plane.z
    with caplog.at_level(logging.INFO, logger="solprop.driver.simulation"):
        build().run(part, resume=True, progress="none")
    assert AFRESH not in caplog.text
    assert_accurate_run(whole, part, reference)


@quiet
@pytest.mark.parametrize(
    ("taken", "resumed", "message"),
    [
        ("native", "numba_cpu", "native engine's applied frozen-residual"),
        ("numba_cpu", "native", "host engine's frozen-residual weights"),
    ],
)
def test_a_frozen_residual_history_does_not_cross_engines(
    build_simulation, taken, resumed, message
):
    # The native engine keeps each sample's generator applied to its state, the host
    # engine the sample's weights: neither can continue the other's history.
    stored = snapshot_at_first_plane(CASES["coupled"](build_simulation, taken)())
    problem, backend = opened(CASES["coupled"](build_simulation, resumed)(), resumed)
    try:
        backend.set_initial(problem.initial.field0, None, z=stored.z)
        reason = backend.restore_march(stored)
        assert reason is not None
        assert message in reason
        # Nothing was installed: the march still has no history to export.
        assert backend.march_snapshot() is None
    finally:
        backend.close()


@quiet
@pytest.mark.parametrize("case", ["plain", "compensated"])
@pytest.mark.parametrize(
    ("taken", "resumed"), [("native", "numba_cpu"), ("numba_cpu", "native")]
)
def test_a_history_of_the_same_rows_crosses_backends(
    build_simulation, taken, resumed, case
):
    # Where the rows mean the same -- the samples, a compensated history's states and
    # rates -- a snapshot continues on the other backend.
    stored = snapshot_at_first_plane(CASES[case](build_simulation, taken)())
    problem, backend = opened(CASES[case](build_simulation, resumed)(), resumed)
    try:
        backend.set_initial(problem.initial.field0, None, z=stored.z)
        assert backend.restore_march(stored) is None
        assert_same_snapshot(backend.march_snapshot(), stored)
    finally:
        backend.close()


def snapshot_at_first_plane(simulation):
    """Return the march snapshot of `simulation`'s backend at its first save plane."""
    problem, backend = opened(simulation, simulation.settings.backend)
    try:
        backend.propagate_to(float(problem.stepper.save_planes[0]))
        stored = backend.march_snapshot()
    finally:
        backend.close()
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
    problem, backend = opened(simulation, "native")
    try:
        backend.set_initial(problem.initial.field0, None, z=stored.z)
        reason = backend.restore_march(dataclasses.replace(stored, rows=rows))
        assert reason is not None
        assert "native backend" in reason
        assert message in reason
        # Refused before any row was copied: the history is still empty.
        assert backend.march_snapshot() is None
        # The intact snapshot is still taken up afterwards.
        assert backend.restore_march(stored) is None
    finally:
        backend.close()


@pytest.mark.medium
def test_a_march_that_cannot_carry_its_history_warns_on_resume(
    build_simulation, tmp_path
):
    # The general path of a comparison method keeps host arrays no snapshot holds:
    # its resume restarts the history, once, with a warning that says why.
    build = soliton(
        build_simulation, "numpy_ref", integrator=ComparisonMultistep(order=3)
    )
    path = tmp_path / "run.h5"
    run_until(build(), path, after=2)
    assert resume(path).march is None
    with pytest.warns(SolpropWarning, match="general Adams path") as caught:
        build().run(path, resume=True, progress="none")
    assert len([one for one in caught if AFRESH in str(one.message)]) == 1


@pytest.mark.medium
def test_a_native_resume_under_modified_noise_continues_the_same_march(
    tmp_path, monkeypatch
):
    # The modified noise model's realisation lives in the native state beside the
    # field; installing the snapshot's state must reload it too, or the restored march
    # evaluates its sources without the noise (a departure of about 1e-8 here). On a
    # prescribed schedule the resumed run is the uninterrupted discrete march, so it is
    # held to the matched bound the host engines meet, not to a restart's envelope.
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        NativeBackend,
    )

    from driver.test_envelope_plasma import build  # noqa: PLC0415 - shared case
    from march_bounds import simulation_bound  # noqa: PLC0415

    options = {"backend": "native", "noise": "modified", "seed": 821}
    whole = build(**options).run(progress="none")
    path = tmp_path / "envelope.h5"
    run_until(build(**options), path, after=1)
    stored = resume(path).march
    assert stored is not None
    restored = []
    original = NativeBackend.restore_march

    def watched(self, snapshot):
        reason = original(self, snapshot)
        restored.append((reason, self.march_snapshot()))
        return reason

    monkeypatch.setattr(NativeBackend, "restore_march", watched)
    build(**options).run(path, resume=True, progress="none")
    [(reason, found)] = restored
    assert reason is None
    assert_same_snapshot(found, stored)
    with open_file(path) as file:
        expected = whole.fields()
        scale = float(np.max(np.abs(expected)))
        departure = float(np.max(np.abs(file.fields() - expected))) / scale
    # The power conversion's two roundings (2.7e-16) and the two runs' own rounding
    # budget (SEP 25), as `test_envelope_hdf5_resume_and_reconstruction` declares it.
    assert departure <= 2.7e-16 + simulation_bound(build(**options))
