# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preserve logical Adams budgets across every CPU backend's saved-run resume."""

import dataclasses
from typing import cast

import numpy as np
import pytest

from sessions.runs import Records
from sessions.runs import run as session_run
from solprop.backends.ledger import CallLedger
from solprop.backends.session import Start
from solprop.backends.sessions import start_session
from solprop.driver import Simulation
from solprop.exceptions import BackendError, ResponseBudgetError
from solprop.output import resume
from solprop.stepper.multistep import Multistep


def tiny_adams(backend):
    """Build a provider-free linear problem that exceeds its four-call budget."""
    return Simulation.from_gnlse(
        0.0,
        1e-5,
        [0.0, 0.0, -1e-26],
        wavelength=800e-9,
        wavelength_limits=(500e-9, 1.6e-6),
        trange=200e-15,
        fwhm=20e-15,
        peak_power=1.0,
        n_save=2,
        backend=backend,
        integrator=Multistep(max_calls=4),
        dz_init=1e-6,
        dz_max=1e-6,
        noise=None,
    )


@pytest.mark.medium
def test_saved_adams_resume_preserves_exhausted_call_budget(backend_name, tmp_path):
    simulation = tiny_adams(backend_name)
    path = tmp_path / "stopped.h5"
    with pytest.raises(ResponseBudgetError, match="budget of 4"):
        simulation.run(path, progress="none")
    stopped = resume(path)
    assert stopped.outcome is not None
    assert stopped.outcome.status == "budget_exhausted"
    assert CallLedger.from_record(stopped.outcome.logical_calls).total == 4
    assert stopped.checkpoint is not None
    assert 0.0 < stopped.plane.z < 1e-5
    with pytest.raises(ResponseBudgetError, match="budget of 4"):
        simulation.run(path, resume=True, progress="none")
    again = resume(path)
    assert again.outcome is not None
    assert again.outcome.logical_calls == stopped.outcome.logical_calls
    assert again.plane.z == stopped.plane.z
    assert again.history.n_steps == stopped.history.n_steps
    # Structural: the exhausted budget pays for no step, so the resumed record keeps
    # the stored plane, losslessly.
    np.testing.assert_array_equal(again.plane.field, stopped.plane.field)


def test_restored_adams_ledger_charges_fresh_startup(backend_name, backend_class):
    # The run lands on the first step's end, 1e-6, as a plane of its own, and resumes
    # there with the ledger but no method history: the resumed march starts afresh.
    simulation = tiny_adams(backend_name)
    problem = simulation.translate()
    run_request = simulation._run_request()
    run_request = run_request.with_stepper(
        dataclasses.replace(run_request.stepper, save_planes=np.array([1e-6, 1e-5]))
    )
    backend = backend_class(
        problem, run_request, grid=simulation.grid, modes=simulation.modes
    )
    records = Records()
    first = session_run(
        backend, problem, run_request, owns=True, until=1e-6, between=records
    )
    plane = first.final
    assert plane.z == 1e-6
    plane_record = records.taken[-1][1]
    assert plane_record is not None
    saved = plane_record["response_calls"]
    assert isinstance(saved, dict)
    before = CallLedger.from_record(saved)
    assert before.total == 2 and before.startup == 1
    ledgers = {"logical": saved}
    start = Start(
        field=plane.field,
        z=1e-6,
        plane_index=1,
        plane_pending=False,
        resume=True,
        ledgers=ledgers,
    )
    resumed = backend_class(
        problem, run_request, grid=simulation.grid, modes=simulation.modes
    )
    session = start_session(resumed, problem, run_request, start)
    try:
        # The saved record must be independently owned by the resumed march.
        cast("list[list[int]]", saved["orders"]).append([5, 1])
        with pytest.raises(ResponseBudgetError, match="budget of 4") as caught:
            finish(session)
        record = session.execution_record()["response_calls"]
    finally:
        session.close()
    assert isinstance(record, dict)
    after = CallLedger.from_record(record)
    assert caught.value.calls == after.total == 4
    assert after.startup == 2
    assert after.accepted_steps == before.accepted_steps + 1
    assert after.orders == [*before.orders, 1]
    assert caught.value.z > 1e-6


def finish(session):
    """Advance a started session to its end; raise a failed run's error."""
    while not session.finished:
        batch = session.advance()
        session.acknowledge(batch.receipt)
        terminal = batch.terminal
        if terminal is not None and terminal.error is not None:
            raise terminal.error


@pytest.mark.parametrize("kind", ["missing", "negative", "boolean"])
def test_adams_restore_rejects_invalid_logical_counts(
    backend_name, backend_class, kind
):
    simulation = tiny_adams(backend_name)
    valid = CallLedger().to_record()
    invalid = dict(valid)
    if kind == "missing":
        invalid.pop("startup")
    else:
        invalid["startup"] = -1 if kind == "negative" else True
    refused_then_fresh(simulation, backend_class, {"logical": invalid}, "logical call")


def test_compiled_adams_refuses_atomic_ledger_without_changing_logical_counts(
    equivalence_backend_class,
):
    simulation = tiny_adams("numpy_ref")
    ledgers = {"logical": CallLedger(startup=2).to_record(), "atomic": {"attempted": 1}}
    refused_then_fresh(
        simulation, equivalence_backend_class, ledgers, "does not restore atomic"
    )


def refused_then_fresh(simulation, backend_class, ledgers, match):
    """A resume carrying `ledgers` is refused at its start, matching `match`, and
    leaves the backend's next run with a fresh ledger.

    The refusal acts on the backend's own ledger, which its metadata reports: a
    restore that took part of the ledgers before refusing the rest would show there.
    """
    problem = simulation.translate()
    run_request = simulation._run_request()
    backend = backend_class(
        problem, run_request, grid=simulation.grid, modes=simulation.modes
    )
    try:
        start = Start(
            field=problem.initial.field0,
            z=float(problem.initial.z),
            ledgers=ledgers,
        )
        with pytest.raises(BackendError, match=match):
            session_run(backend, problem, run_request, start, owns=False)
        assert backend.metadata()["response_calls"] == CallLedger().to_record()
        launch_records = Records()
        fresh = session_run(
            backend, problem, run_request, owns=False, until=0.0, between=launch_records
        )
    finally:
        backend.close()
    launch_record = launch_records.taken[0][1]
    assert fresh.launch is not None and launch_record is not None
    assert launch_record["response_calls"] == CallLedger().to_record()
