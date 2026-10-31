# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preserve logical Adams budgets across every CPU backend's saved-run resume."""

import dataclasses

import numpy as np
import pytest

from solprop.backends.multistep import CallLedger
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
    simulation = tiny_adams(backend_name)
    problem = simulation.translate()
    backend = backend_class(problem, grid=simulation.grid, modes=simulation.modes)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(1e-6)
        field = backend.observe().require_field().copy()
        saved = backend.metadata()["response_calls"]
        assert isinstance(saved, dict)
        before = CallLedger.from_record(saved)
        assert before.total == 2 and before.startup == 1
    finally:
        backend.close()
    resumed = backend_class(problem, grid=simulation.grid, modes=simulation.modes)
    try:
        resumed.set_initial(field, None, z=1e-6)
        resumed.restore_call_ledgers(saved)
        # The saved metadata must be independently owned by the resumed march.
        saved["orders"].append(5)
        with pytest.raises(ResponseBudgetError, match="budget of 4") as caught:
            resumed.propagate_to(1e-5)
        record = resumed.metadata()["response_calls"]
        assert isinstance(record, dict)
        after = CallLedger.from_record(record)
        assert caught.value.calls == after.total == 4
        assert after.startup == 2
        assert after.accepted_steps == before.accepted_steps + 1
        assert after.orders == [*before.orders, 1]
        assert caught.value.z > 1e-6
    finally:
        resumed.close()


@pytest.mark.parametrize("kind", ["missing", "negative", "boolean"])
def test_adams_restore_rejects_invalid_logical_counts(
    backend_name, backend_class, kind
):
    simulation = tiny_adams(backend_name)
    backend = backend_class(
        simulation.translate(), grid=simulation.grid, modes=simulation.modes
    )
    valid = dataclasses.asdict(CallLedger())
    invalid = dict(valid)
    if kind == "missing":
        invalid.pop("startup")
    else:
        invalid["startup"] = -1 if kind == "negative" else True
    try:
        with pytest.raises(BackendError, match="logical call"):
            backend.restore_call_ledgers(invalid)
        assert backend.metadata()["response_calls"] == valid
    finally:
        backend.close()


def test_compiled_adams_refuses_atomic_ledger_without_changing_logical_counts(
    equivalence_backend_class,
):
    simulation = tiny_adams("numpy_ref")
    backend = equivalence_backend_class(
        simulation.translate(), grid=simulation.grid, modes=simulation.modes
    )
    try:
        with pytest.raises(BackendError, match="does not restore atomic"):
            backend.restore_call_ledgers(
                dataclasses.asdict(CallLedger(startup=2)), {"attempted": 1}
            )
        assert backend.metadata()["response_calls"] == dataclasses.asdict(CallLedger())
    finally:
        backend.close()
