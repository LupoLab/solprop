# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check transactional Adams/atomic budgets through a manufactured host binding."""

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from solprop.backends import registry
from solprop.backends.multistep import CallLedger
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import Simulation
from solprop.exceptions import BackendError, ResponseBudgetError
from solprop.nonlinear.tdse import (
    TDSEBasis,
    TDSECalibration,
    TDSEExecution,
    TDSEGasResponse,
    TDSESampling,
)
from solprop.stats import StatsSpec
from solprop.stepper.multistep import Multistep
from solprop.translate.tdse import (
    AtomicCallBudget,
    AtomicCallLedger,
    TDSEResult,
    waveform_identity,
)


class ManufacturedBinding:
    """Return exact zero nonlinearity while charging two radial chunks per call."""

    def __init__(self, descriptor, limit, *, fail_at=None):
        self.descriptor = descriptor
        self.budget = AtomicCallBudget(limit, 2)
        self.fail_at = fail_at
        self.purposes = []
        self.closed = False

    @property
    def ledger(self):
        return self.budget.snapshot()

    def reserve(self, logical_calls):
        self.budget.reserve(logical_calls, z=0.0)

    def release_reservation(self):
        self.budget.release_reservation()

    def restore_ledger(self, ledger):
        self.budget.restore(ledger)

    def record_trial(self, *, accepted):
        del accepted

    def evaluate(self, fields, *, z, purpose, density=None):
        self.purposes.append(purpose)
        self.budget.begin_evaluation(purpose=purpose)
        for _ in range(2):
            self.budget.charge(purpose=purpose, z=z)
            if self.ledger.attempted == self.fail_at:
                self.budget.record_failure()
                raise BackendError("manufactured second-chunk failure")
        return TDSEResult(
            np.zeros_like(fields),
            np.zeros_like(fields),
            input_identity=waveform_identity(fields),
            configuration_identity=self.descriptor.identity,
            valid=np.ones(fields.shape, dtype=bool),
            shadow_orders=(1,),
        )

    def metadata(self):
        return {"kind": "manufactured", "closed": self.closed}

    def memory_ledger(self):
        return {}

    def close(self):
        self.closed = True


@pytest.fixture(params=["interaction", "exponential"])
def march_problem(request):
    # SEP 16: both formulations drive the adapter with the same lifecycle.
    parent = Simulation.from_capillary(
        256e-6,
        1e-5,
        "He",
        0.1,
        lambda0=800e-9,
        wavelength_limits=(400e-9, 1.6e-6),
        trange=60e-15,
        fwhm=2e-15,
        energy=1e-12,
        modes=1,
        n_r=4,
        n_save=2,
        noise=None,
        plasma=False,
        backend="numpy_ref",
        integrator=Multistep(formulation=request.param),
        dz_init=1e-6,
        dz_max=1e-6,
    )
    basis = TDSEBasis(
        "manufactured", (0.0, 1e-9), (8,), 5, 1e-16, source="budget fixture"
    )
    calibration = TDSECalibration(
        "budget-fixture",
        "research",
        basis.identity,
        (400e-9, 2e-6),
        source="budget fixture",
        limitations=("no atomic calculation",),
    )
    response = TDSEGasResponse(
        "helium", basis, 1e-19, "tdse_native", 1.0, calibration, TDSESampling()
    )
    simulation = Simulation(
        parent.grid,
        parent.modes,
        [response],
        parent._field0,
        settings=dataclasses.replace(
            parent.settings,
            tdse_execution=TDSEExecution(),
            # The fixture isolates response accounting: an inherited temporal
            # absorber would add a nonzero derivative and possible rejections.
            absorber=None,
            stats=StatsSpec(names=("energy",)),
        ),
        noise=None,
        profiles=parent.profiles,
        frame=parent.frame,
    )
    return simulation.translate(), simulation.grid


def opened_backend(case, limit, *, fail_at=None):
    problem, grid = case
    assert problem.tdse is not None
    binding = ManufacturedBinding(problem.tdse, limit, fail_at=fail_at)
    backend = NumpyRefBackend(problem, grid=grid, tdse_binding=binding)
    backend.set_initial(problem.initial.field0, None, z=problem.initial.z)
    return backend, binding


def test_whole_startup_trial_is_reserved_before_any_atomic_chunk(march_problem):
    backend, binding = opened_backend(march_problem, 3)
    initial = backend.observe().require_field()
    try:
        with pytest.raises(ResponseBudgetError):
            backend.propagate_to(1e-5)
        assert binding.ledger.attempted == 0
        assert binding.ledger.reserved == 0
        assert backend._adams is not None
        assert backend._adams.ledger.total == 0
        assert backend.observe().z == 0.0
        assert_array_equal(backend.observe().require_field(), initial)
    finally:
        backend.close()


@pytest.mark.parametrize("initial_z", [0.0, 4e-6])
def test_atomic_stop_keeps_accepted_state_and_labels_predictor(
    march_problem, initial_z
):
    backend, binding = opened_backend(march_problem, 5)
    backend.set_initial(march_problem[0].initial.field0, None, z=initial_z)
    try:
        with pytest.raises(ResponseBudgetError, match="atomic-call budget") as caught:
            backend.propagate_to(1e-5)
        observation = backend.observe()
        assert observation.z > initial_z
        assert caught.value.z == observation.z
        assert f"z = {observation.z:g} m" in str(caught.value)
        assert caught.value.calls == binding.ledger.attempted
        assert caught.value.__suppress_context__
        assert_array_equal(caught.value.field, observation.field)
        assert binding.ledger.attempted == 4
        assert binding.ledger.reserved == 0
        assert backend._adams is not None
        assert backend._adams.ledger.total == 2
        assert binding.purposes == ["startup", "predictor"]
        metadata = backend.metadata()
        last = metadata["tdse_last_evaluation"]
        assert isinstance(last, dict)
        assert last["purpose"] == "predictor"
        projection = last["projection"]
        assert isinstance(projection, dict)
        assert projection["kind"] == "embedded_same_rows"
        # Optical observation and permitted statistics do not request another
        # expensive atomic evaluation of the corrected accepted field.
        backend.stats(StatsSpec(names=("energy",)))
        assert binding.ledger.attempted == 4
    finally:
        backend.close()


def test_failed_chunk_is_charged_and_unused_trial_reservation_is_released(
    march_problem,
):
    backend, binding = opened_backend(march_problem, 8, fail_at=2)
    initial = backend.observe().require_field()
    try:
        with pytest.raises(BackendError, match="second-chunk failure"):
            backend.propagate_to(1e-5)
        assert binding.ledger.attempted == 2
        assert binding.ledger.failed_calls == 1
        assert binding.ledger.reserved == 0
        assert backend._adams is not None
        assert backend._adams.ledger.total == 1
        assert backend.observe().z == 0.0
        assert_array_equal(backend.observe().require_field(), initial)
    finally:
        backend.close()


@pytest.mark.parametrize("backend_name", ["numba_cpu", "native"])
def test_registry_refuses_tdse_before_constructing_a_backend(
    march_problem, monkeypatch, backend_name
):
    """An unsupported explicit backend never silently redirects the atomic problem."""
    problem, grid = march_problem

    def forbidden(*args, **kwargs):
        raise AssertionError("no backend is constructed for an unsupported TDSE route")

    monkeypatch.setattr(registry, "_construct", forbidden)
    with pytest.raises(BackendError, match=f"{backend_name} does not execute TDSE"):
        registry.select(backend_name, problem, grid=grid)


def test_direct_numba_construction_refuses_tdse_before_allocating(march_problem):
    """The direct constructor enforces the same named refusal as the registry."""
    problem, grid = march_problem
    with pytest.raises(BackendError, match="numba_cpu does not execute TDSE"):
        NumbaCpuBackend(problem, grid=grid)


def test_tdse_problem_requires_a_binding_even_at_the_direct_backend(march_problem):
    """A valid descriptor cannot create a hidden native runtime."""
    problem, grid = march_problem
    with pytest.raises(BackendError, match="explicit host response binding"):
        NumpyRefBackend(problem, grid=grid)


def test_a_binding_cannot_be_attached_to_an_ordinary_problem(
    march_problem, narrow_grid, linear_problem
):
    """The response ownership is explicit in both directions of the seam."""
    binding = ManufacturedBinding(march_problem[0].tdse, 4)
    problem = linear_problem(narrow_grid, save_planes=[1e-5], dz_init=1e-6)
    with pytest.raises(BackendError, match="explicit host response binding"):
        NumpyRefBackend(problem, grid=narrow_grid, tdse_binding=binding)
    assert binding.ledger.attempted == 0


@pytest.mark.parametrize("name", ["electron_density", "peak_ionisation_rate"])
def test_tdse_runtime_statistics_refuse_legacy_rate_observables(march_problem, name):
    """A dynamic request for rate-model statistics cannot trigger another solve."""
    backend, binding = opened_backend(march_problem, 4)
    try:
        with pytest.raises(BackendError, match="does not define legacy statistics"):
            backend.stats(StatsSpec(names=(name,)))
        assert binding.ledger.attempted == 0
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["logical", "atomic"])
def test_shared_rk_driver_refuses_persisted_response_budgets(
    narrow_grid, linear_problem, kind
):
    """A backend without persistent budget accounting cannot silently reset it."""
    problem = linear_problem(narrow_grid, save_planes=[1e-5], dz_init=1e-6)
    backend = NumbaCpuBackend(
        problem, grid=narrow_grid, threads=1, fftw_threads=1, plan_flag="estimate"
    )
    try:
        backend.restore_call_ledgers({}, None)
        logical = {"startup": 1} if kind == "logical" else {}
        atomic = {"attempted": 1} if kind == "atomic" else None
        message = (
            "require an Adams run" if kind == "logical" else "does not restore atomic"
        )
        with pytest.raises(BackendError, match=message):
            backend.restore_call_ledgers(logical, atomic)
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["logical", "atomic"])
def test_reference_rk_restoration_requires_the_matching_response_path(
    narrow_grid, linear_problem, kind
):
    """Valid ledger records still require their Adams or TDSE execution owner."""
    problem = linear_problem(narrow_grid, save_planes=[1e-5], dz_init=1e-6)
    backend = NumpyRefBackend(problem, grid=narrow_grid)
    try:
        logical = dataclasses.asdict(CallLedger()) if kind == "logical" else {}
        atomic = (
            dataclasses.asdict(AtomicCallLedger(4, 0, 0, 0, 0, 0))
            if kind == "atomic"
            else None
        )
        message = (
            "require an Adams run" if kind == "logical" else "requires a TDSE binding"
        )
        with pytest.raises(BackendError, match=message):
            backend.restore_call_ledgers(logical, atomic)
    finally:
        backend.close()


def test_backend_restores_atomic_charges_without_numerical_history(march_problem):
    """Resume owns a fresh Adams history and keeps every previously charged call."""
    backend, binding = opened_backend(march_problem, 8)
    logical = dataclasses.asdict(
        CallLedger(startup=1, predictor=1, accepted_steps=1, orders=[1])
    )
    atomic = dataclasses.asdict(AtomicCallLedger(8, 4, 2, 2, 0, 0))
    initial = backend.observe().require_field().copy()
    try:
        backend.restore_call_ledgers(logical, atomic)
        assert backend.metadata()["response_calls"] == logical
        assert binding.ledger == AtomicCallLedger(8, 4, 0, 2, 0, 0)
        assert backend._adams is not None
        assert backend._adams._history == []
        assert backend._adams._order == 1
        assert_array_equal(backend.observe().require_field(), initial)
        logical["orders"].append(3)
        assert backend._adams.ledger.orders == [1]
    finally:
        backend.close()


@pytest.mark.parametrize("change", ["missing", "unknown"])
def test_logical_ledger_schema_mismatch_is_refused(change):
    """An old or unrecognized record cannot silently discard cumulative charges."""
    record = dataclasses.asdict(CallLedger())
    if change == "missing":
        del record["startup"]
    else:
        record["future_counter"] = 1
    with pytest.raises(BackendError, match="fields do not match"):
        CallLedger.from_record(record)


@pytest.mark.parametrize("orders", [(1,), [0], [-1], [True], [1.5]])
def test_logical_ledger_restoration_refuses_invalid_predictor_orders(orders):
    """Persisted orders must be a list of positive integer orders, never booleans."""
    record = dataclasses.asdict(CallLedger())
    record["orders"] = orders
    with pytest.raises(BackendError, match="orders must be positive integers"):
        CallLedger.from_record(record)


@pytest.mark.parametrize("count", [-1, True, 0.5])
def test_logical_ledger_restoration_refuses_invalid_charged_counts(count):
    """A persisted charge is a nonnegative integer count."""
    record = dataclasses.asdict(CallLedger())
    record["startup"] = count
    with pytest.raises(BackendError, match="invalid logical call counter startup"):
        CallLedger.from_record(record)
