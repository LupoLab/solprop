# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check transactional Adams/atomic budgets through a manufactured host binding."""

import dataclasses
import itertools
import json
from typing import cast

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from backends._session_probe_b2 import run_session
from sessions.executors import IDENTITY, Started, drive
from solprop.backends import registry
from solprop.backends.multistep import CallLedger
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.session import Start
from solprop.backends.sessions import start_session
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
from solprop.translate.routes import tdse_descriptor
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
        wavelength=800e-9,
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
    return simulation.translate(), simulation._run_request(), simulation.grid


def opened_backend(case, limit, *, fail_at=None):
    problem, run_request, grid = case
    assert tdse_descriptor(problem.route) is not None
    binding = ManufacturedBinding(
        tdse_descriptor(problem.route), limit, fail_at=fail_at
    )
    backend = NumpyRefBackend(problem, run_request, grid=grid, tdse_binding=binding)
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


@pytest.mark.parametrize("backend_name", ["native"])
def test_registry_refuses_tdse_before_constructing_a_backend(
    march_problem, monkeypatch, backend_name
):
    """An unsupported explicit backend never silently redirects the atomic problem."""
    problem, run_request, grid = march_problem

    def forbidden(*args, **kwargs):
        raise AssertionError("no backend is constructed for an unsupported TDSE route")

    monkeypatch.setattr(registry, "_construct", forbidden)
    with pytest.raises(BackendError, match=f"{backend_name} does not execute TDSE"):
        registry.select(backend_name, problem, run_request, grid=grid)


def test_direct_native_construction_refuses_tdse_before_allocating(march_problem):
    """The direct constructor enforces the same named refusal as the registry."""
    native = pytest.importorskip("solprop_native")
    problem, run_request, grid = march_problem
    with pytest.raises(BackendError, match="native does not execute TDSE"):
        native.NativeBackend(problem, run_request, grid=grid)


def test_tdse_problem_requires_a_binding_even_at_the_direct_backend(march_problem):
    """A valid descriptor cannot create a hidden native runtime."""
    problem, run_request, grid = march_problem
    with pytest.raises(BackendError, match="explicit host response binding"):
        NumpyRefBackend(problem, run_request, grid=grid)


def test_a_binding_cannot_be_attached_to_an_ordinary_problem(
    march_problem, narrow_grid, linear_problem
):
    """The response ownership is explicit in both directions of the seam."""
    binding = ManufacturedBinding(tdse_descriptor(march_problem[0].route), 4)
    problem, run_request = linear_problem(narrow_grid, save_planes=[1e-5], dz_init=1e-6)
    with pytest.raises(BackendError, match="explicit host response binding"):
        NumpyRefBackend(problem, run_request, grid=narrow_grid, tdse_binding=binding)
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
    """A Runge-Kutta session keeps no logical ledger and refuses an atomic one.

    The ledgers arrive with the run's start, as a resume hands them to the native
    session (SEP 47 section 5.2).
    """
    native = pytest.importorskip("solprop_native")
    problem, run_request = linear_problem(narrow_grid, save_planes=[1e-5], dz_init=1e-6)
    logical = {"startup": 1} if kind == "logical" else {}
    atomic = {"attempted": 1} if kind == "atomic" else {}
    backend = native.NativeBackend(problem, run_request, grid=narrow_grid)
    try:
        plain = run_session(backend, problem, run_request)
        start = Start(
            field=problem.initial.field0,
            z=float(problem.initial.z),
            ledgers={"logical": logical, "atomic": atomic},
        )
        if kind == "logical":
            # A Runge-Kutta march keeps no logical ledger: the counts stay with the
            # earlier execution as provenance, not installed (SEP 42 section 5.6).
            session = start_session(backend, problem, run_request, start, owns=False)
            try:
                run = drive(Started(session), export=IDENTITY)
                assert "response_calls" not in session.execution_record()
            finally:
                session.close()
            # Nothing installed changes nothing: the plain start's run, bitwise.
            assert run.fields[-1].tobytes() == plain.field.tobytes()
            assert run.accepted_ends().tobytes() == plain.run.accepted_ends().tobytes()
        else:
            with pytest.raises(BackendError, match="does not restore atomic"):
                session = start_session(
                    backend, problem, run_request, start, owns=False
                )
                try:
                    drive(Started(session))
                finally:
                    session.close()
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["logical", "atomic"])
def test_reference_rk_restoration_requires_the_matching_response_path(
    narrow_grid, linear_problem, kind
):
    """An atomic ledger requires its TDSE owner; a logical one is provenance here."""
    problem, run_request = linear_problem(narrow_grid, save_planes=[1e-5], dz_init=1e-6)
    backend = NumpyRefBackend(problem, run_request, grid=narrow_grid)
    try:
        logical = CallLedger().to_record() if kind == "logical" else {}
        atomic = (
            dataclasses.asdict(AtomicCallLedger(4, 0, 0, 0, 0, 0))
            if kind == "atomic"
            else None
        )
        if kind == "logical":
            # Not installed in a Runge-Kutta march (SEP 42 section 5.6).
            backend.restore_call_ledgers(logical, atomic)
            assert backend._adams is None
        else:
            with pytest.raises(BackendError, match="requires a TDSE binding"):
                backend.restore_call_ledgers(logical, atomic)
    finally:
        backend.close()


def test_backend_restores_atomic_charges_without_numerical_history(march_problem):
    """Resume owns a fresh Adams history and keeps every previously charged call."""
    backend, binding = opened_backend(march_problem, 8)
    logical = CallLedger(
        startup=1, predictor=1, accepted_steps=1, orders=[1]
    ).to_record()
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
        cast("list[list[int]]", logical["orders"]).append([3, 1])
        assert backend._adams.ledger.orders == [1]
    finally:
        backend.close()


@pytest.mark.parametrize("change", ["missing", "unknown"])
def test_logical_ledger_schema_mismatch_is_refused(change):
    """An old or unrecognized record cannot silently discard cumulative charges."""
    record = CallLedger().to_record()
    if change == "missing":
        del record["startup"]
    else:
        record["future_counter"] = 1
    with pytest.raises(BackendError, match="fields do not match"):
        CallLedger.from_record(record)


@pytest.mark.parametrize(
    "orders",
    [
        ((1, 1),),
        [1, [1, 1]],
        [[1]],
        [[1, 1, 1]],
        [(1, 1)],
        [[0, 1]],
        [[1, 0]],
        [[-1, 1]],
        [[True, 1]],
        [[1, True]],
        [[1.5, 1]],
    ],
)
def test_logical_ledger_restoration_refuses_invalid_predictor_orders(orders):
    """Persisted orders are [order, count] pairs of positive integers, not booleans."""
    record = CallLedger().to_record()
    record["orders"] = orders
    with pytest.raises(BackendError, match=r"orders must be \[order, count\] pairs"):
        CallLedger.from_record(record)


@pytest.mark.parametrize("orders", [[1], [1, 2, 3, 3, 4, 5, 5, 5], [0], [True, 2]])
def test_a_flat_list_of_orders_is_refused_by_name(orders):
    """An earlier development build of the same output schema wrote one order a step.

    Read as pairs it would be refused for its shape alone, or, were the decoder
    lenient, misread; the refusal says where such a record came from instead.
    """
    record = CallLedger(accepted_steps=len(orders)).to_record()
    record["orders"] = orders
    flat = all(type(one) is int for one in orders)
    match = (
        r"flat form an earlier development build wrote.*rerun the case, or read it "
        r"with that build"
        if flat
        else r"orders must be \[order, count\] pairs"
    )
    with pytest.raises(BackendError, match=match):
        CallLedger.from_record(record)


@pytest.mark.parametrize(
    ("steps", "orders"),
    [(0, [[1, 1]]), (2, [[1, 3]]), (4, [[1, 1], [2, 2]]), (1, [[5, 10**12]])],
)
def test_orders_that_do_not_count_the_accepted_steps_are_refused(steps, orders):
    """One order per accepted step: the counts are checked before they are expanded."""
    record = CallLedger(accepted_steps=steps).to_record()
    record["orders"] = orders
    with pytest.raises(
        BackendError, match=rf"orders count \d+ steps, and the ledger {steps}"
    ):
        CallLedger.from_record(record)


class _CountedSlices(list):
    """A list of orders that keeps the length of every slice taken of it."""

    def __init__(self, values):
        super().__init__(values)
        self.slices: list[int] = []

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if isinstance(key, slice):
            self.slices.append(len(item))
        return item


def test_the_record_encodes_only_the_orders_appended_since_the_last_one():
    """The record is rewritten at every plane; its encoding is extended, not redone."""
    orders = _CountedSlices([1, 2, 3, 3, 4, 5, 5, 5])
    ledger = CallLedger(accepted_steps=len(orders), orders=orders)
    first = ledger.to_record()
    orders.extend([5, 5, 4])
    ledger.accepted_steps += 3
    second = ledger.to_record()
    assert orders.slices == [8, 3]
    assert second["orders"] == [[1, 1], [2, 1], [3, 2], [4, 1], [5, 5], [4, 1]]
    # The records are the ledger's, not views of its encoding.
    cast("list[list[int]]", first["orders"])[-1][1] = 99
    assert ledger.to_record()["orders"] == second["orders"]


@pytest.mark.parametrize("change", ["replaced", "truncated", "rewritten"])
def test_the_record_is_encoded_afresh_when_the_orders_are_not_extended(change):
    """A load or resume replaces the orders; anything but an extension starts over."""
    ledger = CallLedger(accepted_steps=5, orders=[1, 2, 3, 3, 3])
    ledger.to_record()
    if change == "replaced":
        ledger.orders = [4, 4]
    elif change == "truncated":
        del ledger.orders[2:]
        ledger.orders.append(5)
    else:
        del ledger.orders[1:]
        ledger.orders.extend([4, 4, 4, 4])
    ledger.accepted_steps = len(ledger.orders)
    fresh = CallLedger(accepted_steps=len(ledger.orders), orders=list(ledger.orders))
    assert ledger.to_record() == fresh.to_record()


@pytest.mark.parametrize(
    "orders", [[], [1], [1, 1, 1], [1, 2, 3, 4, 5, 5, 5, 4, 5], [3, 1, 3]]
)
def test_the_ledger_record_run_length_encodes_its_orders_losslessly(orders):
    """The record holds one pair per run of equal orders and decodes step by step."""
    ledger = CallLedger(accepted_steps=len(orders), orders=list(orders))
    record = ledger.to_record()
    runs = cast("list[list[int]]", record["orders"])
    assert sum(count for _, count in runs) == len(orders)
    assert all(a[0] != b[0] for a, b in itertools.pairwise(runs))
    restored = CallLedger.from_record(json.loads(json.dumps(record)))
    assert restored == ledger
    assert restored.orders is not ledger.orders


@pytest.mark.parametrize("count", [-1, True, 0.5])
def test_logical_ledger_restoration_refuses_invalid_charged_counts(count):
    """A persisted charge is a nonnegative integer count."""
    record = CallLedger().to_record()
    record["startup"] = count
    with pytest.raises(BackendError, match="invalid logical call counter startup"):
        CallLedger.from_record(record)
