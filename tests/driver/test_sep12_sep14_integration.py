# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep atomic budgets and checkpoints intact across the shared Adams march."""

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import Simulation
from solprop.driver import tdse as runtime_module
from solprop.exceptions import (
    BackendError,
    DriverError,
    ResponseBudgetError,
    TDSECostWarning,
)
from solprop.log import NullReporter
from solprop.nonlinear.tdse import (
    TDSEBasis,
    TDSECalibration,
    TDSEExecution,
    TDSEGasResponse,
    TDSESampling,
)
from solprop.output import MemoryOutput, open_file, resume
from solprop.stats import StatsSpec
from solprop.stepper.multistep import ComparisonMultistep, Multistep
from solprop.translate.tdse import AtomicCallBudget, TDSEResult, waveform_identity

# These fixtures run a TDSE response on a multi-node Gauss-Kronrod rule on purpose;
# the run-start cost warning SEP 20 adds is filtered by its class alone and
# asserted once below (test_this_module_s_tdse_fixture_raises_the_cost_warning).
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.TDSECostWarning")

# Binary-exact positions keep the ledger fixture at six whole steps; decimal
# accumulation can legitimately invoke the equal-approach landing rule.
_ATOMIC_STEP_M = 2.0**-20


class _Binding:
    """Return zero nonlinearity, with two charged chunks and an optional failure."""

    def __init__(self, descriptor, limit, *, fail_at=None):
        self.descriptor = descriptor
        self.budget = AtomicCallBudget(limit, 2)
        self.fail_at = fail_at
        self.purposes = []
        self.inputs = []
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
        self.inputs.append(fields.copy())
        self.budget.begin_evaluation(purpose=purpose)
        for _ in range(2):
            self.budget.charge(purpose=purpose, z=z)
            if self.ledger.attempted == self.fail_at:
                self.budget.record_failure()
                raise BackendError(f"manufactured {purpose} chunk failure")
        return TDSEResult(
            np.zeros_like(fields),
            np.zeros_like(fields),
            input_identity=waveform_identity(fields),
            configuration_identity=self.descriptor.identity,
            valid=np.ones(fields.shape, dtype=bool),
            shadow_orders=(1,),
        )

    def metadata(self):
        return {"kind": "manufactured shared-Adams fixture", "closed": self.closed}

    def memory_ledger(self):
        return {}

    def close(self):
        self.closed = True


def _atomic_simulation(*, method=None, limit=20, step_planes=0):
    parent = Simulation.from_capillary(
        256e-6,
        10 * _ATOMIC_STEP_M,
        "He",
        0.1,
        lambda0=800e-9,
        wavelength_limits=(400e-9, 1.6e-6),
        trange=60e-15,
        fwhm=6e-15,
        energy=1e-12,
        modes=1,
        n_r=4,
        n_save=2,
        plasma=False,
        noise=None,
        backend="numpy_ref",
        integrator=Multistep(formulation="interaction") if method is None else method,
        dz_init=_ATOMIC_STEP_M,
        dz_max=_ATOMIC_STEP_M,
        step_planes=step_planes,
    )
    basis = TDSEBasis(
        "manufactured", (0.0, 1e-9), (8,), 5, 1e-16, source="merge contract fixture"
    )
    calibration = TDSECalibration(
        "sep12-sep14-contract",
        "research",
        basis.identity,
        (400e-9, 2e-6),
        source="manufactured zero current; no atomic calculation",
        limitations=("no physical qualification",),
    )
    response = TDSEGasResponse(
        "helium", basis, 1e-19, "tdse_native", 1.0, calibration, TDSESampling()
    )
    return Simulation(
        parent.grid,
        parent.modes,
        [response],
        parent._field0,
        settings=dataclasses.replace(
            parent.settings,
            tdse_execution=TDSEExecution(max_atomic_calls=limit),
            # Zero source isolates accounting; linear dispersion and attenuation
            # still distinguish the future field from the launch field.
            absorber=None,
            stats=StatsSpec(names=("energy",)),
        ),
        initial_z=4 * _ATOMIC_STEP_M,
        noise=None,
        profiles=parent.profiles,
        frame=parent.frame,
    )


# SEP 16 section 7.2: both formulations drive the adapter with the same lifecycle --
# reservations, evaluation labels, budgets, resume and the reconciled ledgers.
FORMULATIONS = pytest.mark.parametrize(
    "formulation", ["interaction", "exponential"], ids=["interaction", "exponential"]
)


@pytest.mark.medium
@FORMULATIONS
def test_fused_predictor_failure_charges_both_ledgers_before_invocation(formulation):
    simulation = _atomic_simulation(method=Multistep(formulation=formulation))
    problem = simulation.translate()
    assert problem.tdse is not None
    binding = _Binding(problem.tdse, 20, fail_at=3)
    backend = NumpyRefBackend(problem, grid=simulation.grid, tdse_binding=binding)
    try:
        backend.set_initial(problem.initial.field0, None, z=problem.initial.z)
        initial = backend.observe()
        assert backend._adams is not None
        # The interaction formulation's fused path, or the exponential one's own.
        assert backend._adams._fused is (formulation == "interaction")
        with pytest.raises(BackendError, match="predictor chunk failure"):
            backend.propagate_to(simulation.grid.length)
        actual = backend.observe()
        assert binding.purposes == ["startup", "predictor"]
        assert binding.ledger.logical_evaluations == 2
        assert binding.ledger.attempted == 3
        assert binding.ledger.failed_calls == 1
        assert binding.ledger.reserved == 0
        calls = backend._adams.ledger
        assert (calls.startup, calls.predictor, calls.corrected) == (1, 1, 0)
        assert calls.accepted_steps == calls.rejected_steps == 0
        assert actual.z == initial.z == simulation.initial_z
        assert_array_equal(actual.field, initial.field)
    finally:
        backend.close()
        binding.close()


@pytest.mark.medium
def test_corrected_provider_failure_checkpoints_the_previous_accepted_state(
    monkeypatch,
):
    method = ComparisonMultistep(order=1, corrector="pece", corrector_order=2)
    simulation = _atomic_simulation(method=method)
    descriptor = simulation.translate().tdse
    assert descriptor is not None
    binding = _Binding(descriptor, 20, fail_at=5)
    monkeypatch.setattr(
        runtime_module, "prepare_binding", lambda *args, **kwargs: binding
    )
    output = MemoryOutput()
    with pytest.raises(BackendError, match="corrected chunk failure"):
        simulation.run(output, progress="none")
    assert binding.closed
    assert binding.purposes == ["startup", "predictor", "corrected"]
    # This guard makes rollback observable even though the manufactured nonlinear
    # current vanishes: the linear evolution changed the attempted end field.
    assert not np.array_equal(binding.inputs[0], binding.inputs[-1])
    checkpoint = output.terminal_checkpoint
    assert checkpoint is not None and checkpoint.landed
    assert checkpoint.plane.z == simulation.initial_z
    assert checkpoint.next_save_index == 0
    assert checkpoint.plane.step_history.n_steps == 0
    assert_array_equal(checkpoint.plane.field, output.fields(convention="fft")[0])
    outcome = output.outcome
    assert outcome is not None and outcome.status == "failed"
    assert outcome.z == checkpoint.plane.z
    assert outcome.logical_calls["startup"] == 1
    assert outcome.logical_calls["predictor"] == 1
    assert outcome.logical_calls["corrected"] == 1
    assert outcome.logical_calls["accepted_steps"] == 0
    assert outcome.atomic_calls["attempted"] == 5
    assert outcome.atomic_calls["failed_calls"] == 1
    assert outcome.atomic_calls["reserved"] == 0


def _install_bindings(monkeypatch):
    bindings = []

    def prepare(descriptor, execution, *args, **kwargs):
        binding = _Binding(descriptor, execution.max_atomic_calls)
        bindings.append(binding)
        return binding

    monkeypatch.setattr(runtime_module, "prepare_binding", prepare)
    return bindings


@pytest.mark.medium
@FORMULATIONS
def test_interior_natural_save_resumes_without_resetting_atomic_budget(
    monkeypatch, tmp_path, formulation
):
    simulation = _atomic_simulation(
        method=Multistep(formulation=formulation), limit=4, step_planes=1
    )
    bindings = _install_bindings(monkeypatch)
    path = tmp_path / "natural-atomic.h5"
    with pytest.raises(ResponseBudgetError):
        simulation.run(path, progress="none")
    first = resume(path)
    assert first.checkpoint is not None and first.checkpoint.landed
    assert first.checkpoint.next_save_index == 0
    assert first.n_saved == 2
    assert simulation.initial_z < first.plane.z < simulation.grid.length
    with open_file(path) as saved:
        positions = np.asarray(saved["z"]).copy()
        assert_array_equal(positions, [simulation.initial_z, first.plane.z])
    assert first.outcome is not None and first.outcome.status == "budget_exhausted"
    assert first.outcome.z == first.plane.z
    assert first.outcome.atomic_calls["attempted"] == 4
    assert first.outcome.logical_calls["accepted_steps"] == 1

    with pytest.raises(ResponseBudgetError):
        simulation.run(path, resume=True, progress="none")
    second = resume(path)
    assert len(bindings) == 2 and bindings[1].purposes == []
    assert second.outcome is not None
    assert second.outcome.atomic_calls == first.outcome.atomic_calls
    assert second.outcome.logical_calls == first.outcome.logical_calls
    assert second.n_saved == first.n_saved and second.plane.z == first.plane.z
    # A no-work resume divides the saved modal field by sqrt(Re n_eff), then
    # multiplies it back at checkpointing. Bound those real scalings, including
    # complex division's reciprocal, by gamma_4; storage conjugations are exact.
    assert second.plane.field is not None and first.plane.field is not None
    assert_allclose(
        second.plane.field,
        first.plane.field,
        rtol=4 * np.finfo(float).eps / (1 - 4 * np.finfo(float).eps),
        atol=0.0,
    )
    assert second.history.n_steps == first.history.n_steps

    simulation.settings = dataclasses.replace(
        simulation.settings, tdse_execution=TDSEExecution(max_atomic_calls=20)
    )
    simulation.run(path, resume=True, progress="none")
    final = resume(path)
    assert final.checkpoint is None
    assert final.plane.z == simulation.grid.length
    assert final.outcome is not None and final.outcome.status == "complete"
    calls = final.outcome.logical_calls
    # The resume continues the march's history from the checkpoint's snapshot: no
    # second startup evaluation, and its two atomic calls are not spent again.
    assert calls["startup"] == 1
    assert calls["accepted_steps"] == 6
    assert calls["predictor"] == 6
    orders = calls["orders"]
    assert isinstance(orders, list) and orders[1] == 1
    assert final.outcome.atomic_calls["attempted"] == 14
    assert final.outcome.atomic_calls["reserved"] == 0
    assert all(binding.closed for binding in bindings)
    with open_file(path) as saved:
        final_positions = np.asarray(saved["z"])
        assert_array_equal(final_positions[:2], positions)
        assert np.all(np.diff(final_positions) > 0)
        assert final_positions[-1] == simulation.grid.length
        assert final.n_saved == 7


def _dense_simulation(*, step_planes):
    return Simulation.from_gnlse(
        0.0,
        1e-5,
        [0.0, 0.0, -1e-26],
        lambda0=800e-9,
        wavelength_limits=(500e-9, 1.6e-6),
        trange=200e-15,
        fwhm=20e-15,
        peak_power=1.0,
        save_planes=[1.5e-6, 1e-5],
        backend="numpy_ref",
        integrator=Multistep(formulation="interaction"),
        dz_init=1e-6,
        dz_max=1e-6,
        noise=None,
        interpolate_planes=True,
        step_planes=step_planes,
        stats=StatsSpec(names=("energy",)),
    )


@pytest.mark.medium
@pytest.mark.parametrize("step_planes", [0, 1], ids=["interpolated", "natural"])
def test_dense_observation_failure_records_actual_checkpoint_landing(
    monkeypatch, step_planes
):
    simulation = _dense_simulation(step_planes=step_planes)
    original = NumpyRefBackend.stats
    observed = []

    def fail_second(self, spec):
        observed.append(self._z)
        if len(observed) == 2:
            raise DriverError("manufactured observation statistics failure")
        return original(self, spec)

    monkeypatch.setattr(NumpyRefBackend, "stats", fail_second)
    output = MemoryOutput()
    with pytest.raises(DriverError, match="observation statistics failure"):
        simulation.run(output, progress="none")
    checkpoint = output.terminal_checkpoint
    assert checkpoint is not None
    assert checkpoint.plane.z == observed[-1]
    assert checkpoint.landed is bool(step_planes)
    assert checkpoint.next_save_index == (1 if step_planes == 0 else 0)
    history = checkpoint.plane.step_history
    ends = (history.z + history.h)[history.accepted]
    assert ends.size > 0
    if step_planes:
        assert ends[-1] == checkpoint.plane.z
    else:
        assert checkpoint.plane.z == 1.5e-6
        assert ends[-1] > checkpoint.plane.z
    assert output.outcome is not None and output.outcome.status == "failed"
    assert output.outcome.z == checkpoint.plane.z


@pytest.mark.medium
def test_dense_natural_reporter_failure_retains_saved_endpoint_classification():
    class FailingReporter(NullReporter):
        def plane(self, z, summary):
            raise DriverError("manufactured reporter failure after saving")

    simulation = _dense_simulation(step_planes=1)
    output = MemoryOutput()
    with pytest.raises(DriverError, match="reporter failure after saving"):
        simulation.run(output, progress=FailingReporter())
    checkpoint = output.terminal_checkpoint
    assert checkpoint is not None and checkpoint.landed
    # The equal-approach rule may shorten this natural step. Its actual saved
    # accepted endpoint, rather than the requested step length, defines landing.
    assert 0.0 < checkpoint.plane.z < simulation.settings.save_planes[0]
    assert checkpoint.plane.z == (
        output["step_history/z"][-1] + output["step_history/h"][-1]
    )
    assert output["step_history/accepted"][-1]
    assert checkpoint.next_save_index == 0
    assert output.n_saved == 2
    assert_array_equal(checkpoint.plane.field, output.fields(convention="fft")[-1])
    # The history is already in the regular observation; a terminal checkpoint
    # retains its landing classification without duplicating those attempts.
    assert checkpoint.plane.step_history.n_steps == 0
    assert_array_equal(output["step_history/counts"], [0, 1])
    assert output.outcome is not None and output.outcome.status == "failed"
    assert output.outcome.z == checkpoint.plane.z


@pytest.mark.medium
def test_this_module_s_tdse_fixture_raises_the_cost_warning(monkeypatch):
    # The module filters the warning by class (above); this asserts it once here.
    simulation = _atomic_simulation(limit=20)
    _install_bindings(monkeypatch)
    with pytest.warns(TDSECostWarning, match="one atom in every atomic call") as record:
        simulation.run(MemoryOutput(), progress="none")
    assert len([one for one in record if one.category is TDSECostWarning]) == 1
