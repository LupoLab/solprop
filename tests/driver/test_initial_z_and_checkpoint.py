# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise interior launches and restartable stops on tiny optical problems."""

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from tests.backends.test_tdse_march import ManufacturedBinding
from tests.driver.test_run import NullBackend

from solprop.backends.memory import MemoryEstimate
from solprop.backends.multistep import CallLedger
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import select
from solprop.boundaries import absorber
from solprop.driver import Simulation, SimulationSettings, run_problem
from solprop.driver import simulation as simulation_module
from solprop.driver import tdse as runtime_module
from solprop.driver.memory import MemoryAdmission, MemoryLimit
from solprop.exceptions import (
    DriverError,
    MemoryBudgetError,
    MemoryBudgetWarning,
    ResponseBudgetError,
    TDSECostWarning,
)
from solprop.grid import FieldGrid, RadialGrid
from solprop.linear import Frame
from solprop.noise import NoiseRealisation
from solprop.nonlinear.tdse import (
    TDSEBasis,
    TDSECalibration,
    TDSEExecution,
    TDSEGasResponse,
    TDSESampling,
)
from solprop.output import (
    MemoryOutput,
    RunOutcome,
    SavedPlane,
    StepHistory,
    TerminalCheckpoint,
    resume,
)
from solprop.output.hdf5 import ResumeState
from solprop.output.schema import Convention, RunMetadata, serialise
from solprop.profiles import PressureGradient
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import PIController
from solprop.stepper.multistep import Multistep
from solprop.translate.tdse import AtomicCallLedger

# These fixtures run a TDSE response on a multi-node Gauss-Kronrod rule on purpose;
# the run-start cost warning SEP 20 adds is filtered by its class alone and
# asserted once below (test_this_module_s_tdse_fixture_raises_the_cost_warning).
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.TDSECostWarning")


def tiny_simulation(*, max_calls=None):
    return Simulation.from_gnlse(
        0.0,
        1e-5,
        [0.0, 0.0, -1e-26],
        lambda0=800e-9,
        wavelength_limits=(500e-9, 1.6e-6),
        trange=200e-15,
        fwhm=20e-15,
        peak_power=1.0,
        n_save=2,
        backend="numpy_ref",
        integrator=Multistep(max_calls=max_calls),
        dz_init=1e-6,
        dz_max=1e-6,
        noise=None,
    )


@pytest.fixture
def atomic_simulation():
    parent = Simulation.from_capillary(
        256e-6,
        1e-5,
        "He",
        0.1,
        lambda0=800e-9,
        wavelength_limits=(400e-9, 1.6e-6),
        trange=60e-15,
        fwhm=6e-15,
        energy=1e-9,
        modes=1,
        n_r=4,
        n_save=2,
        plasma=False,
        noise=None,
        backend="numpy_ref",
        integrator=Multistep(),
    )
    basis = TDSEBasis(
        "manufactured", (0.0, 1e-9), (8,), 5, 1e-16, source="contract fixture"
    )
    calibration = TDSECalibration(
        "research-fixture",
        "research",
        basis.identity,
        (400e-9, 2e-6),
        source="contract fixture",
        limitations=("unqualified",),
    )
    response = TDSEGasResponse(
        "helium", basis, 1e-19, "tdse_native", 1.0, calibration, TDSESampling(2)
    )
    return Simulation(
        parent.grid,
        parent.modes,
        [response],
        parent._field0,
        settings=dataclasses.replace(
            parent.settings,
            tdse_execution=TDSEExecution(max_atomic_calls=4),
            stats=StatsSpec(names=("energy", "peak_power")),
        ),
        noise=None,
        profiles=parent.profiles,
        configuration=parent.configuration,
        frame=parent.frame,
    )


def test_with_input_retains_absolute_tables_and_filters_only_explicit_launch():
    simulation = tiny_simulation()
    original = simulation.translate()
    launch = simulation.with_input(simulation._field0, initial_z=5e-6)
    problem = launch.translate()
    assert launch.initial_z == problem.initial.z == 5e-6
    assert problem.linear is original.linear
    assert launch.grid is simulation.grid
    assert launch.frame is simulation.frame
    assert launch.metadata().initial_z == 5e-6
    assert (
        launch.metadata().configuration_hash()
        != simulation.metadata().configuration_hash()
    )
    assert_array_equal(problem.stepper.save_planes, [1e-5])
    again = launch.with_input(launch._field0)
    assert again.initial_z == 5e-6
    assert_array_equal(again.settings.save_planes, launch.settings.save_planes)


def test_interior_input_uses_actual_z_modal_weight():
    simulation = Simulation.from_capillary(
        256e-6,
        0.001,
        "He",
        PressureGradient.two_point(0.001, 0.1, 1.0),
        lambda0=800e-9,
        wavelength_limits=(400e-9, 1.6e-6),
        trange=60e-15,
        fwhm=6e-15,
        energy=1e-9,
        modes=1,
        n_r=4,
        n_save=3,
        plasma=False,
        noise=None,
        backend="numpy_ref",
        integrator=Multistep(),
    )
    launch = simulation.with_input(simulation._field0, initial_z=0.0005)
    mask = simulation.grid.active_mask
    expected = simulation._field0[..., mask] / np.sqrt(
        np.real(
            simulation.modes[0].neff(simulation.grid.omega[mask], z=launch.initial_z)
        )
    )
    # The independently sampled effective index and its Chebyshev table agree to
    # the table fit accuracy; zero-position normalization differs by much more.
    assert_allclose(
        launch.translate().initial.field0[0][..., mask], expected, rtol=1e-9, atol=0
    )
    rebuilt = Simulation(
        simulation.grid,
        simulation.modes,
        simulation.responses,
        simulation._field0,
        settings=dataclasses.replace(simulation.settings, save_planes=[0.001]),
        initial_z=0.0005,
        noise=None,
        profiles=simulation.profiles,
        frame=simulation.frame,
        configuration=simulation.configuration,
    )
    assert_allclose(
        rebuilt.translate().initial.field0,
        launch.translate().initial.field0,
        rtol=1e-12,
        atol=0,
    )


@pytest.mark.parametrize("z", [-1.0, np.nan, np.inf, 1e-5, 2e-5])
def test_interior_launch_refuses_invalid_position(z):
    simulation = tiny_simulation()
    with pytest.raises(DriverError, match="initial_z"):
        simulation.with_input(simulation._field0, initial_z=z)


def test_interior_launch_requires_explicit_field():
    simulation = tiny_simulation()
    with pytest.raises(DriverError, match="explicit modal spectrum"):
        simulation.with_input(simulation.initial, initial_z=5e-6)


@pytest.mark.medium
def test_interior_run_saves_its_actual_entrance_and_completes():
    simulation = tiny_simulation()
    launch = simulation.with_input(simulation._field0, initial_z=5e-6)
    output = launch.run(progress="none")
    assert isinstance(output, MemoryOutput)
    assert_array_equal(output["z"], [5e-6, 1e-5])
    assert output.outcome is not None and output.outcome.status == "complete"
    assert output.terminal_checkpoint is None


@pytest.mark.medium
def test_budget_stop_is_separate_and_resume_cannot_reset_its_budget(tmp_path):
    simulation = tiny_simulation(max_calls=4)
    path = tmp_path / "budget.h5"
    with pytest.raises(ResponseBudgetError):
        simulation.run(path, progress="none")
    stopped = resume(path)
    assert stopped.n_saved == 1
    assert 0 < stopped.plane.z < 1e-5
    assert stopped.checkpoint is not None and stopped.checkpoint.next_save_index == 0
    assert stopped.outcome is not None and stopped.outcome.status == "budget_exhausted"
    calls = stopped.outcome.logical_calls
    with pytest.raises(ResponseBudgetError):
        simulation.run(path, resume=True, progress="none")
    again = resume(path)
    assert again.plane.z == stopped.plane.z
    assert again.outcome is not None and again.outcome.logical_calls == calls
    assert_array_equal(again.plane.field, stopped.plane.field)
    assert again.history.n_steps == stopped.history.n_steps


def test_resume_uses_checkpoint_position_and_rejects_inconsistent_index(tmp_path):
    simulation = tiny_simulation()
    problem = simulation.translate()
    memory = MemoryOutput()
    memory.open(simulation.metadata())
    memory.save(SavedPlane(z=0.0, field=simulation._field0))
    memory.record_outcome(
        RunOutcome("failed", 5e-6),
        TerminalCheckpoint(SavedPlane(z=5e-6, field=simulation._field0), 0),
    )
    path = tmp_path / "checkpoint.h5"
    memory.to_file(path)
    state = resume(path)
    backend = select("numpy_ref", problem, grid=simulation.grid, modes=simulation.modes)
    try:
        assert simulation._prime(backend, problem, state) == 0
        assert backend.observe().z == 5e-6
        assert state.checkpoint is not None
        invalid = dataclasses.replace(
            state, checkpoint=dataclasses.replace(state.checkpoint, next_save_index=1)
        )
        with pytest.raises(DriverError, match="next_save_index"):
            simulation._prime(backend, problem, invalid)
    finally:
        backend.close()


def test_ordinary_resume_without_outcome_restores_plane_and_starts_fresh_ledgers(
    tmp_path,
):
    simulation = tiny_simulation()
    simulation.settings = dataclasses.replace(
        simulation.settings, save_planes=[5e-6, 1e-5]
    )
    problem = simulation.translate()
    memory = MemoryOutput()
    memory.open(simulation.metadata())
    memory.save(SavedPlane(z=0.0, field=simulation._field0))
    memory.save(SavedPlane(z=5e-6, field=simulation._field0))
    path = tmp_path / "without-outcome.h5"
    memory.to_file(path)
    state = resume(path)
    assert state.outcome is None
    backend = select("numpy_ref", problem, grid=simulation.grid, modes=simulation.modes)
    try:
        assert simulation._prime(backend, problem, state) == 1
        observed = backend.observe()
        assert observed.z == 5e-6
        assert_array_equal(
            observed.require_field()[..., problem.grids.active_mask],
            problem.initial.field0[..., problem.grids.active_mask],
        )
        assert np.all(observed.require_field()[..., ~problem.grids.active_mask] == 0)
        assert backend.metadata()["response_calls"] == dataclasses.asdict(CallLedger())
    finally:
        backend.close()


@pytest.mark.medium
def test_statistics_failure_keeps_drained_history_without_retrying_statistics(
    monkeypatch,
):
    simulation = tiny_simulation()
    problem = simulation.translate()
    backend = select("numpy_ref", problem, grid=simulation.grid, modes=simulation.modes)
    backend.set_initial(problem.initial.field0, None)
    calls = []
    original = NumpyRefBackend.stats

    def fail_second(self, spec):
        calls.append(1)
        if len(calls) == 2:
            raise DriverError("manufactured statistics failure")
        return original(self, spec)

    monkeypatch.setattr(NumpyRefBackend, "stats", fail_second)
    memory = MemoryOutput()
    try:
        with pytest.raises(DriverError, match="manufactured statistics failure"):
            run_problem(
                problem,
                backend,
                memory,
                metadata=simulation.metadata(),
                progress="none",
            )
    finally:
        backend.close()
    assert len(calls) == 2
    assert memory.n_saved == 1
    assert memory.outcome is not None and memory.outcome.status == "failed"
    assert memory.terminal_checkpoint is not None
    assert memory.terminal_checkpoint.plane.z == 1e-5
    assert memory.terminal_checkpoint.plane.step_history.n_steps > 0


def test_backend_construction_failure_closes_owned_binding(
    monkeypatch, atomic_simulation
):
    simulation = atomic_simulation
    closed = []

    class Binding:
        """The binding protocol, of which only `close` is reached."""

        ledger = None

        def evaluate(self, fields, *, z, purpose, density=None):
            raise AssertionError("not reached")

        reserve = release_reservation = restore_ledger = record_trial = evaluate
        metadata = memory_ledger = evaluate

        def close(self):
            closed.append("binding")

    # Native preparation is replaced, so this failure-path check never starts an
    # atomic runtime or imports the optional provider.
    monkeypatch.setattr(
        runtime_module, "prepare_binding", lambda *args, **kwargs: Binding()
    )

    def refuse(*args, **kwargs):
        raise DriverError("manufactured backend construction failure")

    monkeypatch.setattr(simulation_module, "select", refuse)
    with pytest.raises(DriverError, match="construction failure"):
        simulation.run(progress="none")
    assert closed == ["binding"]


def test_tdse_refines_once_and_execution_does_not_change_scientific_identity(
    atomic_simulation,
):
    simulation = atomic_simulation
    assert simulation.grid.n_to == 2 * simulation._base_grid.n_to
    assert simulation.settings.absorber is not None
    old_boundary = absorber(
        simulation._base_grid,
        boundary_length=simulation.settings.absorber.reference_length,
    )
    assert_array_equal(
        simulation.settings.absorber.temporal_rate, old_boundary.temporal_rate
    )
    assert_array_equal(
        simulation.settings.absorber.spectral_rate, old_boundary.spectral_rate
    )
    assert_array_equal(
        simulation.settings.absorber.temporal_rate_fine[::2],
        old_boundary.temporal_rate_fine,
    )
    launch = simulation.with_input(simulation._field0, initial_z=5e-6)
    assert launch.grid.n_to == simulation.grid.n_to
    assert_array_equal(launch.grid.t, simulation.grid.t)
    identity = simulation.metadata().configuration_hash()
    simulation.settings = dataclasses.replace(
        simulation.settings,
        tdse_execution=TDSEExecution(
            device="cuda", max_atomic_calls=8, max_batch_rows=1
        ),
    )
    assert simulation.metadata().configuration_hash() == identity


def test_tdse_does_not_guess_how_to_resample_custom_absorbers(atomic_simulation):
    simulation = atomic_simulation
    standard = absorber(simulation._base_grid)
    custom = dataclasses.replace(
        standard, temporal_rate_fine=0.5 * standard.temporal_rate_fine
    )
    with pytest.raises(DriverError, match="custom absorber"):
        Simulation(
            simulation._base_grid,
            simulation.modes,
            simulation.responses,
            simulation._field0,
            settings=dataclasses.replace(simulation.settings, absorber=custom),
            noise=None,
            profiles=simulation.profiles,
            configuration=simulation.configuration,
        )


def test_tdse_refuses_implicit_execution_and_dense_saves(atomic_simulation):
    simulation = atomic_simulation
    with pytest.raises(DriverError, match="explicit numpy_ref"):
        dataclasses.replace(simulation.settings, backend="auto")
    with pytest.raises(DriverError, match="tdse_execution"):
        simulation.settings = dataclasses.replace(
            simulation.settings, tdse_execution=None
        )
    with pytest.raises(DriverError, match="landed"):
        simulation.settings = dataclasses.replace(
            simulation.settings, interpolate_planes=True
        )


def write_atomic_state(
    simulation, path, *, complete=False, missing_ledger=False, metadata=None
):
    memory = MemoryOutput()
    memory.open(simulation.metadata() if metadata is None else metadata)
    memory.save(SavedPlane(z=0.0, field=simulation._field0))
    z = 1e-5 if complete else 5e-6
    checkpoint = (
        None
        if complete
        else TerminalCheckpoint(SavedPlane(z=z, field=simulation._field0), 0)
    )
    if complete:
        memory.save(SavedPlane(z=z, field=simulation._field0))
    memory.record_outcome(
        RunOutcome(
            "complete" if complete else "failed",
            z,
            logical_calls=dataclasses.asdict(CallLedger(startup=1, predictor=1)),
            atomic_calls={}
            if missing_ledger
            else dataclasses.asdict(AtomicCallLedger(4, 2, 0, 2, 0, 0)),
            execution={"tdse": {"device": "cpu"}},
        ),
        checkpoint,
    )
    memory.to_file(path)


@pytest.mark.parametrize("case", ["missing", "cap"])
def test_tdse_resume_refusal_precedes_native_preparation(
    atomic_simulation, monkeypatch, tmp_path, case
):
    simulation = atomic_simulation
    path = tmp_path / "invalid-resume.h5"
    write_atomic_state(simulation, path, missing_ledger=case == "missing")
    if case == "cap":
        simulation.settings = dataclasses.replace(
            simulation.settings, tdse_execution=TDSEExecution(max_atomic_calls=1)
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("native preparation must follow resume admission")

    monkeypatch.setattr(runtime_module, "prepare_binding", forbidden)
    with pytest.raises(
        DriverError, match=r"cumulative call ledgers|below previously attempted"
    ):
        simulation.run(path, resume=True, progress="none")


def test_tdse_completed_resume_prepares_no_resources(
    atomic_simulation, monkeypatch, tmp_path
):
    path = tmp_path / "complete.h5"
    write_atomic_state(atomic_simulation, path, complete=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("completed output needs no new resources")

    monkeypatch.setattr(runtime_module, "prepare_binding", forbidden)
    monkeypatch.setattr(simulation_module, "select", forbidden)
    output = atomic_simulation.run(path, resume=True, progress="none")
    assert output.n_saved == 2


def test_shared_memory_budget_counts_optical_and_atomic_storage(
    atomic_simulation, monkeypatch
):
    monkeypatch.setattr(
        runtime_module,
        "estimate_binding_memory",
        lambda *args, **kwargs: {"atomic": 70},
    )
    problem = atomic_simulation.translate()
    execution = TDSEExecution(memory_limit=200)
    estimate = MemoryEstimate("numpy_ref", {"base": 60})
    with pytest.raises(MemoryBudgetError, match="combined optical and TDSE"):
        simulation_module._tdse_memory_execution(
            problem, execution, MemoryAdmission(estimate, MemoryLimit(100))
        )
    admitted = simulation_module._tdse_memory_execution(
        problem, execution, MemoryAdmission(estimate, MemoryLimit(150))
    )
    assert admitted.memory_limit == 150
    with pytest.warns(MemoryBudgetWarning, match="combined optical and TDSE"):
        advisory = simulation_module._tdse_memory_execution(
            problem, execution, MemoryAdmission(estimate, MemoryLimit(100, "warn"))
        )
    assert advisory.memory_limit == 200


@pytest.mark.medium
def test_resume_saves_checkpoint_on_missing_final_plane_before_completion(
    monkeypatch, tmp_path
):
    simulation = tiny_simulation()
    path = tmp_path / "missing-final.h5"
    original = NumpyRefBackend.stats
    attempts = []

    def fail_final(self, spec):
        attempts.append(1)
        if len(attempts) == 2:
            raise DriverError("manufactured final statistics failure")
        return original(self, spec)

    monkeypatch.setattr(NumpyRefBackend, "stats", fail_final)
    with pytest.raises(DriverError, match="final statistics"):
        simulation.run(path, progress="none")
    first = resume(path)
    assert first.n_saved == 1 and first.plane.z == 1e-5
    monkeypatch.setattr(NumpyRefBackend, "stats", original)
    simulation.run(path, resume=True, progress="none")
    finished = resume(path)
    assert finished.n_saved == 2
    assert finished.checkpoint is None
    assert finished.outcome is not None and finished.outcome.status == "complete"
    assert finished.history.n_steps == first.history.n_steps


@pytest.mark.parametrize("initial_z", [np.nan, -1.0, 1e-5, 2e-5])
def test_constructor_refuses_invalid_launch_before_translation(initial_z):
    simulation = tiny_simulation()
    with pytest.raises(DriverError, match="initial_z"):
        Simulation(
            simulation.grid,
            simulation.modes,
            simulation.responses,
            simulation._field0,
            settings=simulation.settings,
            initial_z=initial_z,
        )


def test_interior_launch_refuses_modified_noise_before_realisation():
    simulation = tiny_simulation()

    def launched(model):
        return Simulation(
            simulation.grid,
            simulation.modes,
            simulation.responses,
            simulation._field0,
            settings=simulation.settings,
            initial_z=5e-6,
            noise=NoiseRealisation(np.zeros_like(simulation._field0), 1, model),
        )

    with pytest.raises(DriverError, match="refuses the modified noise model"):
        launched("modified")
    # SEP 37: an explicit input draw enters at the launch.
    admitted = launched("input").noise
    assert admitted is not None
    assert admitted.model == "input"


@pytest.mark.parametrize("case", ["outside", "field", "logical", "cap", "atomic"])
def test_resume_rejects_incompatible_state_before_backend_allocation(case):
    simulation = tiny_simulation(max_calls=1 if case == "cap" else None)
    problem = simulation.translate()
    saved = SavedPlane(z=5e-6, field=simulation._field0)
    outcome = None
    if case == "outside":
        saved = dataclasses.replace(saved, z=2e-5)
        message = "outside"
    elif case == "field":
        saved = dataclasses.replace(saved, field=np.full_like(saved.field, np.nan))
        message = "finite stored field"
    elif case in ("logical", "cap"):
        outcome = RunOutcome(
            "failed", saved.z, logical_calls=dataclasses.asdict(CallLedger(startup=2))
        )
        if case == "logical":
            problem = dataclasses.replace(
                problem, stepper=dataclasses.replace(problem.stepper, integrator=None)
            )
        message = "Adams integration" if case == "logical" else "below previously"
    else:
        outcome = RunOutcome(
            "failed",
            saved.z,
            atomic_calls=dataclasses.asdict(AtomicCallLedger(4, 2, 0, 1, 0, 0)),
        )
        message = "explicit TDSE execution"
    state = ResumeState(
        simulation.metadata(), saved, 1, StepHistory.empty(), outcome=outcome
    )
    with pytest.raises(DriverError, match=message):
        simulation_module._validate_resume(problem, state, None)


def test_atomic_memory_admission_refuses_an_ordinary_problem():
    problem = tiny_simulation().translate()
    admission = MemoryAdmission(
        MemoryEstimate("numpy_ref", {"base": 60}), MemoryLimit(100)
    )
    with pytest.raises(DriverError, match="TDSE descriptor"):
        simulation_module._tdse_memory_execution(problem, TDSEExecution(), admission)


@pytest.mark.parametrize("case", ["unused", "mixed", "missing"])
def test_tdse_construction_refuses_inconsistent_response_ownership(
    atomic_simulation, case
):
    simulation = atomic_simulation
    responses = simulation.responses
    settings = simulation.settings
    if case == "unused":
        responses = ()
        message = "requires a TDSEGasResponse"
    elif case == "mixed":
        responses = (*responses, *responses)
        message = "complete electronic response"
    else:
        settings = dataclasses.replace(settings, tdse_execution=None)
        message = "explicit tdse_execution"
    with pytest.raises(DriverError, match=message):
        Simulation(
            simulation._base_grid,
            simulation.modes,
            responses,
            simulation._field0,
            settings=settings,
        )


@pytest.mark.parametrize("case", ["unused", "noise", "collar", "route"])
def test_tdse_settings_revalidation_refuses_unsupported_execution(
    atomic_simulation, case
):
    simulation = atomic_simulation
    settings = simulation.settings
    if case == "unused":
        simulation.responses = ()
        message = "requires a TDSEGasResponse"
    elif case == "noise":
        simulation.noise = NoiseRealisation(
            np.zeros_like(simulation._field0), 1, "modified"
        )
        message = "modified"
    elif case == "collar":
        settings = dataclasses.replace(settings, absorber_scheme="split_step")
        message = "in-stage temporal collar"
    else:
        simulation.wall = object()
        message = "uncoupled modal capillary"
    with pytest.raises(DriverError, match=message):
        simulation_module._check_tdse_execution(simulation, settings)


class LegacyOutput:
    """Delegate ordinary output while deliberately offering no checkpoint protocol."""

    def __init__(self):
        self.stored = MemoryOutput()

    def open(self, metadata, *, force=False):
        self.stored.open(metadata, force=force)

    def save(self, plane):
        self.stored.save(plane)

    def close(self):
        self.stored.close()

    @property
    def n_saved(self):
        return self.stored.n_saved

    @property
    def metadata(self):
        return self.stored.metadata

    @property
    def stat_names(self):
        return self.stored.stat_names

    @property
    def step_stat_names(self):
        return self.stored.step_stat_names

    def __getitem__(self, key):
        return self.stored[key]

    def __contains__(self, key):
        return key in self.stored

    def fields(self, convention: Convention = "optics"):
        return self.stored.fields(convention)


@pytest.mark.parametrize("entry", ["simulation", "run_problem"])
def test_tdse_requires_checkpoint_output_before_preparing_native_resources(
    atomic_simulation, monkeypatch, entry
):
    def forbidden(*args, **kwargs):
        raise AssertionError("resource preparation must follow output admission")

    monkeypatch.setattr(runtime_module, "prepare_binding", forbidden)
    with pytest.raises(DriverError, match="terminal checkpoints"):
        if entry == "simulation":
            atomic_simulation.run(LegacyOutput(), progress="none")
        else:
            run_problem(
                atomic_simulation.translate(),
                NullBackend(),
                LegacyOutput(),
                metadata=atomic_simulation.metadata(),
                progress="none",
            )


@pytest.mark.medium
def test_legacy_custom_output_retains_ordinary_success_and_failure_behavior():
    simulation = tiny_simulation()
    output = LegacyOutput()
    simulation.run(output, progress="none")
    assert output.n_saved == 2
    limited = tiny_simulation(max_calls=1)
    stopped = LegacyOutput()
    with pytest.raises(ResponseBudgetError):
        limited.run(stopped, progress="none")
    assert stopped.n_saved == 1


@pytest.mark.medium
def test_save_failure_after_append_does_not_duplicate_drained_history():
    class FlushFailure(MemoryOutput):
        def save(self, plane):
            super().save(plane)
            if self.n_saved == 2:
                raise DriverError("manufactured flush failure")

    output = FlushFailure()
    with pytest.raises(DriverError, match="flush failure"):
        tiny_simulation().run(output, progress="none")
    assert output.n_saved == 2
    assert output.terminal_checkpoint is not None
    assert output.terminal_checkpoint.plane.step_history.n_steps == 0
    assert output.terminal_checkpoint.plane.step_stats.n_rows == 0
    assert output["step_history/z"].size > 0


@pytest.mark.medium
def test_checkpoint_write_failure_preserves_original_error_and_note(caplog):
    class FailedCheckpoint(MemoryOutput):
        def record_outcome(self, outcome, checkpoint=None):
            if checkpoint is not None:
                raise DriverError("manufactured checkpoint write failure")
            super().record_outcome(outcome, checkpoint)

    with pytest.raises(ResponseBudgetError) as caught:
        tiny_simulation(max_calls=1).run(FailedCheckpoint(), progress="none")
    assert any("checkpoint write failure" in note for note in caught.value.__notes__)
    assert "could not persist the terminal checkpoint" in caplog.text


def test_run_refuses_lost_execution_settings_before_native_preparation(
    atomic_simulation, monkeypatch
):
    # A cached problem can outlive an externally corrupted settings record.
    atomic_simulation.translate()
    atomic_simulation.metadata()
    object.__setattr__(atomic_simulation.settings, "tdse_execution", None)

    def forbidden(*args, **kwargs):
        raise AssertionError("no native preparation with missing execution settings")

    monkeypatch.setattr(runtime_module, "prepare_binding", forbidden)
    with pytest.raises(DriverError, match="explicit tdse_execution"):
        atomic_simulation.run(progress="none")


@pytest.mark.medium
@pytest.mark.parametrize("limit", [4, None])
def test_driver_records_binding_facts_before_close_on_success_and_budget_stop(
    atomic_simulation, monkeypatch, limit
):
    simulation = atomic_simulation
    simulation.settings = dataclasses.replace(
        simulation.settings,
        tdse_execution=TDSEExecution(max_atomic_calls=limit),
        dz_init=1e-6,
        dz_max=1e-6,
    )
    binding = ManufacturedBinding(simulation.translate().tdse, limit)
    monkeypatch.setattr(
        runtime_module, "prepare_binding", lambda *args, **kwargs: binding
    )
    output = MemoryOutput()
    if limit is None:
        simulation.run(output, progress="none")
        expected_status = "complete"
    else:
        with pytest.raises(ResponseBudgetError):
            simulation.run(output, progress="none")
        expected_status = "budget_exhausted"
    assert binding.closed
    assert output.metadata is not None and output.metadata.backend is not None
    original_facts = output.metadata.backend.settings["tdse"]
    assert isinstance(original_facts, dict) and original_facts["closed"] is False
    assert output.outcome is not None
    assert output.outcome.status == expected_status
    latest_facts = output.outcome.execution["tdse"]
    assert isinstance(latest_facts, dict) and latest_facts["closed"] is False
    assert output.outcome.atomic_calls == dataclasses.asdict(binding.ledger)
    assert binding.ledger.attempted > 0
    assert binding.ledger.reserved == 0


def test_interior_metadata_retains_an_explicit_absolute_frame():
    simulation = tiny_simulation()
    frame = Frame.comoving(simulation.grid)
    launch = Simulation(
        simulation.grid,
        simulation.modes,
        simulation.responses,
        simulation._field0,
        settings=simulation.settings,
        frame=frame,
        initial_z=5e-6,
    )
    assert launch.metadata().configuration["frame"] is frame


def test_legacy_resume_without_outcome_loads_field_without_restoring_ledgers(
    monkeypatch,
):
    simulation = tiny_simulation()
    problem = simulation.translate()
    state = ResumeState(
        dataclasses.replace(simulation.metadata(), schema_version=9),
        SavedPlane(z=5e-6, field=simulation._field0),
        1,
        StepHistory.empty(),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy files carry no call ledgers to restore")

    monkeypatch.setattr(NumpyRefBackend, "restore_call_ledgers", forbidden)
    backend = select("numpy_ref", problem, grid=simulation.grid, modes=simulation.modes)
    try:
        assert simulation._prime(backend, problem, state) == 0
        assert backend.observe().z == state.plane.z
    finally:
        backend.close()


@pytest.mark.medium
def test_interior_atomic_budget_message_matches_terminal_checkpoint(
    atomic_simulation, monkeypatch
):
    """A refused trial reports its accepted absolute z in text and stored state."""
    atomic_simulation.settings = dataclasses.replace(
        atomic_simulation.settings,
        tdse_execution=TDSEExecution(max_atomic_calls=4),
        absorber=None,
        dz_init=1e-6,
        dz_max=1e-6,
    )
    simulation = atomic_simulation.with_input(atomic_simulation._field0, initial_z=5e-6)
    binding = ManufacturedBinding(simulation.translate().tdse, 4)
    monkeypatch.setattr(
        runtime_module, "prepare_binding", lambda *args, **kwargs: binding
    )
    output = MemoryOutput()
    with pytest.raises(ResponseBudgetError) as caught:
        simulation.run(output, progress="none")
    assert output.outcome is not None
    assert output.outcome.status == "budget_exhausted"
    assert output.terminal_checkpoint is not None
    position = output.terminal_checkpoint.plane.z
    assert position > simulation.initial_z
    assert caught.value.z == output.outcome.z == position
    assert f"z = {position:g} m" in str(caught.value)
    assert str(caught.value) in output.outcome.message
    assert caught.value.calls == binding.ledger.attempted == 4
    assert binding.ledger.reserved == 0


def test_spatial_resume_retains_physical_field_without_modal_normalisation(monkeypatch):
    grid = FieldGrid.from_limits(1e-5, 800e-9, (600e-9, 1.2e-6), 60e-15)
    simulation = Simulation.from_radial(
        grid,
        RadialGrid(3e-4, 8),
        SpatialMedium(),
        [],
        SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-9), 60e-6),
        settings=SimulationSettings(
            backend="numpy_ref",
            controller=PIController(rtol=1e-6, atol=1e-14),
            save_planes=[grid.length],
            dz_init=1e-6,
            dz_max=grid.length,
        ),
        noise=None,
    )
    problem = simulation.translate()
    backend = select("numpy_ref", problem, grid=grid, modes=simulation.modes)
    try:
        backend.set_initial(problem.initial.field0, None)
        field = backend.observe().require_field()[0].copy()
        state = ResumeState(
            simulation.metadata(),
            SavedPlane(z=5e-6, field=field),
            1,
            StepHistory.empty(),
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("spatial physical fields have no modal power weight")

        monkeypatch.setattr(simulation_module, "power_amplitude", forbidden)
        assert simulation._prime(backend, problem, state) == 0
        restored = backend.observe()
        assert restored.z == state.plane.z
        assert_array_equal(restored.require_field()[0], field)
    finally:
        backend.close()


def test_the_gated_native_source_changes_resume_identity_not_the_atomic_drive(
    atomic_simulation,
):
    # The cubic closure's source is ungated, the native one's gated and edge-free:
    # one field changes the source and the resume key, never the drive or absorber.
    native = atomic_simulation
    assert native.responses[0].kerr_model == "tdse_native"
    original = Simulation(
        native._base_grid,
        native.modes,
        [dataclasses.replace(native.responses[0], kerr_model="optmat_cubic")],
        native._field0,
        settings=native.settings,
        profiles=native.profiles,
        noise=None,
    )
    for simulation in (original, native):
        record = serialise(simulation.responses[0])
        assert isinstance(record, dict) and "source_filter" not in record
    assert_array_equal(original._field0, native._field0)
    assert_array_equal(original.grid.t_fine, native.grid.t_fine)
    assert original.settings.absorber is not None
    assert native.settings.absorber is not None
    assert_array_equal(
        original.settings.absorber.temporal_rate_fine,
        native.settings.absorber.temporal_rate_fine,
    )
    plain, gated = original.metadata(), native.metadata()
    assert plain.tdse is not None and gated.tdse is not None
    assert "source_filter" not in plain.tdse and "source_filter" not in gated.tdse
    assert plain.configuration_hash() != gated.configuration_hash()
    for metadata in (plain, gated):
        assert (
            RunMetadata.from_json(metadata.to_json()).configuration_hash()
            == metadata.configuration_hash()
        )


@pytest.mark.medium
def test_this_module_s_tdse_fixture_raises_the_cost_warning(
    atomic_simulation, monkeypatch
):
    # The module filters the warning by class (above); this asserts it once here.
    simulation = atomic_simulation
    simulation.settings = dataclasses.replace(
        simulation.settings,
        tdse_execution=TDSEExecution(max_atomic_calls=None),
        dz_init=1e-6,
        dz_max=1e-6,
    )
    binding = ManufacturedBinding(simulation.translate().tdse, None)
    monkeypatch.setattr(
        runtime_module, "prepare_binding", lambda *args, **kwargs: binding
    )
    with pytest.warns(TDSECostWarning, match="one atom in every atomic call") as record:
        simulation.run(MemoryOutput(), progress="none")
    assert len([one for one in record if one.category is TDSECostWarning]) == 1
