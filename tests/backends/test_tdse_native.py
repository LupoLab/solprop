# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The TDSE response on native against the reference (SEP 49 section 7.3).

Manufactured bindings replace the driver's binding, not the provider, and need no
worker: ``edge_source_cases.ManufacturedBinding`` returns a manufactured remainder
beside a large linear shadow. Native evaluates the source through its host source
hook, the reference through its own evaluation; both call the shared assembly
(``solprop.backends.tdse_source``), so ``F`` agrees bit for bit for the same field and
position, and what each backend does after it is compared within its budget.
"""

import dataclasses
import signal
from typing import Any

import numpy as np
import pytest

from bounds import EPS, assert_within
from edge_source_cases import ManufacturedBinding
from sessions.executors import IDENTITY, Started, drive
from solprop.backends import tdse_source
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.session import Start
from solprop.backends.sessions import start_session
from solprop.driver import Simulation
from solprop.exceptions import (
    BackendError,
    ResponseBudgetError,
    RunTerminated,
    TDSEArchiveError,
)
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
from solprop.translate.tdse import TDSEResult, lattice_times, waveform_identity

native = pytest.importorskip("solprop_native")

pytestmark = pytest.mark.medium

LENGTH = 1e-5


def _law(times):
    """A manufactured remainder ``S(t)``, A m per atom."""
    return 1e-27 * np.exp(-((times / 8e-15) ** 2)) * np.sin(2.4e15 * times)


def _case(
    formulation: Any = "interaction",
    kerr_model: Any = "tdse_native",
    *,
    length=LENGTH,
    **changes,
):
    """A short helium capillary with a manufactured TDSE response."""
    dz_max = changes.pop("dz_max", 1e-6)
    parent = Simulation.from_capillary(
        256e-6,
        length,
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
        integrator=Multistep(formulation=formulation),
        dz_init=min(1e-6, dz_max),
        dz_max=dz_max,
    )
    basis = TDSEBasis(
        "manufactured", (0.0, 1e-9), (8,), 5, 1e-16, source="native fixture"
    )
    calibration = TDSECalibration(
        "native-fixture",
        "research",
        basis.identity,
        (400e-9, 2e-6),
        source="native fixture",
        limitations=("no atomic calculation",),
    )
    responses = [
        TDSEGasResponse(
            "helium",
            basis,
            1e-19,
            kerr_model,
            1.0 if kerr_model == "tdse_native" else 2.0,
            calibration,
            TDSESampling(),
        )
    ]
    simulation = Simulation(
        parent.grid,
        parent.modes,
        responses,
        parent._field0,
        settings=dataclasses.replace(
            parent.settings,
            tdse_execution=TDSEExecution(),
            stats=StatsSpec(names=("energy",)),
            **changes,
        ),
        noise=None,
        profiles=parent.profiles,
        frame=parent.frame,
    )
    return simulation.translate(), simulation._run_request(), simulation.grid


class Unbudgeted(ManufacturedBinding):
    """The manufactured histories with no budget: a lone evaluation's, any purpose.

    A cubic replacement's third-order current rides beside them.
    """

    def evaluate(self, fields, *, z, purpose, density=None):
        del z, purpose, density
        first = self.linear * fields
        times = lattice_times(self.descriptor.samples, self.descriptor.spacing)
        remainder = np.broadcast_to(_law(times), fields.shape)
        cubic = self.descriptor.kerr_model != "tdse_native"
        third = 2.0**-120 * fields**3 if cubic else None
        return TDSEResult(
            first + remainder + (0.0 if third is None else third),
            first,
            input_identity=waveform_identity(fields),
            configuration_identity=self.descriptor.identity,
            valid=np.ones(fields.shape, dtype=bool),
            shadow_orders=(1, 3) if cubic else (1,),
            third_order_current=third,
        )


class Recording(ManufacturedBinding):
    """Record every binding call, with a failure or a hang injected on request."""

    def __init__(self, descriptor, limit, *, fail_evaluation=None, fail_record=None):
        super().__init__(descriptor, limit, _law)
        self.calls = []
        self.fail_evaluation = fail_evaluation
        self.fail_record = fail_record

    def reserve(self, logical_calls):
        self.calls.append(("reserve", logical_calls))
        super().reserve(logical_calls)

    def evaluate(self, fields, *, z, purpose, density=None):
        self.calls.append(("evaluate", purpose, z))
        if (
            self.fail_evaluation is not None
            and len(self.purposes) == self.fail_evaluation
        ):
            self.purposes.append(purpose)
            raise self.failure()
        return super().evaluate(fields, z=z, purpose=purpose, density=density)

    def failure(self) -> BaseException:
        return BackendError("manufactured provider failure")

    def record_trial(self, *, accepted):
        self.calls.append(("record_trial", accepted))
        trials = sum(1 for call in self.calls if call[0] == "record_trial")
        if self.fail_record is not None and trials == self.fail_record:
            raise TDSEArchiveError("manufactured archive failure")

    def release_reservation(self):
        self.calls.append(("release",))
        super().release_reservation()


class Interrupted(Recording):
    """Raise SIGTERM's exception from an evaluation, as the worker's wait raises a
    signal it took (a held one, on native)."""

    def failure(self) -> BaseException:
        return RunTerminated(signal.SIGTERM)


def _run(backend, problem, request, *, work_limit=256) -> Any:
    """Drive a session of `backend` to its end; return the run and its record.

    A signal's exception that the session raises rather than reports, as a run loop
    receives it, is returned in place of the run.
    """
    session = start_session(
        backend, problem, request, Start.entrance(problem), owns=False
    )
    try:
        run = drive(
            Started(session), export=IDENTITY, work_limit=work_limit, raises=False
        )
        record = dict(session.execution_record())
    except RunTerminated as signalled:
        return signalled, {"z": session.status().z}
    finally:
        session.close()
    return run, record


def _pair(case, binding_class=Recording, *, limit=1000, **options):
    """Run the case on both backends with a fresh binding each."""
    problem, request, grid = case
    descriptor = tdse_descriptor(problem.route)
    runs = {}
    for name, construct in (
        ("numpy_ref", NumpyRefBackend),
        ("native", native.NativeBackend),
    ):
        bound = binding_class(descriptor, limit, **options)
        backend = construct(problem, request, grid=grid, tdse_binding=bound)
        try:
            run, record = _run(backend, problem, request)
            metadata = backend.metadata()
        finally:
            backend.close()
        runs[name] = (run, record, bound, metadata)
    return runs


def _reference_source(problem, request, grid, binding, field, z):
    """``F`` as the reference forms it at `field` and `z`, its scalars and prefactor."""
    reference = NumpyRefBackend(problem, request, grid=grid, tdse_binding=binding)
    try:
        reference.set_initial(field, None)
        bundle = reference._stage_scalars(z, z)
        fields = tdse_source.node_fields(problem, field, bundle)
        result = binding.evaluate(
            fields, z=z, purpose="startup", density=bundle.density
        )
        polarisation, current = tdse_source.atomic_sources(
            fields, result, problem, bundle
        )
        source, _ = tdse_source.project_atomic_sources(
            polarisation,
            current,
            result,
            problem,
            bundle,
            reference._prefactor(bundle),
            edge=tdse_source.atomic_edge_source(result, problem, bundle),
        )
    finally:
        reference.close()
    return source


@pytest.mark.parametrize("kerr_model", ["tdse_native", "optmat_cubic"])
def test_one_evaluation_is_the_reference_s_f_and_stage(kerr_model):
    """Item 1: ``F`` bit for bit; the stage output within the one-stage budget."""
    problem, request, grid = _case(kerr_model=kerr_model)
    descriptor = tdse_descriptor(problem.route)
    field = np.array(problem.initial.field0)
    reference_binding = Unbudgeted(descriptor, 10, _law)
    expected = _reference_source(problem, request, grid, reference_binding, field, 0.0)
    binding = Unbudgeted(descriptor, 10, _law)
    backend = native.NativeBackend(problem, request, grid=grid, tdse_binding=binding)
    try:
        hook = backend._host_source
        out = np.empty(problem.grids.n_omega, dtype=np.complex128)
        hook.evaluate(field.reshape(-1), 0.0, "startup", out)
        assert hook.last["purpose"] == "startup"
    finally:
        backend.close()
    # The same shared code on the same field and position: an exact identity. What
    # each backend does after F is compared over whole runs below.
    np.testing.assert_array_equal(out, expected.reshape(-1))
    assert np.any(out != 0)


@pytest.mark.parametrize("formulation", ["interaction", "exponential"])
def test_the_call_sequence_ledgers_and_metadata_are_the_reference_s(formulation):
    """Item 2 on a normal run, PEC: every call in the same order on both."""
    runs = _pair(_case(formulation))
    reference, ours = runs["numpy_ref"], runs["native"]
    assert [call[:2] for call in ours[2].calls] == [
        call[:2] for call in reference[2].calls
    ]
    # The positions follow step proposals that are smooth functions of independently
    # rounded defect norms (a rejection's defect cancels to a few digits), so they
    # agree within an accuracy envelope, not bits; the sequence above is exact.
    zs = np.array([call[2] for call in ours[2].calls if call[0] == "evaluate"])
    zr = np.array([call[2] for call in reference[2].calls if call[0] == "evaluate"])
    np.testing.assert_allclose(zs, zr, rtol=1e-9, atol=0)
    assert ours[2].ledger == reference[2].ledger
    for key in ("atomic_calls", "tdse_source"):
        assert ours[3][key] == reference[3][key]
    assert ours[3]["tdse_last_evaluation"]["purpose"] == "predictor"
    assert ours[0].terminal.error is None and reference[0].terminal.error is None


def test_a_budget_refusal_stops_both_at_the_same_step_with_the_same_charges():
    """Item 2 and 3: a reservation the atomic budget refuses, before the attempt."""
    runs = _pair(_case(), limit=7)
    reference, ours = runs["numpy_ref"], runs["native"]
    for run, _, binding, _ in (reference, ours):
        error = run.terminal.error
        assert isinstance(error, ResponseBudgetError)
        assert "atomic-call budget cannot reserve" in str(error)
        assert binding.ledger.reserved == 0
    assert ours[2].calls == reference[2].calls
    assert ours[0].terminal.error.z == reference[0].terminal.error.z
    assert ours[0].terminal.error.calls == reference[0].terminal.error.calls


@pytest.mark.parametrize("binding_class", [Recording, Interrupted])
def test_a_failed_evaluation_keeps_the_last_accepted_state(binding_class):
    """Item 3: a provider failure, or a signal at the boundary, before commit."""
    runs = _pair(_case(), binding_class, fail_evaluation=12)
    reference, ours = runs["numpy_ref"], runs["native"]
    positions = []
    for run, _, binding, _ in (reference, ours):
        if binding_class is Recording:
            assert isinstance(run.terminal.error, BackendError)
            positions.append(run.terminal.z)
        elif isinstance(run, RunTerminated):
            # The reference's session lets the signal through to the run loop...
            pass
        else:
            # ...native's reports it as its terminal error, the batch booked and the
            # state eligible for the checkpoint; the run loop raises it the same.
            assert isinstance(run.terminal.error, RunTerminated)
            assert run.terminal.eligibility.value == "eligible"
            positions.append(run.terminal.z)
        assert binding.ledger.reserved == 0
        # The failed evaluation's trial was not recorded; its reservation released.
        assert binding.calls[-1] == ("release",)
    # The last accepted step's end, to the steps' independent rounding (see above).
    assert all(position > 0 for position in positions)
    if len(positions) == 2:
        assert positions[1] == pytest.approx(positions[0], rel=1e-9, abs=0)
    assert [call[:2] for call in ours[2].calls] == [
        call[:2] for call in reference[2].calls
    ]
    assert ours[2].ledger == reference[2].ledger


def test_an_archive_failure_after_an_accepted_commit_keeps_the_new_state():
    """Item 3: the commit is the boundary; the archive's error is the run's."""
    runs = _pair(_case(), fail_record=2)
    reference, ours = runs["numpy_ref"], runs["native"]
    for run, _, binding, _ in (reference, ours):
        assert isinstance(run.terminal.error, TDSEArchiveError)
        assert binding.ledger.reserved == 0
        assert binding.calls[-1] == ("release",)
    # The second accepted step stands: the run stops at its end, on both.
    assert ours[0].terminal.z == reference[0].terminal.z
    assert [call[:2] for call in ours[2].calls] == [
        call[:2] for call in reference[2].calls
    ]


@pytest.mark.parametrize("work_limit", [1, 256])
def test_a_short_adaptive_run_agrees_within_its_per_step_budget(work_limit):
    """Item 4: one attempt per call and the production bound alike.

    The case's collar holds its steps to about a two-hundredth of its length, so a
    run of 0.2 um is some four hundred attempts, the same on both backends: each step
    is one stage of the one-step budget, 16 eps (tests/bounds.py's rung), accumulated
    linearly, the in-stage absorber damping rather than amplifying. Measured 1.3e-13
    on an Apple M5 Max against a bound of 1.4e-12.
    """
    problem, request, grid = _case(length=2e-7, dz_max=2e-7)
    descriptor = tdse_descriptor(problem.route)
    fields, ledgers = {}, {}
    for name, construct in (
        ("numpy_ref", NumpyRefBackend),
        ("native", native.NativeBackend),
    ):
        binding = ManufacturedBinding(descriptor, 10_000, _law)
        backend = construct(problem, request, grid=grid, tdse_binding=binding)
        try:
            run, _ = _run(backend, problem, request, work_limit=work_limit)
        finally:
            backend.close()
        assert run.terminal.error is None
        fields[name] = run.fields[-1]
        ledgers[name] = (binding.ledger, run.status.attempts, run.status.rejected)
    # The same attempts and rejections, so no decision forked on this case.
    assert ledgers["native"] == ledgers["numpy_ref"]
    attempts = ledgers["native"][1]
    assert attempts > 8
    assert_within(
        fields["native"], fields["numpy_ref"], attempts * 16 * EPS, what="short run"
    )


def test_a_device_build_refuses_the_tdse_response_by_name(monkeypatch):
    """Item 6, by the refusal's rule: no device build runs a host source."""
    problem, request, grid = _case()
    monkeypatch.setattr(native.backend, "_on_device", lambda: True)
    binding = ManufacturedBinding(tdse_descriptor(problem.route), 10, _law)
    with pytest.raises(BackendError, match="runs on the native CPU build"):
        native.NativeBackend(problem, request, grid=grid, tdse_binding=binding)


def test_a_tdse_problem_needs_its_binding_and_a_binding_its_problem():
    problem, request, grid = _case()
    with pytest.raises(BackendError, match="explicit host response binding"):
        native.NativeBackend(problem, request, grid=grid)


def test_the_memory_ledger_lists_the_host_rows_and_the_binding_s():
    problem, request, grid = _case()
    binding = ManufacturedBinding(tdse_descriptor(problem.route), 10, _law)
    binding.memory_ledger = lambda: {"transport.solprop": 64}
    backend = native.NativeBackend(problem, request, grid=grid, tdse_binding=binding)
    try:
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    n = problem.grids.n_omega
    assert ledger["host_source"] == 3 * 16 * n
    assert ledger["tdse_transport.solprop"] == 64
