# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The TDSE response on native against the reference (SEP 49 section 7.3).

Manufactured bindings replace the driver's binding, not the provider, and need no
worker: ``edge_source_cases.ManufacturedBinding`` returns a manufactured remainder
beside a large linear shadow. Native evaluates the source through its host source
hook, the reference through its own evaluation; both call the shared assembly
(``solprop.backends.tdse_source``), so ``F`` agrees bit for bit for the same field and
position, and what each backend does after it is compared within its budget. The
remainder is strong enough that ``F`` moves every compared field seven orders or more
above its budget, and each budgeted comparison checks that it does.
"""

import dataclasses
import gc
import math
import signal
import weakref
from typing import Any, cast

import numpy as np
import pytest

from bounds import EPS, assert_within, fft_bound
from edge_source_cases import ManufacturedBinding
from sessions.executors import IDENTITY, Started, drive
from solprop import _interrupts
from solprop.backends import tdse_source
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.multistep import HostHistory
from solprop.backends.session import Start
from solprop.backends.sessions import start_session
from solprop.driver import Simulation
from solprop.exceptions import (
    BackendError,
    ConvergenceError,
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
    """A manufactured remainder ``S(t)``, A m per atom.

    Nine orders above the accounting fixtures' 1e-27 A m, so that ``F`` changes a run's
    field far above any budget here: a source that never reached the field would fail.
    """
    return 1e-18 * np.exp(-((times / 8e-15) ** 2)) * np.sin(2.4e15 * times)


def _silent(times):
    """No remainder: the same binding, its source zero."""
    return np.zeros_like(times)


def _case(
    formulation: Any = "interaction",
    kerr_model: Any = "tdse_native",
    *,
    length=LENGTH,
    noise=None,
    **changes,
):
    """A short helium capillary with a manufactured TDSE response."""
    dz_max = changes.pop("dz_max", 1e-6)
    dz_init = changes.pop("dz_init", min(1e-6, dz_max))
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
        noise=noise,
        seed=None if noise is None else 7,
        plasma=False,
        backend="numpy_ref",
        integrator=Multistep(formulation=formulation),
        dz_init=dz_init,
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
        noise=parent.noise,
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

    def __init__(
        self, descriptor, limit, *, fail_evaluation=None, fail_record=None, signal=False
    ):
        super().__init__(descriptor, limit, _law)
        self.signal = signal
        self.raised = None
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
            if self.signal:
                self.held()
            raise self.failure()
        return super().evaluate(fields, z=z, purpose=purpose, density=density)

    def failure(self) -> BaseException:
        return BackendError("manufactured provider failure")

    def record_trial(self, *, accepted):
        self.calls.append(("record_trial", accepted))
        commits = self.calls.count(("record_trial", True))
        if accepted and self.fail_record is not None and commits == self.fail_record:
            raise TDSEArchiveError("manufactured archive failure")

    def release_reservation(self):
        self.calls.append(("release",))
        super().release_reservation()

    def held(self) -> None:
        """Take a real SIGTERM, then raise it as the worker's wait does between polls.

        The signal arrives inside the session's commit, which holds it; only the
        session's ``interruptible`` scope lets ``raise_held`` raise it here.
        """
        signal.raise_signal(signal.SIGTERM)
        try:
            _interrupts.raise_held()
        except RunTerminated as error:
            self.raised = error
            raise


class ReleaseFails(Recording):
    """Fail the first release, after releasing."""

    def release_reservation(self):
        super().release_reservation()
        if self.calls.count(("release",)) == 1:
            raise BackendError("manufactured release failure")


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


def _unsourced(case, budget, field, *, work_limit=256):
    """Assert that `field` is far from the run with no source: the comparison holds `F`.

    - ``budget``: the comparison's relative bound, which the source's effect must
      exceed a millionfold.
    """
    problem, request, grid = case
    binding = ManufacturedBinding(tdse_descriptor(problem.route), 10**6, _silent)
    backend = native.NativeBackend(problem, request, grid=grid, tdse_binding=binding)
    try:
        run, _ = _run(backend, problem, request, work_limit=work_limit)
    finally:
        backend.close()
    effect = np.linalg.norm(field - run.fields[-1]) / np.linalg.norm(field)
    assert effect > 1e6 * budget, f"the source moves the field by {effect:.3g} only"


def _reference_startup(case, monkeypatch):
    """The reference's startup: the field it evaluates, its ``F`` and row zero."""
    problem, request, grid = case
    seen: dict[str, Any] = {}
    nonlinear = NumpyRefBackend._nonlinear_tdse
    project = tdse_source.project_atomic_sources
    start = HostHistory.start

    def recording_nonlinear(self, lab, bundle, inverse, purpose):
        seen.setdefault("field", np.array(lab).reshape(-1))
        seen.setdefault("z", bundle.z1)
        return nonlinear(self, lab, bundle, inverse, purpose)

    def recording_project(*args, **kwargs):
        result = project(*args, **kwargs)
        seen.setdefault("source", np.array(result[0]).reshape(-1))
        return result

    def recording_start(self, z):
        row = start(self, z)
        seen.setdefault("row", np.array(row).reshape(-1))
        seen.setdefault("history", self)
        return row

    with monkeypatch.context() as patch:
        patch.setattr(NumpyRefBackend, "_nonlinear_tdse", recording_nonlinear)
        patch.setattr(tdse_source, "project_atomic_sources", recording_project)
        patch.setattr(HostHistory, "start", recording_start)
        binding = Unbudgeted(tdse_descriptor(problem.route), 10**6, _law)
        backend = NumpyRefBackend(problem, request, grid=grid, tdse_binding=binding)
        session = start_session(
            backend, problem, request, Start.entrance(problem), owns=False
        )
        try:
            while session.status().attempts < 1:
                batch = session.advance(work_limit=1)
                if batch.receipt is not None:
                    session.acknowledge(batch.receipt)
            seen["after"] = np.array(seen["history"]._store[0]).reshape(-1)
            status = session.status()
            seen["attempts"], seen["position"] = status.attempts, status.z
        finally:
            session.close()
            backend.close()
    return seen


def _native_startup(case):
    """Native's startup through the C++ gather and the binding: the hook's field and
    source, and the history's row zero after the first attempt."""
    problem, request, grid = case
    binding = Unbudgeted(tdse_descriptor(problem.route), 10**6, _law)
    backend = native.NativeBackend(problem, request, grid=grid, tdse_binding=binding)
    seen: dict[str, Any] = {}
    hook = backend._host_source
    evaluate = hook.evaluate

    def recording(field, z, purpose, out):
        evaluate(field, z, purpose, out)
        if "field" not in seen:
            seen.update(
                field=np.array(field),
                writable=field.flags.writeable,
                source=np.array(out),
                z=z,
                purpose=purpose,
            )

    # The engine looks the entry point up on the object at each call.
    hook.evaluate = recording
    session = start_session(
        backend, problem, request, Start.entrance(problem), owns=False
    )
    try:
        raw = cast("Any", session)._native
        batch = raw.advance(1, math.inf)
        while raw.status().attempts < 1:
            raw.acknowledge(batch.sequence)
            batch = raw.advance(1, math.inf)
        seen["attempts"], seen["position"] = raw.status().attempts, raw.status().z
        snapshot = raw.march_snapshot(batch.sequence, raw.status().generation)
        assert snapshot is not None
        seen["row"] = np.array(snapshot[2]["store"][0]).reshape(-1)
    finally:
        session.close()
        backend.close()
    return seen


@pytest.mark.parametrize("noise", [None, "input"])
@pytest.mark.parametrize("kerr_model", ["tdse_native", "optmat_cubic"])
@pytest.mark.parametrize("formulation", ["interaction", "exponential"])
def test_the_startup_evaluation_is_the_reference_s(
    formulation, kerr_model, noise, monkeypatch
):
    """Item 1: the field the engine gathers and ``F`` bit for bit, through the C++
    gather and the binding; the history's first row after the first attempt bit for
    bit without the in-stage absorber, within one transform's bound with it."""
    for absorber in (False, True):
        changes: dict[str, Any] = {} if absorber else {"absorber": None}
        case = _case(formulation, kerr_model, noise=noise, **changes)
        expected = _reference_startup(case, monkeypatch)
        ours = _native_startup(case)
        assert ours["purpose"] == "startup"
        assert ours["z"] == expected["z"]
        assert not ours["writable"]
        np.testing.assert_array_equal(ours["field"], expected["field"])
        np.testing.assert_array_equal(ours["source"], expected["source"])
        assert np.any(ours["source"] != 0)
        if not absorber:
            # The first attempt, the initial step, is accepted on both: row zero is
            # the startup output, carried to the step's end in the interaction
            # formulation, kept as it is in the exponential one.
            assert ours["attempts"] == expected["attempts"] == 1
            assert ours["position"] == expected["position"] > 0
            if formulation == "exponential":
                np.testing.assert_array_equal(expected["after"], expected["row"])
            np.testing.assert_array_equal(ours["row"], expected["after"])
        else:
            # The collar rejects the initial step: row zero is the startup output.
            assert ours["position"] == case[0].initial.z
            peak = np.max(np.abs(expected["row"]))
            assert_within(
                ours["row"],
                expected["row"],
                fft_bound(case[0].grids.n_to) * peak,
                elementwise=True,
                what="startup row with the in-stage absorber",
            )


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
    # The fields the sequences came from, within the per-attempt budget (below).
    budget = ours[0].status.attempts * 16 * EPS
    assert_within(ours[0].fields[-1], reference[0].fields[-1], budget, what="run")
    _unsourced(_case(formulation), budget, ours[0].fields[-1])


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


def test_an_abandoned_attempt_s_failed_release_is_a_note_on_the_run_s_error():
    """Section 6.3: the step-size floor abandons the first attempt, whose release
    fails; that failure rides on the floor's error as a note, on both."""
    case = _case(length=1e-2, dz_max=1e-4, dz_init=1e-4, dz_min=9e-5, absorber=None)
    runs = _pair(case, ReleaseFails)
    errors = [runs[name][0].terminal.error for name in ("numpy_ref", "native")]
    assert isinstance(errors[0], ConvergenceError)
    assert type(errors[1]) is type(errors[0])
    assert str(errors[1]) == str(errors[0])
    assert (
        errors[1].__notes__
        == errors[0].__notes__
        == ["releasing the source reservation failed: manufactured release failure"]
    )
    assert runs["native"][2].calls == runs["numpy_ref"][2].calls


def _held_signal_run(case) -> tuple[float, weakref.ref]:
    """Run `case` on native with a real SIGTERM held in an evaluation; check the stop.

    Returns the run's last position and a weak reference to its binding, which the
    signal's exception reaches through its traceback (the binding keeps it).
    """
    problem, request, grid = case
    binding = Recording(
        tdse_descriptor(problem.route), 1000, fail_evaluation=12, signal=True
    )
    backend = native.NativeBackend(problem, request, grid=grid, tdse_binding=binding)
    try:
        with _interrupts.handled():
            if signal.getsignal(signal.SIGTERM) is not _interrupts._handle:
                pytest.skip("SIGTERM has a handler of its own in this process")
            run, _ = _run(backend, problem, request)
    finally:
        backend.close()
    # Raised by the evaluation's own `raise_held`, not after the commit ended.
    assert isinstance(binding.raised, RunTerminated)
    assert run.terminal.error is binding.raised
    assert run.terminal.eligibility.value == "eligible"
    assert binding.ledger.reserved == 0
    assert binding.calls[-1] == ("release",)
    return run.terminal.z, weakref.ref(binding)


def test_a_held_signal_stops_a_native_evaluation_before_its_commit():
    """Item 3 and section 6.7: a real SIGTERM inside the session's commit is held, and
    the session's ``interruptible`` scope lets the evaluation's wait raise it.

    The closed backend leaves nothing the collector cannot see: the binding, in a
    cycle through the kept exception's traceback, is freed.
    """
    case = _case()
    z, binding = _held_signal_run(case)
    gc.collect()
    assert binding() is None
    # The last accepted step, as a provider failure there leaves it.
    failed = _pair(case, fail_evaluation=12)["native"][0]
    assert z == failed.terminal.z > 0


def test_an_archive_failure_after_an_accepted_commit_keeps_the_new_state():
    """Item 3: the commit is the boundary; the archive's error is the run's, and the
    terminal state is the committed step's, its next proposal included."""
    problem, request, grid = _case()
    descriptor = tdse_descriptor(problem.route)
    ends = {}
    for name, construct in (
        ("numpy_ref", NumpyRefBackend),
        ("native", native.NativeBackend),
    ):
        binding = Recording(descriptor, 1000, fail_record=2)
        backend = construct(problem, request, grid=grid, tdse_binding=binding)
        snapshots = []

        def keep(session, batch, snapshots=snapshots):
            if batch.terminal is not None:
                snapshots.append(session.march_snapshot(batch.terminal.ticket))

        session = start_session(
            backend, problem, request, Start.entrance(problem), owns=False
        )
        try:
            run = drive(Started(session), export=IDENTITY, raises=False, between=keep)
        finally:
            session.close()
            backend.close()
        assert isinstance(run.terminal.error, TDSEArchiveError)
        assert run.terminal.eligibility.value == "eligible"
        assert binding.ledger.reserved == 0
        assert binding.calls[-2:] == [("record_trial", True), ("release",)]
        ends[name] = (run.terminal.z, snapshots[0], binding.calls)
        assert run.terminal.history is not None
        step = float(run.terminal.history.dz[-1])
    (z, snapshot, calls), (reference_z, expected, reference_calls) = (
        ends["native"],
        ends["numpy_ref"],
    )
    # The second accepted step stands: the run stops at its end, on both.
    assert z == pytest.approx(reference_z, rel=1e-12, abs=0) and z > 0
    assert snapshot.z == z
    # The committed step's own proposal, not the one it was taken with.
    assert snapshot.proposal == pytest.approx(expected.proposal, rel=1e-9, abs=0)
    assert expected.proposal != pytest.approx(step, rel=1e-6, abs=0)
    assert [call[:2] for call in calls] == [call[:2] for call in reference_calls]


def _agreement(case, work_limit):
    """Run `case` on both backends; return the native and reference final fields and
    the attempts each took, which must be the same."""
    problem, request, grid = case
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
    return fields["native"], fields["numpy_ref"], ledgers["native"][1]


@pytest.mark.parametrize("work_limit", [1, 256])
def test_eight_steps_agree_within_the_eight_step_budget(work_limit):
    """Item 4: eight steps, one attempt per call and the production bound.

    Without the in-stage absorber's collar the steps reach their cap. Cap and length
    are binary-exact (2^-27 m, 2^-24 m), so eight steps of the cap land on the end
    with no split final step: each one stage of the one-step budget, 16 eps,
    accumulated linearly to 128 eps.
    """
    case = _case(length=2.0**-24, dz_max=2.0**-27, absorber=None)
    ours, reference, attempts = _agreement(case, work_limit)
    # Eight accepted steps; the startup sample is evaluated inside the first.
    assert attempts == 8
    assert_within(ours, reference, 128 * EPS, what="eight steps")
    _unsourced(case, 128 * EPS, ours, work_limit=work_limit)


def test_a_collar_held_run_agrees_within_its_per_attempt_budget():
    """Item 4 with the in-stage absorber, whose collar holds the steps to about a
    two-hundredth of the run: some four hundred attempts, each one stage of 16 eps,
    accumulated linearly, the absorber damping rather than amplifying."""
    case = _case(length=2e-7, dz_max=2e-7)
    ours, reference, attempts = _agreement(case, 256)
    assert attempts > 200
    budget = attempts * 16 * EPS
    assert_within(ours, reference, budget, what="collar-held run")
    _unsourced(case, budget, ours)


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
    # The field and source rows, each one complex128 spectrum.
    assert ledger["host_source"] == 2 * 16 * n
    assert ledger["tdse_transport.solprop"] == 64


def test_closing_the_backend_under_an_open_session_drops_the_host_source():
    """Closing while a session leases the state closes it, host source and all."""
    problem, request, grid = _case()
    binding = ManufacturedBinding(tdse_descriptor(problem.route), 10, _law)
    backend = native.NativeBackend(problem, request, grid=grid, tdse_binding=binding)
    session = start_session(
        backend, problem, request, Start.entrance(problem), owns=False
    )
    try:
        backend.close()
        assert backend.closed
        backend.close()
    finally:
        session.close()
