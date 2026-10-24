# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Backend construction, refusals, metadata, and persistent-state accounting."""

from __future__ import annotations

import copy
import dataclasses
import functools
import math
from types import SimpleNamespace
from typing import NoReturn, cast

import numpy as np
import numpy.typing as npt
import pytest
from bounds import EPS, assert_within, step_bound
from conftest import PhaseACase, RamanCase
from solprop.backends.driver import StepDriver
from solprop.backends.low_storage import schedule_key
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import Simulation
from solprop.exceptions import BackendError
from solprop.grid import EnvelopeGrid, FreeGrid, RadialGrid
from solprop.noise import NoiseModel
from solprop.nonlinear.responses import DispersiveKerrResponse, KerrResponse
from solprop.physdata.nonlinear_data import OptmatKerr
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, PIController
from solprop.stepper.multistep import Multistep
from solprop.translate import masks
from solprop.translate.coupled import CoupledFlowTables
from solprop.translate.linop import delta_phi
from solprop.translate.problem import Flags, KernelProblem, assemble
from solprop.translate.stage import terms

import solprop_native
from solprop_native import NativeBackend, _core, build_info, check_build
from solprop_native.backend import (
    _native_problem,  # pyright: ignore[reportPrivateUsage]
    _ResidentLowStorage,  # pyright: ignore[reportPrivateUsage]
)


def test_check_build_accepts_the_compiled_contract(phase_a_case: PhaseACase) -> None:
    problem, _ = phase_a_case
    assert check_build() is None
    assert check_build(problem) is None


def test_rk_source_dispatch_counter_resets_and_excludes_fsal_reuse(
    phase_a_case: PhaseACase,
) -> None:
    """SEP 30 counts executed stages rather than inferred accepted-step work."""
    problem, grid = phase_a_case
    target = float(problem.stepper.save_planes[-1])
    stepper = dataclasses.replace(
        problem.stepper,
        integrator=None,
        auto_init=False,
        schedule=FixedSchedule.uniform(0.0, target, target / 2),
    )
    configured = dataclasses.replace(problem, stepper=stepper)
    backend = NativeBackend(configured, grid=grid)
    try:
        backend.set_initial(configured.initial.field0, None)
        assert backend.metadata()["source_evaluations"] == 0
        backend.propagate_to(target / 2)
        assert backend.metadata()["source_evaluations"] == 7
        backend.propagate_to(target)
        assert backend.metadata()["source_evaluations"] == 13
        backend.set_initial(configured.initial.field0, None)
        assert backend.metadata()["source_evaluations"] == 0
    finally:
        backend.close()


@pytest.mark.parametrize("missing", ["phi", "prefactor"])
def test_native_modal_payload_requires_both_fixed_coefficient_arrays(
    phase_a_case: PhaseACase, missing: str
) -> None:
    """The separable representation must not relax ordinary modal validation."""
    problem, grid = phase_a_case
    backend = NativeBackend(problem, grid=grid)
    try:
        terms = backend._terms  # pyright: ignore[reportPrivateUsage]
        damaged = dataclasses.replace(
            terms,
            phi=None if missing == "phi" else terms.phi,
            prefactor=None if missing == "prefactor" else terms.prefactor,
        )
        with pytest.raises(BackendError, match="device-ready fixed-term problem"):
            _native_problem(problem, damaged)
    finally:
        backend.close()


def test_spatial_statistics_require_geometry(phase_a_case: PhaseACase) -> None:
    """Refuse a spatial reduction on an ordinary modal state before native entry."""
    problem, grid = phase_a_case
    backend = NativeBackend(problem, grid=grid)
    try:
        with pytest.raises(BackendError, match="require spatial geometry"):
            backend._spatial_statistics({"energy"})  # pyright: ignore[reportPrivateUsage]
    finally:
        backend.close()


def test_spatial_statistics_refuse_a_malformed_native_result(
    phase_a_case: PhaseACase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check the native packed-result boundary independently of native reductions."""
    original, grid = phase_a_case
    problem = assemble(
        grid,
        None,
        [],
        np.zeros((8, 1, grid.n_omega), complex),
        spatial_grid=RadialGrid(1e-3, 8),
        medium=SpatialMedium(),
        controller=original.stepper.controller,
        save_planes=[1e-3],
        dz_max=1e-3,
        dz_init=1e-4,
    )
    backend = NativeBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)

        def malformed_statistics(_flags: int, _factor: float, _z: float) -> list[float]:
            return [0.0, 0.0]

        malformed = SimpleNamespace(spatial_statistics=malformed_statistics)
        with monkeypatch.context() as patch:
            patch.setattr(backend, "_native_state", malformed)
            with pytest.raises(BackendError, match="returned 2 values; consumed 1"):
                backend.stats(StatsSpec(names=("energy",)))
    finally:
        backend.close()


def test_a_solprop_without_a_driver_member_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seam is the driver surface and the bundles, not solprop's version string."""
    monkeypatch.delattr(StepDriver, "_hoist")
    with pytest.raises(BackendError, match=r"has no StepDriver\._hoist"):
        check_build()


def test_stage_observation_owns_its_host_copy_and_checks_lifetime(
    phase_a_case: PhaseACase,
) -> None:
    """A stage derivative is observed directly without exposing persistent views."""
    problem, grid = phase_a_case
    backend = NativeBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    state = backend._native_state  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RuntimeError, match="stage"):
        state.stage_to_host(0)
    backend._trial_step(0.0, 1e-4)  # pyright: ignore[reportPrivateUsage]
    derivative = state.stage_to_host(0)
    assert derivative.shape == (1, grid.n_omega)
    assert derivative.dtype == np.complex128
    assert np.max(np.abs(derivative)) > 0
    saved = derivative.copy()
    derivative.fill(0)
    # Exact: a second host copy of the same device stage, not a recomputation.
    np.testing.assert_array_equal(state.stage_to_host(0), saved)
    with pytest.raises(IndexError, match="stage"):
        state.stage_to_host(problem.stepper.tableau.n_stages)
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        state.stage_to_host(0)


def test_a_stage_bundle_without_a_packed_field_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        solprop_native,
        "STAGE_SCALARS_FIELDS",
        (*solprop_native.STAGE_SCALARS_FIELDS, "gone"),
    )
    with pytest.raises(BackendError, match=r"has no StageScalars\.gone"):
        check_build()


def test_the_solprop_version_is_recorded_and_never_compared(
    phase_a_case: PhaseACase, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, grid = phase_a_case

    def elsewhere(name: str) -> str:
        return f"{name}-elsewhere"

    monkeypatch.setattr(solprop_native.backend, "version", elsewhere)
    # The build facts are read once per process; read them afresh under the patch,
    # through a fresh cache the patch removes again.
    facts = solprop_native.backend._build_facts  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(
        solprop_native.backend, "_build_facts", functools.cache(facts.__wrapped__)
    )
    assert check_build(problem) is None
    backend = NativeBackend(problem, grid=grid)
    try:
        record = backend.metadata()
    finally:
        backend.close()
    assert record["solprop_version"] == "solprop-elsewhere"
    assert record["solprop_version_built_against"] != "solprop-elsewhere"


@pytest.mark.parametrize(
    "declared", [2, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 34]
)
def test_schema_mismatch_names_both_versions(
    phase_a_case: PhaseACase, declared: int
) -> None:
    # The two independently qualified schema-18 branches are not the combined seam.
    problem, _ = phase_a_case
    spoiled = dataclasses.replace(problem, schema_version=declared)
    with pytest.raises(BackendError, match=rf"version 35.*version {declared}"):
        check_build(spoiled)


@pytest.mark.parametrize("mode", ["tabulate", "recompute"])
def test_the_low_storage_march_steps_as_the_reference(
    phase_a_case: PhaseACase, mode: str
) -> None:
    # solprop SEP 24 phase 3: the resident engine, one native call a step, against
    # the reference's host engine on the same schedule.
    problem, grid = phase_a_case
    target = float(problem.stepper.save_planes[-1])
    steps = 4
    stepper = dataclasses.replace(
        problem.stepper,
        integrator=LowStorageRungeKutta(),
        schedule=FixedSchedule.uniform(0.0, target, target / steps),
        propagator_mode=mode,
    )
    marched = dataclasses.replace(problem, stepper=stepper)
    found: list[npt.NDArray[np.complex128]] = []
    for backend_class in (NumpyRefBackend, NativeBackend):
        backend = backend_class(marched, grid=grid)
        backend.set_initial(marched.initial.field0, None)
        backend.propagate_to(target)
        found.append(backend.observe().require_field())
        if backend_class is NativeBackend:
            ledger = backend.memory_ledger()
            assert ledger["stages"] == ledger["advanced"] == 0
            assert backend.metadata()["source_evaluations"] == 4 * steps
        backend.close()
    reference, field = found
    # Four transform round trips a step (no stage combination: one register read),
    # and the flows' exponents, formed from the table or the phase terms against the
    # reference's propagator: a few roundings of the largest phase a step sweeps.
    swept = float(np.max(np.abs(delta_phi(marched.linear, 0.0, target))))
    bound = step_bound(grid.n_to, steps, stages=4) + 64 * EPS * swept
    assert_within(field, reference, bound, what=f"the low-storage march, {mode}")


def test_a_low_storage_table_never_built_is_refused(phase_a_case: PhaseACase) -> None:
    # solprop SEP 24's closing guard: the native state refuses to reuse a constant
    # operator's table it has not built since the field was loaded, rather than step
    # with unwritten flows. solprop's engine keys the table from NaN after every load,
    # so only a stale key reaches it: forged here.
    problem, grid = phase_a_case
    target = float(problem.stepper.save_planes[-1])
    stepper = dataclasses.replace(
        problem.stepper,
        integrator=LowStorageRungeKutta(),
        schedule=FixedSchedule.uniform(0.0, target, target / 4),
        propagator_mode="tabulate",
    )
    marched = dataclasses.replace(problem, stepper=stepper)
    backend = NativeBackend(marched, grid=grid)
    try:
        backend.set_initial(marched.initial.field0, None)
        march = backend._low_storage  # pyright: ignore[reportPrivateUsage]
        assert march is not None
        engine = march.engine
        assert isinstance(engine, _ResidentLowStorage)
        engine._table_step = schedule_key(  # pyright: ignore[reportPrivateUsage]
            engine._keys,  # pyright: ignore[reportPrivateUsage]
            engine._edges,  # pyright: ignore[reportPrivateUsage]
            0.0,
            target / 4,
        )
        with pytest.raises(RuntimeError, match="table has not been built"):
            backend.propagate_to(target)
    finally:
        backend.close()


def test_an_adams_integrator_steps_as_the_reference(phase_a_case: PhaseACase) -> None:
    # solprop's shared Adams march (SEP 13, SEP 14 phase 2) with this backend's one
    # source evaluation a step: the reference's steps and calls, and its field to the
    # rounding of the two evaluation orders.
    problem, grid = phase_a_case
    stepper = dataclasses.replace(problem.stepper, integrator=Multistep())
    adams = dataclasses.replace(problem, stepper=stepper)
    target = float(adams.stepper.save_planes[-1])
    found: list[tuple[npt.NDArray[np.complex128], int]] = []
    for backend_class in (NumpyRefBackend, NativeBackend):
        backend = backend_class(adams, grid=grid)
        backend.set_initial(adams.initial.field0, None)
        summary = backend.propagate_to(target)
        found.append((backend.observe().require_field(), summary.n_steps))
        backend.close()
    (reference, reference_steps), (field, steps) = found
    assert steps == reference_steps
    # solprop's own parity test measures 8e-16 to 1.8e-15 of the peak.
    scale = float(np.max(np.abs(reference)))
    np.testing.assert_allclose(field, reference, rtol=0, atol=1e-12 * scale)


def test_an_adams_integrator_observes_densely_as_the_reference(
    phase_a_case: PhaseACase,
) -> None:
    # An interpolated plane (solprop ADR-036, ADR-039) lies inside a step and is loaded
    # natively for the observation; the march resumes from its own host state.
    problem, grid = phase_a_case
    stepper = dataclasses.replace(
        problem.stepper, integrator=Multistep(), interpolate_planes=True
    )
    adams = dataclasses.replace(problem, stepper=stepper)
    found: list[npt.NDArray[np.complex128]] = []
    for backend_class in (NumpyRefBackend, NativeBackend):
        backend = backend_class(adams, grid=grid)
        backend.set_initial(adams.initial.field0, None)
        fields: list[npt.NDArray[np.complex128]] = []
        for plane in adams.stepper.save_planes:
            backend.propagate_to(float(plane))
            fields.append(backend.observe().require_field())
        found.append(np.stack(fields))
        backend.close()
    reference, field = found
    scale = float(np.max(np.abs(reference)))
    np.testing.assert_allclose(field, reference, rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize("dense_output", [False, True])
@pytest.mark.parametrize("source_filter", ["none", "time_collar", "gated_derivative"])
@pytest.mark.parametrize("with_adams", [False, True])
def test_tdse_is_refused_by_name_before_integrator_or_native_setup(
    phase_a_case: PhaseACase,
    monkeypatch: pytest.MonkeyPatch,
    with_adams: bool,
    source_filter: str,
    dense_output: bool,
) -> None:
    """Refuse TDSE before native setup or either supported march is selected."""
    problem, grid = phase_a_case
    # This deliberately bypasses scientific validation to isolate the adapter's
    # earliest refusal. Neither a valid atomic basis nor an optional provider is
    # needed to reject a route that this backend never executes.
    requested = copy.copy(problem)
    object.__setattr__(requested, "tdse", SimpleNamespace(source_filter=source_filter))
    object.__setattr__(
        requested,
        "stepper",
        dataclasses.replace(
            problem.stepper,
            integrator=Multistep() if with_adams else None,
            interpolate_planes=dense_output,
        ),
    )

    def forbidden(*_arguments: object, **_keywords: object) -> NoReturn:
        raise AssertionError("unsupported TDSE reached native setup")

    monkeypatch.setattr(solprop_native.backend, "require_device_ready", forbidden)
    monkeypatch.setattr(_core, "State", forbidden)
    with pytest.raises(BackendError, match=r"TDSEGasResponse.*SEP 12"):
        NativeBackend(requested, grid=grid)


@pytest.mark.parametrize(
    "member",
    ["restore_call_ledgers", "_reserve_source_calls", "_release_source_calls"],
)
def test_schema_requires_the_inherited_response_accounting_hooks(
    monkeypatch: pytest.MonkeyPatch, member: str
) -> None:
    monkeypatch.delattr(StepDriver, member)
    with pytest.raises(BackendError, match=rf"has no StepDriver\.{member}"):
        check_build()


def test_runge_kutta_native_resume_refuses_response_ledgers(
    phase_a_case: PhaseACase,
) -> None:
    problem, grid = phase_a_case
    backend = NativeBackend(problem, grid=grid)
    try:
        backend.restore_call_ledgers({}, {})
        with pytest.raises(
            BackendError, match="logical response calls require an Adams run"
        ):
            backend.restore_call_ledgers({"startup": 1}, {})
        with pytest.raises(BackendError, match="does not restore"):
            backend.restore_call_ledgers({}, {"attempted": 1})
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("group", "leaf"),
    [
        ("linear", "coupled_flow"),
        ("linear", "modal_metric"),
        ("transform", "modal_tables"),
        ("responses", "regional_responses"),
    ],
)
def test_every_schema_14_general_modal_leaf_is_refused_by_name(
    phase_a_case: PhaseACase, group: str, leaf: str
) -> None:
    """A lone general-modal leaf on an ordinary problem is refused by name.

    The version check cannot catch these: this wheel and the problem both declare 14.
    All four leaves are executed (the general modal route and the coupled flow), in
    the combinations solprop's one
    admission list admits, which this adapter runs under its own name. A lone table,
    metric or response is refused by that list before anything is allocated; the
    coupled flow is a real leaf here, a constant-Schur section carrying two
    Chebyshev degrees, which the list refuses as the reference does.

    The sentinel is set with ``object.__setattr__`` on a copy rather than through
    ``dataclasses.replace``, which re-runs the group's own validation: solprop's
    ``Transform`` refuses a ``modal_tables`` record beside the static matrices that
    record replaces, so a ``replace`` never reaches this adapter at all. What is
    under test here is the adapter's refusal, not solprop's.
    """
    problem, grid = phase_a_case
    assert getattr(getattr(problem, group), leaf) is None
    spoiled_group = copy.copy(getattr(problem, group))
    object.__setattr__(spoiled_group, leaf, _leaf_value(problem, leaf))
    spoiled = dataclasses.replace(problem, **{group: spoiled_group})
    expected = {
        "coupled_flow": r"native executes a constant-Schur coupled flow",
        "modal_metric": r"native executes linear\.modal_metric",
        "modal_tables": r"native executes a general modal guide",
        "regional_responses": r"native executes responses\.regional_responses",
    }[leaf]
    with pytest.raises(BackendError, match=expected):
        NativeBackend(spoiled, grid=grid)


def _leaf_value(problem: KernelProblem, leaf: str) -> object:
    """Return a leaf the shared admission list refuses on this problem."""
    if leaf != "coupled_flow":
        return object()
    n_omega = problem.shape[-1]
    n_modes = problem.shape[1]
    return CoupledFlowTables(
        np.linspace(1.0e15, 2.0e15, n_omega),
        np.array([0.0, 1.0]),
        np.zeros((1, 2, n_modes, n_modes, n_omega), dtype=complex),
        None,
        None,
        None,
        None,
        None,
        np.zeros((1, n_omega)),
        np.zeros((1, n_omega)),
        np.ones((1, n_omega)),
        ("constant_schur",),
        "lab",
        "schur",
    )


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        ({"single_batch": False}, "NativeBackend owns one member"),
        ({"has_spectral_responses": True}, "spectral responses other than envelope"),
    ],
)
def test_unsupported_contracts_are_refused_by_name(
    phase_a_case: PhaseACase, flags: dict[str, bool], message: str
) -> None:
    problem, grid = phase_a_case
    changed = dataclasses.replace(problem.flags, **flags)
    spoiled = dataclasses.replace(problem, flags=changed)
    with pytest.raises(BackendError, match=message):
        NativeBackend(spoiled, grid=grid)


@pytest.mark.parametrize(
    ("evaluator", "path"),
    [
        ("symmetric", {"alone": True, "in_place": True, "borrowed": True}),
        ("polynomial", {"alone": True, "in_place": False, "borrowed": False}),
    ],
)
def test_the_dispersive_electronic_leaf_runs_on_the_free_space_route(
    phase_a_case: PhaseACase, evaluator: str, path: dict[str, bool]
) -> None:
    """solprop SEP 6's leaf runs chunk by chunk on the Cartesian route (Phase 3b).

    A lone symmetric response filters each chunk's spectrum in place and, on an
    envelope grid, its product is the chunk's own polarisation; the polynomial holds
    its products. solprop's ``test_native_dispersive_routes.py`` holds the route to
    its reference.
    """
    original, grid = phase_a_case
    space = FreeGrid(1e-3, 4, 1.5e-3, 6, window_factor=0)
    response = DispersiveKerrResponse(
        OptmatKerr("Ar").dispersive_model(), evaluator, thg=False
    )
    problem = assemble(
        grid,
        None,
        [response],
        np.zeros((24, 1, grid.n_omega), complex),
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=original.stepper.controller,
        save_planes=[1e-3],
        dz_max=1e-3,
        dz_init=1e-4,
        density=4.9e25,
    )
    assert problem.responses.dispersive_kerr is not None
    backend = NativeBackend(problem, grid=grid)
    try:
        assert backend.metadata()["dispersive_path"] == path
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    held = ledger["dispersive_products"] + ledger["dispersive_product_spectra"]
    assert (held == 0) == (evaluator == "symmetric")


def test_batch_refusal_happens_before_native_allocation(
    phase_a_case: PhaseACase,
) -> None:
    problem, grid = phase_a_case
    shape: tuple[int, ...] = (2, *problem.shape[1:])
    flags = Flags.from_content(
        shape,
        problem.linear,
        problem.transform,
        problem.responses,
        problem.initial,
    )
    spoiled = dataclasses.replace(problem, shape=shape, flags=flags)
    with pytest.raises(BackendError, match="use NativeBatchBackend"):
        NativeBackend(spoiled, grid=grid)


def test_statistics_do_not_observe_the_full_field(
    phase_a_case: PhaseACase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The statistics seam transfers reductions, not the protected state."""
    problem, grid = phase_a_case
    backend = NativeBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)

    def fail_if_observed(state: object) -> NoReturn:
        pytest.fail(f"statistics observed native state {state!r}")

    monkeypatch.setattr(
        backend,
        "_observed_field",
        fail_if_observed,
    )
    values = backend.stats(
        StatsSpec(names=("omega0", "energy", "peak_power", "fwhm_t"))
    )
    assert set(values) == {"omega0", "energy", "peak_power", "fwhm_t"}
    backend.close()


def test_raman_metadata_memoization_and_conditional_ledger(
    raman_case: RamanCase,
) -> None:
    """Raman plans are persistent and the exact ordered rate tuple is memoized."""
    problem, grid = raman_case
    backend = NativeBackend(problem, grid=grid)
    coefficients = problem.responses.raman_coefficients
    assert coefficients is not None
    rates = tuple(float(table(0.0)) for table in coefficients.decay_rates)
    backend._raman_spectrum(rates)  # pyright: ignore[reportPrivateUsage]
    assert backend._raman_transforms == 1  # pyright: ignore[reportPrivateUsage]
    backend._raman_spectrum(rates)  # pyright: ignore[reportPrivateUsage]
    assert backend._raman_transforms == 1  # pyright: ignore[reportPrivateUsage]
    backend._raman_spectrum(tuple(reversed(rates)))  # pyright: ignore[reportPrivateUsage]
    expected = 1 if len(set(rates)) == 1 else 2
    assert backend._raman_transforms == expected  # pyright: ignore[reportPrivateUsage]
    metadata = backend.metadata()
    ledger = backend.memory_ledger()
    assert metadata["raman_enabled"] is True
    assert metadata["raman_terms"] == coefficients.n_terms
    assert metadata["raman_drive"] == "cycle_averaged"
    assert cast(float, metadata["raman_planning_seconds"]) >= 0.0
    team_size = cast(int, metadata["device_team_size"])
    team_limit = cast(int, metadata["device_team_size_limit"])
    assert team_size <= team_limit
    if metadata["device_backend"] == "cuda" and problem.grids.n_to < 8192:
        assert team_size == min(team_limit, 256)
    assert ledger["raman_undamped"] == coefficients.undamped.nbytes
    assert ledger["raman_kernel_spectrum"] == (problem.grids.n_to + 1) * 16
    backend.close()


def _raman_native_payload(raman_case: RamanCase) -> dict[str, object]:
    problem, grid = raman_case
    backend = NativeBackend(problem, grid=grid)
    try:
        return _native_problem(  # pyright: ignore[reportPrivateUsage]
            problem,
            backend._terms,  # pyright: ignore[reportPrivateUsage]
        )
    finally:
        backend.close()


@pytest.mark.parametrize("spoiling", ["dtype", "rank", "contiguity"])
def test_native_raman_rows_refuse_hidden_array_conversions(
    raman_case: RamanCase, spoiling: str
) -> None:
    """The private seam accepts exact arrays and never copies a near miss."""
    payload = _raman_native_payload(raman_case)
    rows = cast(np.ndarray, payload["raman_undamped"])
    if spoiling == "dtype":
        payload["raman_undamped"] = rows.astype(np.float32)
    elif spoiling == "rank":
        payload["raman_undamped"] = rows.ravel()
    else:
        payload["raman_undamped"] = np.zeros(
            (rows.shape[0], 2 * rows.shape[1]), dtype=np.float64
        )[:, ::2]
    with pytest.raises((TypeError, ValueError)):
        _core.State(payload)


def test_native_raman_rows_accept_read_only_input(raman_case: RamanCase) -> None:
    """Construction reads Raman rows but does not require write access to them."""
    payload = _raman_native_payload(raman_case)
    rows = np.array(cast(np.ndarray, payload["raman_undamped"]), copy=True)
    rows.setflags(write=False)
    payload["raman_undamped"] = rows
    state = _core.State(payload)
    state.close()


def test_native_raman_rates_refuse_wrong_dtype_and_shape(
    raman_case: RamanCase,
) -> None:
    """The diagnostic preparation seam has the same exact-array contract as trials."""
    state = _core.State(_raman_native_payload(raman_case))
    try:
        with pytest.raises(TypeError):
            state.prepare_raman(
                np.zeros(1, dtype=np.float32)  # pyright: ignore[reportArgumentType]
            )
        with pytest.raises(ValueError, match="shape"):
            state.prepare_raman(np.zeros((1, 1), dtype=np.float64))
    finally:
        state.close()


def test_metadata_and_memory_ledger_are_self_consistent(
    phase_a_case: PhaseACase,
) -> None:
    problem, grid = phase_a_case
    backend = NativeBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    metadata = backend.metadata()
    ledger = backend.memory_ledger()
    n = problem.shape[-1]
    stages = problem.stepper.tableau.n_stages
    assert metadata["backend"] == "native"
    assert metadata["schema_version"] == 36
    assert cast(float, metadata["planning_seconds"]) >= 0.0
    # A one-mode envelope at one node. A host build folds its one-element transforms
    # into the response column (solprop SEP 25), whatever the GEMM policy. A device
    # build's forced vendor policy multiplies its real transform matrices into the
    # samples' interleaved real view; its hybrid policy keeps the custom kernel under
    # its shape guard (one mode).
    build = solprop_native.build_info()
    if build.device_backend == "none":
        expected_provider = "folded"
    elif build.gemm_provider == "KOKKOSBLAS_TPL":
        expected_provider = (
            str(metadata["synthesis_provider"]).removesuffix("-interleaved-real")
            + "-interleaved-real"
        )
        assert str(metadata["synthesis_provider"]).endswith("-interleaved-real")
    else:
        expected_provider = "custom"
    assert metadata["synthesis_provider"] == expected_provider
    assert metadata["projection_provider"] == expected_provider
    assert "every whole transform on the host" in cast(
        str, metadata["gemm_shape_policy"]
    )
    team_size = cast(int, metadata["device_team_size"])
    team_limit = cast(int, metadata["device_team_size_limit"])
    assert team_size <= team_limit
    if metadata["device_backend"] == "cuda" and problem.grids.n_to < 8192:
        assert team_size == min(team_limit, 256)
    else:
        assert team_size == team_limit
    assert ledger["state"] == n * np.dtype(np.complex128).itemsize
    assert ledger["advanced"] == ledger["state"]
    assert ledger["stages"] == stages * ledger["state"]
    assert ledger["error_partials"] == 3 * np.dtype(np.float64).itemsize
    assert all(value >= 0 for value in ledger.values())
    backend.close()


def test_build_refuses_a_missing_separable_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        solprop_native,
        "SEPARABLE_OPERATOR_FIELDS",
        (*solprop_native.SEPARABLE_OPERATOR_FIELDS, "missing_residual"),
    )
    with pytest.raises(
        BackendError, match=r"has no SeparableOperator\.missing_residual"
    ):
        solprop_native.check_build()


@pytest.mark.parametrize(
    ("tuple_name", "bundle"),
    [
        ("LINEAR_OPERATOR_FIELDS", "LinearOperator"),
        ("TRANSFORM_FIELDS", "Transform"),
        ("RESPONSE_FIELDS", "CompiledResponses"),
        ("MODAL_TABLES_FIELDS", "ModalTransformTables"),
        ("REGIONAL_RESPONSE_FIELDS", "RegionalResponseTables"),
        ("COUPLED_FLOW_FIELDS", "CoupledFlowTables"),
        ("DISPERSIVE_KERR_FIELDS", "DispersiveKerrTerms"),
    ],
)
def test_build_refuses_a_renamed_general_modal_leaf(
    monkeypatch: pytest.MonkeyPatch, tuple_name: str, bundle: str
) -> None:
    """A renamed leaf is named by `check_build`, not an `AttributeError` later.

    Reading `problem.linear.coupled_flow` on a solprop that renamed it raises an
    `AttributeError` from inside the adapter's refusal, with no useful context; the
    frozen name is looked up once, at build check, and reported with the member.
    """
    monkeypatch.setattr(
        solprop_native,
        tuple_name,
        (*getattr(solprop_native, tuple_name), "renamed_leaf"),
    )
    with pytest.raises(BackendError, match=rf"has no {bundle}\.renamed_leaf"):
        solprop_native.check_build()


def test_the_leaf_member_tuples_name_what_the_installed_solprop_has() -> None:
    """Each nested member the adapter reads exists as a field or property today.

    `check_build` passing already says so; this names the three tuples and the classes
    they guard, so a tuple entry that no class has fails here
    by name before any backend is built.
    """
    from solprop.translate.coupled import CoupledFlowTables
    from solprop.translate.modal_tables import ModalTransformTables
    from solprop.translate.responses import DispersiveKerrTerms, RegionalResponseTables

    for bundle, wanted in (
        (ModalTransformTables, solprop_native.MODAL_TABLES_FIELDS),
        (RegionalResponseTables, solprop_native.REGIONAL_RESPONSE_FIELDS),
        (CoupledFlowTables, solprop_native.COUPLED_FLOW_FIELDS),
        (DispersiveKerrTerms, solprop_native.DISPERSIVE_KERR_FIELDS),
    ):
        fields = {field.name for field in dataclasses.fields(bundle)}
        for name in wanted:
            assert name in fields or isinstance(getattr(bundle, name), property), (
                f"{bundle.__name__}.{name}"
            )


@pytest.mark.parametrize("kind", ["radial", "free"])
def test_temporal_statistics_are_refused_on_a_spatial_problem(
    phase_a_case: PhaseACase, kind: str
) -> None:
    """Mirror numpy_ref before native entry: a peak power has no meaning on Hankel
    or DFT columns, and the Cartesian native statistics buffers hold one node
    chunk."""
    original, grid = phase_a_case
    space: RadialGrid | FreeGrid = (
        RadialGrid(1e-3, 8)
        if kind == "radial"
        else FreeGrid(1e-3, 4, 1.5e-3, 6, window_factor=0)
    )
    problem = assemble(
        grid,
        None,
        [],
        np.zeros((space.n_columns, 1, grid.n_omega), complex),
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=original.stepper.controller,
        save_planes=[1e-3],
        dz_max=1e-3,
        dz_init=1e-4,
    )
    backend = NativeBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        for name in ("peak_power", "fwhm_t", "energy_window", "omega0"):
            with pytest.raises(BackendError, match="not a spatial saved-field"):
                backend.stats(StatsSpec(names=(name,)))
        assert np.all(backend.stats(StatsSpec(names=("energy",)))["energy"] == 0)
    finally:
        backend.close()


def test_packed_real_blas_radial_envelope_matches_the_reference_when_eligible() -> None:
    """The packed real GEMM is production policy for whole scalar radial envelopes of
    at least 256 nodes and 256 fine samples on HYBRID/TPL host builds; pin its
    selection, its ledger row and its agreement with the reference at that
    shape."""
    grid = EnvelopeGrid.from_limits(1e-4, 800e-9, (600e-9, 1.2e-6), 300e-15, thg=False)
    assert grid.n_to == 256
    space = RadialGrid(1e-3, 256)
    start = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), 1e-4
    ).build(grid, space, medium=SpatialMedium())
    problem = assemble(
        grid,
        None,
        [KerrResponse(1e-22, thg=False)],
        start,
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1e-4],
        dz_max=1e-4,
        dz_init=1e-5,
        density=1.0,
    )
    assert problem.transform.node_chunk == problem.transform.n_nodes
    eps = np.finfo(float).eps
    # The ladder bound of the paired solprop tests: one fine FFT depth and two
    # length-n_r radial reductions per evaluation.
    bound = 16 * eps * math.ceil(math.log2(grid.n_to)) + 2 * (
        space.n_r * eps / (1 - space.n_r * eps)
    )
    reference = NumpyRefBackend(problem, grid=grid)
    try:
        reference.set_initial(problem.initial.field0, None)
        reference._hoist(0.0)  # pyright: ignore[reportPrivateUsage]
        expected = np.asarray(
            reference._nonlinear(  # pyright: ignore[reportPrivateUsage]
                reference._require_state(),  # pyright: ignore[reportPrivateUsage]
                reference._stage_scalars(0.0, 0.0),  # pyright: ignore[reportPrivateUsage]
                np.ones(problem.shape, complex),
            )[0]
        )
    finally:
        reference.close()
    backend = NativeBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        info = build_info()
        # The native rule: a HYBRID or KOKKOSBLAS_TPL build on a host execution
        # space, at this shape; the dev build is HYBRID without TPL BLAS and packs too.
        eligible = (
            info.gemm_provider in {"HYBRID", "KOKKOSBLAS_TPL"}
            and info.device_backend == "none"
        )
        provider = str(backend.metadata()["synthesis_provider"])
        ledger = backend.memory_ledger()
        if eligible:
            assert provider.endswith("-packed-real"), provider
            assert ledger["radial_blas_buffers"] == 32 * space.n_r * grid.n_to
        else:
            assert not provider.endswith("-packed-real"), provider
            assert ledger.get("radial_blas_buffers", 0) == 0
        backend._trial_step(0.0, 0.0)  # pyright: ignore[reportPrivateUsage]
        state = backend._native_state  # pyright: ignore[reportPrivateUsage]
        actual = np.asarray(state.stage_to_host(0)).reshape(problem.shape[1:])
    finally:
        backend.close()
    scale = float(np.max(np.abs(expected)))
    assert scale > 0
    np.testing.assert_allclose(actual, expected, rtol=0, atol=bound * scale)


def test_schema_seven_periodic_rows_drain_and_reset(phase_a_case: PhaseACase) -> None:
    """Inherited storage owns rows across observations and a fresh initial load."""
    original, grid = phase_a_case
    problem = dataclasses.replace(
        original,
        stats_spec=StatsSpec(names=("energy",), periodic=("energy",), period=1),
    )
    backend = NativeBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(float(problem.stepper.save_planes[-1]))
        observed = backend.observe()
        rows = observed.step_stats
        assert rows.n_rows > 0
        # Exact: integer accounting of the accepted steps and, below, the drained
        # rows' own storage, which a later observation or reload must not touch.
        np.testing.assert_array_equal(
            rows.step, np.flatnonzero(observed.step_history.accepted)
        )
        assert np.all(np.isfinite(rows.values["energy"]))
        saved = rows.values["energy"].copy()
        assert backend.observe().step_stats.n_rows == 0
        backend.set_initial(problem.initial.field0, None)
        assert backend.observe().step_stats.n_rows == 0
        np.testing.assert_array_equal(rows.values["energy"], saved)
    finally:
        backend.close()


def test_envelope_raman_ignores_field_beat_flag(raman_case: RamanCase) -> None:
    """The envelope's complex quadrature always contributes to its Raman drive.

    The flag is ignored, so both runs are the same discrete computation on one
    prescribed step and agree to its rounding budget (solprop SEP 25): seven stages,
    each a transform round trip and the Raman convolution's on the doubled axis.
    Honouring the flag would drop the drive's quadrature and move the field by far more.
    """
    original, grid = raman_case
    problem = dataclasses.replace(
        original,
        stepper=dataclasses.replace(
            original.stepper, schedule=FixedSchedule.uniform(0.0, 1e-5, 1e-5)
        ),
    )
    leaf = problem.responses.raman_terms[0]
    changed_leaf = dataclasses.replace(leaf, thg=True)
    changed = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses,
            raman_terms=(changed_leaf,),
        ),
    )
    fields: list[np.ndarray] = []
    for candidate in (problem, changed):
        backend = NativeBackend(candidate, grid=grid)
        try:
            backend.set_initial(candidate.initial.field0, None)
            backend.propagate_to(1e-5)
            fields.append(backend.observe().require_field().copy())
            assert backend.metadata()["raman_drive"] == "cycle_averaged"
        finally:
            backend.close()
    bound = step_bound(grid.n_to, 1) + step_bound(2 * grid.n_to, 1)
    assert_within(fields[1], fields[0], bound, what="field")


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.parametrize(
    ("noise", "stashed"), [(None, True), ("input", True), ("modified", False)]
)
def test_the_plasma_stash_is_off_under_modified_noise(
    noise: NoiseModel | None, stashed: bool
) -> None:
    """Modified noise scans the committed field for the plasma statistics.

    The last stage's input carries the noise draw there, so the stash would describe
    a state the run never adopted; solprop's
    ``tests/driver/test_envelope_plasma.py`` holds the statistics to numpy_ref's.
    """
    simulation = Simulation.from_capillary(
        125e-6,
        8e-6,
        "Ar",
        1.0,
        lambda0=800e-9,
        wavelength_limits=(400e-9, 2e-6),
        trange=80e-15,
        fwhm=20e-15,
        energy=1e-5,
        plasma=True,
        kerr=False,
        raman=False,
        n_r=4,
        noise=noise,
        seed=821 if noise else None,
        stats=StatsSpec(
            names=("peak_ionisation_rate",), periodic=("peak_ionisation_rate",)
        ),
        backend="numpy_ref",
    )
    problem = simulation.translate()
    assert problem.responses.plasma_channels
    payload = _native_problem(  # pyright: ignore[reportPrivateUsage]
        problem, terms(problem)
    )
    assert payload["stash_plasma"] is stashed


def test_the_plasma_stash_is_off_under_the_split_step_absorber() -> None:
    """The split-step absorber changes the field after the last stage.

    The last stage's input is then not the committed field, so the plasma statistics
    are scanned afresh, as numba_cpu does (its stash needs a first-same-as-last step).
    """
    payloads = {}
    for scheme in ("nonlinear", "split_step"):
        simulation = Simulation.from_capillary(
            125e-6,
            8e-6,
            "Ar",
            1.0,
            lambda0=800e-9,
            wavelength_limits=(400e-9, 2e-6),
            trange=80e-15,
            fwhm=20e-15,
            energy=1e-5,
            plasma=True,
            kerr=False,
            raman=False,
            n_r=4,
            stats=StatsSpec(
                names=("peak_ionisation_rate",), periodic=("peak_ionisation_rate",)
            ),
            backend="numpy_ref",
        )
        simulation.settings = dataclasses.replace(
            simulation.settings, absorber_scheme=scheme
        )
        problem = simulation.translate()
        payloads[scheme] = _native_problem(  # pyright: ignore[reportPrivateUsage]
            problem, terms(problem)
        )
    assert payloads["nonlinear"]["stash_plasma"] is True
    assert payloads["split_step"]["split_step_absorber"] is True
    assert payloads["split_step"]["stash_plasma"] is False


def test_the_held_current_collar_member_is_checked_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(masks, "Grids", dataclasses.make_dataclass("Grids", []))
    with pytest.raises(BackendError, match=r"has no Grids\.t_window_fine_derivative"):
        check_build()
