# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Backend construction, refusals, metadata, and persistent-state accounting."""

from __future__ import annotations

import copy
import dataclasses
import functools
import math
import re
from types import SimpleNamespace
from typing import NoReturn, cast

import numpy as np
import numpy.typing as npt
import pytest
from bounds import EPS, assert_within, step_bound
from conftest import PhaseACase, RamanCase
from legacy_geometry import LEGACY_GRID
from session_runs import committed_steps, load_native, native_trial, run
from solprop.backends.events import ObservationReady
from solprop.backends.low_storage import schedule_key
from solprop.backends.node_chunk import plan_node_chunk
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.session import Start
from solprop.driver import Simulation
from solprop.exceptions import BackendError, TranslationError
from solprop.grid import EnvelopeGrid, FreeGrid, RadialGrid
from solprop.noise import NoiseModel
from solprop.nonlinear.responses import DispersiveKerrResponse, KerrResponse
from solprop.physdata.nonlinear_data import OptmatKerr
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, PIController
from solprop.stepper.multistep import Multistep
from solprop.translate.coupled import CoupledFlowTables
from solprop.translate.linop import delta_phi
from solprop.translate.problem import PropagationProblem, assemble
from solprop.translate.stage import terms

import solprop_native
from solprop_native import NativeBackend, _core, build_info, check_build
from solprop_native.backend import (
    _lower,  # pyright: ignore[reportPrivateUsage]
    _LowerPlan,  # pyright: ignore[reportPrivateUsage]
    _ResidentLowStorage,  # pyright: ignore[reportPrivateUsage]
)
from solprop_native.session import NativeSession


def test_check_build_accepts_the_compiled_contract() -> None:
    assert check_build() is None


def test_rk_source_dispatch_counter_resets_and_excludes_fsal_reuse(
    phase_a_case: PhaseACase,
) -> None:
    """SEP 30 counts executed stages rather than inferred accepted-step work."""
    problem, run_request, grid = phase_a_case
    target = float(run_request.stepper.save_planes[-1])
    stepper = dataclasses.replace(
        run_request.stepper,
        integrator=None,
        auto_init=False,
        schedule=FixedSchedule.uniform(0.0, target, target / 2),
    )
    # One step to the first plane is seven stages; the second step reuses the first's
    # last stage (first same as last), so two steps are thirteen. A second run on the
    # same backend loads its start again, which resets the count.
    for planes, count, expected in (
        ((target / 2,), 1, 7),
        ((target / 2, target), 2, 13),
    ):
        configured_request = run_request.with_stepper(
            dataclasses.replace(
                stepper,
                save_planes=planes,
                schedule=FixedSchedule.uniform(0.0, planes[-1], planes[-1] / count),
            )
        )
        backend = NativeBackend(problem, configured_request, grid=grid)
        try:
            assert backend.metadata()["source_evaluations"] == 0
            for _ in range(2):
                run(backend, problem, configured_request)
                assert backend.metadata()["source_evaluations"] == expected
        finally:
            backend.close()


@pytest.mark.parametrize("missing", ["phi", "prefactor"])
def test_native_modal_payload_requires_both_fixed_coefficient_arrays(
    phase_a_case: PhaseACase, missing: str
) -> None:
    """The separable representation must not relax ordinary modal validation."""
    problem, run_request, grid = phase_a_case
    backend = NativeBackend(problem, run_request, grid=grid)
    try:
        terms = backend._terms  # pyright: ignore[reportPrivateUsage]
        damaged = dataclasses.replace(
            terms,
            phi=None if missing == "phi" else terms.phi,
            prefactor=None if missing == "prefactor" else terms.prefactor,
        )
        with pytest.raises(BackendError, match="device-ready fixed-term problem"):
            _lower(problem, run_request, _LowerPlan(terms=damaged))
    finally:
        backend.close()


def test_spatial_statistics_require_geometry(phase_a_case: PhaseACase) -> None:
    """Refuse a spatial reduction on an ordinary modal state before native entry,
    with the reference's message (``backends.session.statistics_unavailable``)."""
    problem, run_request, grid = phase_a_case
    backend = NativeBackend(problem, run_request, grid=grid)
    try:
        for name in ("beam_radius", "fwhm_r", "on_axis_intensity", "absorbed_power"):
            asked = run_request.for_run(stats=StatsSpec(names=("energy", name)))
            with pytest.raises(
                BackendError, match=f"the '{name}' statistic requires spatial geometry"
            ):
                backend.prepare(problem, asked)
    finally:
        backend.close()


def test_spatial_statistics_refuse_a_malformed_native_result(
    phase_a_case: PhaseACase,
) -> None:
    """Check the native packed-result boundary independently of native reductions.

    The native session names each plane's packed evaluation with its plane lowering
    (:meth:`solprop_native._statistics.StatisticsLowering.unpack`), here handed one
    value more than the plan's.
    """
    original_request = phase_a_case[1]
    grid = phase_a_case[2]
    problem, run_request = assemble(
        grid,
        None,
        [],
        np.zeros((8, 1, grid.n_omega), complex),
        spatial_grid=RadialGrid(1e-3, 8),
        medium=SpatialMedium(),
        controller=original_request.stepper.controller,
        save_planes=[1e-3],
        dz_max=1e-3,
        dz_init=1e-4,
    )
    run_request = run_request.for_run(stats=StatsSpec(names=("energy",)))
    backend = NativeBackend(problem, run_request, grid=grid)
    session = backend.prepare(problem, run_request).start(Start.entrance(problem))
    try:
        assert isinstance(session, NativeSession)
        lowering = session._plane_lowering  # pyright: ignore[reportPrivateUsage]
        # Energy is the statistics plan's: one value, and the evaluation hands two.
        assert lowering.names == ("energy",)
        with pytest.raises(BackendError, match="returned 2 values; consumed 1"):
            lowering.unpack(np.zeros(2))
    finally:
        session.close()


def test_stage_observation_owns_its_host_copy_and_checks_lifetime(
    phase_a_case: PhaseACase,
) -> None:
    """A stage derivative is observed directly without exposing persistent views.

    The trial is the native state's own (``rk_trial``) on the backend's staged rows,
    from the launch loaded into the state.
    """
    problem, run_request, grid = phase_a_case
    backend = NativeBackend(problem, run_request, grid=grid)
    state = load_native(backend, problem.initial.field0, clear_fsal=False)
    with pytest.raises(RuntimeError, match="stage"):
        state.stage_to_host(0)
    native_trial(backend, 0.0, 1e-4)
    derivative = state.stage_to_host(0)
    assert derivative.shape == (1, grid.n_omega)
    assert derivative.dtype == np.complex128
    assert np.max(np.abs(derivative)) > 0
    saved = derivative.copy()
    derivative.fill(0)
    # Exact: a second host copy of the same device stage, not a recomputation.
    np.testing.assert_array_equal(state.stage_to_host(0), saved)
    with pytest.raises(IndexError, match="stage"):
        state.stage_to_host(run_request.stepper.tableau.n_stages)
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        state.stage_to_host(0)


def test_the_solprop_version_is_recorded_and_never_compared(
    phase_a_case: PhaseACase, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, run_request, grid = phase_a_case

    def elsewhere(name: str) -> str:
        return f"{name}-elsewhere"

    monkeypatch.setattr(solprop_native.backend, "version", elsewhere)
    # The build facts are read once per process; read them afresh under the patch,
    # through a fresh cache the patch removes again.
    facts = solprop_native.backend._build_facts  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(
        solprop_native.backend, "_build_facts", functools.cache(facts.__wrapped__)
    )
    assert check_build() is None
    backend = NativeBackend(problem, run_request, grid=grid)
    try:
        record = backend.metadata()
    finally:
        backend.close()
    assert record["solprop_version"] == "solprop-elsewhere"
    assert record["solprop_version_built_against"] != "solprop-elsewhere"


@pytest.mark.parametrize("mode", ["tabulate", "recompute"])
def test_the_low_storage_march_steps_as_the_reference(
    phase_a_case: PhaseACase, mode: str
) -> None:
    # solprop SEP 24 phase 3: the resident engine, one native call a step, against
    # the reference's host engine on the same schedule.
    problem, run_request, grid = phase_a_case
    target = float(run_request.stepper.save_planes[-1])
    steps = 4
    stepper = dataclasses.replace(
        run_request.stepper,
        integrator=LowStorageRungeKutta(),
        schedule=FixedSchedule.uniform(0.0, target, target / steps),
        propagator_mode=mode,
    )
    marched, marched_request = problem, run_request.with_stepper(stepper)
    reference = run(
        NumpyRefBackend(marched, marched_request, grid=grid),
        marched,
        marched_request,
        owns=True,
    ).fields[-1]
    backend = NativeBackend(marched, marched_request, grid=grid)
    record: dict[str, object] = {}
    try:
        field = run(backend, marched, marched_request, record=record).fields[-1]
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    assert ledger["stages"] == ledger["advanced"] == 0
    assert record["source_evaluations"] == 4 * steps
    # Four transform round trips a step (no stage combination: one register read),
    # and the flows' exponents, formed from the table or the phase terms against the
    # reference's propagator: a few roundings of the largest phase a step sweeps.
    swept = float(np.max(np.abs(delta_phi(marched.linear, 0.0, target))))
    bound = step_bound(grid.n_to, steps, stages=4) + 64 * EPS * swept
    assert_within(field, reference, bound, what=f"the low-storage march, {mode}")


@pytest.mark.parametrize("asked", ["auto", "tabulate", "recompute"])
def test_the_record_names_the_propagators_a_run_used(
    phase_a_case: PhaseACase, asked: str
) -> None:
    # #464: only the low-storage march holds flows. Under Runge-Kutta every request
    # holds the one whole-step propagator, so the record says recompute for each.
    problem, run_request, grid = phase_a_case
    held = {}
    for mode in ("recompute", asked):
        stepper = dataclasses.replace(run_request.stepper, propagator_mode=mode)
        backend = NativeBackend(problem, run_request.with_stepper(stepper), grid=grid)
        try:
            held[mode] = backend.memory_ledger()["propagators"]
            assert backend.metadata()["propagator_mode"] == "recompute"
        finally:
            backend.close()
    assert held[asked] == held["recompute"]


@pytest.mark.parametrize("asked", ["tabulate", "recompute"])
def test_the_record_names_the_low_storage_march_s_flows(
    phase_a_case: PhaseACase, asked: str
) -> None:
    problem, run_request, grid = phase_a_case
    target = float(run_request.stepper.save_planes[-1])
    stepper = dataclasses.replace(
        run_request.stepper,
        integrator=LowStorageRungeKutta(),
        schedule=FixedSchedule.uniform(0.0, target, target / 4),
        propagator_mode=asked,
    )
    backend = NativeBackend(problem, run_request.with_stepper(stepper), grid=grid)
    try:
        assert backend.metadata()["propagator_mode"] == asked
    finally:
        backend.close()


def test_a_problem_without_its_responses_is_linear_and_admitted(
    phase_a_case: PhaseACase,
) -> None:
    # The response order is derived from the leaves, so removing the Kerr leaf
    # removes its stage too: a consistent, linear problem.
    problem, run_request, grid = phase_a_case
    linear = dataclasses.replace(
        problem,
        responses=dataclasses.replace(problem.responses, kerr_coefficients=None),
    )
    assert len(linear.responses.response_order) == 1
    NativeBackend(linear, run_request, grid=grid).close()


def test_a_low_storage_table_never_built_is_refused(phase_a_case: PhaseACase) -> None:
    # solprop SEP 24's closing guard: the native state refuses to reuse a constant
    # operator's table it has not built since the field was loaded, rather than step
    # with unwritten flows. solprop's engine keys the table from NaN after every load,
    # so only a stale key reaches it: forged here.
    problem, run_request, grid = phase_a_case
    target = float(run_request.stepper.save_planes[-1])
    stepper = dataclasses.replace(
        run_request.stepper,
        integrator=LowStorageRungeKutta(),
        schedule=FixedSchedule.uniform(0.0, target, target / 4),
        propagator_mode="tabulate",
    )
    marched, marched_request = problem, run_request.with_stepper(stepper)
    backend = NativeBackend(marched, marched_request, grid=grid)
    try:
        # The resident engine on the loaded state, with the key of the first step's
        # table, which no step built.
        load_native(backend, marched.initial.field0, clear_fsal=False)
        method = marched_request.stepper.integrator
        assert isinstance(method, LowStorageRungeKutta)
        engine = _ResidentLowStorage(backend, method)
        engine._table_step = schedule_key(  # pyright: ignore[reportPrivateUsage]
            engine._keys,  # pyright: ignore[reportPrivateUsage]
            engine._edges,  # pyright: ignore[reportPrivateUsage]
            0.0,
            target / 4,
        )
        with pytest.raises(RuntimeError, match="table has not been built"):
            engine.step(0.0, target / 4)
    finally:
        backend.close()


def test_an_adams_integrator_steps_as_the_reference(phase_a_case: PhaseACase) -> None:
    # solprop's shared Adams march (SEP 13, SEP 14 phase 2) with this backend's one
    # source evaluation a step: the reference's steps and calls, and its field to the
    # rounding of the two evaluation orders.
    problem, run_request, grid = phase_a_case
    stepper = dataclasses.replace(run_request.stepper, integrator=Multistep())
    adams, adams_request = problem, run_request.with_stepper(stepper)
    found: list[tuple[npt.NDArray[np.complex128], tuple[int, int]]] = []
    for backend_class in (NumpyRefBackend, NativeBackend):
        backend = backend_class(adams, adams_request, grid=grid)
        result = run(backend, adams, adams_request, owns=True)
        accepted = committed_steps(result)
        found.append((result.fields[-1], (accepted, result.attempts - accepted)))
    (reference, reference_steps), (field, native_steps) = found
    assert native_steps == reference_steps
    # solprop's own parity test measures 8e-16 to 1.8e-15 of the peak.
    scale = float(np.max(np.abs(reference)))
    np.testing.assert_allclose(field, reference, rtol=0, atol=1e-12 * scale)


def test_an_adams_integrator_observes_densely_as_the_reference(
    phase_a_case: PhaseACase,
) -> None:
    # An interpolated plane (solprop ADR-036, ADR-039) lies inside a step and is loaded
    # natively for the observation; the march resumes from its own host state.
    problem, run_request, grid = phase_a_case
    stepper = dataclasses.replace(
        run_request.stepper, integrator=Multistep(), interpolate_planes=True
    )
    adams, adams_request = problem, run_request.with_stepper(stepper)
    found: list[npt.NDArray[np.complex128]] = []
    for backend_class in (NumpyRefBackend, NativeBackend):
        backend = backend_class(adams, adams_request, grid=grid)
        result = run(backend, adams, adams_request, owns=True)
        # The save planes, after the launch.
        assert [plane.z for plane in result.all_planes()[1:]] == list(
            adams_request.stepper.save_planes
        )
        found.append(np.stack(result.fields[1:]))
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
    """Refuse TDSE without its binding before native setup or a march is selected.

    The CPU build runs the TDSE response through its host source (solprop SEP 49),
    which needs the driver's binding; a problem without one is refused by name first.
    """
    problem, run_request, grid = phase_a_case
    # This deliberately bypasses scientific validation to isolate the adapter's
    # earliest refusal. Neither a valid atomic basis nor an optional provider is
    # needed to refuse a route that comes without its binding.
    requested = copy.copy(problem)
    route = copy.copy(problem.route)
    object.__setattr__(
        route,
        "tdse",
        SimpleNamespace(descriptor=SimpleNamespace(), source_filter=source_filter),
    )
    object.__setattr__(requested, "route", route)
    requested_request = run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper,
            integrator=Multistep() if with_adams else None,
            interpolate_planes=dense_output,
        )
    )

    def forbidden(*_arguments: object, **_keywords: object) -> NoReturn:
        raise AssertionError("unsupported TDSE reached native setup")

    monkeypatch.setattr(solprop_native.backend, "require_device_ready", forbidden)
    monkeypatch.setattr(_core, "State", forbidden)
    with pytest.raises(BackendError, match="explicit host response binding"):
        NativeBackend(requested, requested_request, grid=grid)


def _resumed(problem: PropagationProblem, ledgers: dict[str, object]) -> Start:
    """Return the problem's launch as a start that restores `ledgers`."""
    return Start(
        field=problem.initial.field0, z=float(problem.initial.z), ledgers=ledgers
    )


def test_runge_kutta_native_resume_keeps_no_logical_ledger(
    phase_a_case: PhaseACase,
) -> None:
    # A resumed start's ledgers (the native session restores them at its load).
    problem, run_request, grid = phase_a_case
    backend = NativeBackend(problem, run_request, grid=grid)
    try:
        empty: tuple[dict[str, object], ...] = ({}, {"logical": {}, "atomic": {}})
        for ledgers in empty:
            start = _resumed(problem, ledgers)
            run(backend, problem, run_request, start)
        # A Runge-Kutta march keeps no logical ledger: the counts stay with the
        # earlier execution as provenance, not installed (SEP 42 section 5.6).
        start = _resumed(problem, {"logical": {"startup": 1}, "atomic": {}})
        record: dict[str, object] = {}
        run(backend, problem, run_request, start, record=record)
        assert "response_calls" not in record
        assert "response_calls" not in backend.metadata()
        start = _resumed(problem, {"logical": {}, "atomic": {"attempted": 1}})
        with pytest.raises(BackendError, match="does not restore"):
            run(backend, problem, run_request, start)
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
    the combinations solprop's problem validation admits, which this adapter runs
    where it is built. A lone table, metric or response is refused there before
    anything is allocated: a sentinel by its type, the coupled flow, a real leaf of
    another grid, by its axis.

    The sentinel is set with ``object.__setattr__`` on a copy rather than through
    ``dataclasses.replace``, which re-runs the group's own validation: solprop's
    ``Transform`` refuses a ``modal_tables`` record beside the static matrices that
    record replaces, so a ``replace`` never reaches this adapter at all.
    """
    problem, run_request, grid = phase_a_case
    assert getattr(getattr(problem, group), leaf) is None
    spoiled_group = copy.copy(getattr(problem, group))
    object.__setattr__(spoiled_group, leaf, _leaf_value(problem, leaf))
    if group == "transform":
        # The transform is the route's, whose own check refuses tables on a modal
        # route: the copy is spoiled past it, as the transform was.
        route = copy.copy(problem.route)
        object.__setattr__(route, "transform", spoiled_group)
        spoiled = copy.copy(problem)
        object.__setattr__(spoiled, "route", route)
    else:
        spoiled = dataclasses.replace(problem, **{group: spoiled_group})
    # Refused by name: its type or its consistency with the state first, then the
    # route's policy. Tables on the modal route are its transform's, which the route
    # names.
    expected = (
        "with general modal tables"
        if leaf == "modal_tables"
        else re.escape(f"{group}.{leaf}")
    )
    with pytest.raises(TranslationError, match=expected):
        NativeBackend(spoiled, run_request, grid=grid)


def _leaf_value(problem: PropagationProblem, leaf: str) -> object:
    """Return a leaf the problem's validation refuses on this problem."""
    if leaf != "coupled_flow":
        return object()
    n_omega = problem.shape[-1]
    n_modes = problem.shape[0]
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
    _, base_request, grid = phase_a_case
    space = FreeGrid(1e-3, 4, 1.5e-3, 6, collar_fraction=0)
    response = DispersiveKerrResponse(
        OptmatKerr("Ar").dispersive_model(), evaluator, thg=False
    )
    problem, run_request = assemble(
        grid,
        None,
        [response],
        np.zeros((24, 1, grid.n_omega), complex),
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=base_request.stepper.controller,
        save_planes=[1e-3],
        dz_max=1e-3,
        dz_init=1e-4,
        density=4.9e25,
    )
    assert problem.responses.dispersive_kerr is not None
    backend = NativeBackend(problem, run_request, grid=grid)
    try:
        assert backend.metadata()["dispersive_path"] == path
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    held = ledger["dispersive_products"] + ledger["dispersive_product_spectra"]
    assert (held == 0) == (evaluator == "symmetric")


def test_statistics_do_not_observe_the_full_field(
    phase_a_case: PhaseACase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The statistics seam transfers reductions, not the protected state.

    A native session's planes carry the statistics, and nothing exports a field:
    the session's record counts no export and no exported byte.
    """
    problem, run_request, grid = phase_a_case
    names = ("omega0", "energy", "peak_power", "fwhm_t")
    run_request = run_request.for_run(stats=StatsSpec(names=names))
    backend = NativeBackend(problem, run_request, grid=grid)

    def fail_if_observed(state: object) -> NoReturn:
        pytest.fail(f"statistics observed native state {state!r}")

    monkeypatch.setattr(backend, "_observed_field", fail_if_observed)
    session = backend.prepare(problem, run_request).start(Start.entrance(problem))
    observed: list[dict[str, np.ndarray]] = []
    try:
        while not session.finished:
            batch = session.advance()
            observed.extend(
                dict(event.stats)
                for event in batch.events
                if isinstance(event, ObservationReady)
            )
            session.acknowledge(batch.receipt)
        work = cast("dict[str, int]", session.execution_record()["native_session"])
    finally:
        session.close()
    assert len(observed) == 2  # the launch and the save plane
    for values in observed:
        assert set(values) == set(names)
        assert all(np.all(np.isfinite(value)) for value in values.values())
    assert work["exports"] == work["export_bytes"] == 0


def test_raman_metadata_memoization_and_conditional_ledger(
    raman_case: RamanCase,
) -> None:
    """Raman plans are persistent and the exact ordered rate tuple is memoized."""
    problem, run_request, grid = raman_case
    backend = NativeBackend(problem, run_request, grid=grid)
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


def _raman_native_payload(raman_case: RamanCase) -> _core.Problem:
    problem, run_request, grid = raman_case
    backend = NativeBackend(problem, run_request, grid=grid)
    try:
        return _lower(
            problem,
            run_request,
            _LowerPlan(terms=backend._terms),  # pyright: ignore[reportPrivateUsage]
        )
    finally:
        backend.close()


@pytest.mark.parametrize("spoiling", ["dtype", "rank", "contiguity"])
def test_native_raman_rows_refuse_hidden_array_conversions(
    raman_case: RamanCase, spoiling: str
) -> None:
    """The private seam accepts exact arrays and never copies a near miss."""
    payload = _raman_native_payload(raman_case)
    raman = payload.raman
    rows = raman.undamped.reshape(raman.n_terms, -1)

    def _spoil() -> None:
        if spoiling == "dtype":
            raman.undamped = rows.astype(np.float32)  # pyright: ignore[reportAttributeAccessIssue]
        elif spoiling == "rank":
            raman.undamped = rows.ravel()
        else:
            raman.undamped = np.zeros(
                (rows.shape[0], 2 * rows.shape[1]), dtype=np.float64
            )[:, ::2]
        _core.State(payload)

    with pytest.raises((TypeError, ValueError)):
        _spoil()


@pytest.mark.parametrize("spoiling", ["undamped", "tensor_weights"])
def test_native_raman_tables_refuse_a_transposed_layout(
    raman_case: RamanCase, spoiling: str
) -> None:
    """A table with the right element count but its axes swapped is refused by name."""
    payload = _raman_native_payload(raman_case)
    raman = payload.raman
    if spoiling == "undamped":
        raman.undamped = raman.undamped.reshape(-1, raman.n_terms)
    else:
        raman.tensor_weights = raman.tensor_weights.reshape(2, raman.n_terms)
    with pytest.raises(ValueError, match=f"raman_{spoiling} must have shape"):
        _core.State(payload)


@pytest.mark.parametrize("spoiling", ["flat", "rank_four", "count"])
def test_native_fixed_terms_refuse_a_wrong_layout(
    phase_a_case: PhaseACase, spoiling: str
) -> None:
    """The fixed terms are (n_phi_terms, ...) arrays of rank two or three."""
    problem, run_request, grid = phase_a_case
    backend = NativeBackend(problem, run_request, grid=grid)
    try:
        terms = backend._terms  # pyright: ignore[reportPrivateUsage]
        payload = _lower(problem, run_request, _LowerPlan(terms=terms))
    finally:
        backend.close()
    assert terms.phi is not None
    # As the adapter lays them out: (n_phi_terms, rows, n_omega).
    phi = np.ascontiguousarray(terms.phi).reshape(
        terms.phi.shape[0], -1, problem.shape[-1]
    )
    if spoiling == "flat":
        payload.linear.phi_terms = phi.reshape(-1)
        message = "rank-two or rank-three"
    elif spoiling == "rank_four":
        payload.linear.phi_terms = phi.reshape(1, *phi.shape)
        message = "rank-two or rank-three"
    else:
        # The same elements, the leading axis no longer the term count.
        assert phi.shape[::-1][0] != payload.linear.n_phi_terms
        payload.linear.phi_terms = phi.reshape(phi.shape[::-1])
        message = "phi_terms must hold n_phi_terms rows"
    with pytest.raises(ValueError, match=message):
        _core.State(payload)


def test_native_raman_rows_accept_read_only_input(raman_case: RamanCase) -> None:
    """Construction reads Raman rows but does not require write access to them."""
    payload = _raman_native_payload(raman_case)
    rows = payload.raman.undamped.reshape(payload.raman.n_terms, -1)
    rows.setflags(write=False)
    payload.raman.undamped = rows
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
    problem, run_request, grid = phase_a_case
    backend = NativeBackend(problem, run_request, grid=grid)
    metadata = backend.metadata()
    ledger = backend.memory_ledger()
    n = problem.shape[-1]
    stages = run_request.stepper.tableau.n_stages
    assert metadata["backend"] == "native"
    assert metadata["source_digest"] == build_info().source_digest
    assert metadata["openmp_wait_policy"] == solprop_native.WAIT_POLICY
    assert metadata["openmp_proc_bind"] == solprop_native.PROC_BIND
    # NumPy's own BLAS pool is held while the backend lives (solprop_native._host_blas).
    assert metadata["host_blas"] in {"none", "unavailable"} or str(
        metadata["host_blas"]
    ).endswith(("thread", "threads"))
    assert cast(float, metadata["planning_seconds"]) >= 0.0
    # A one-mode envelope at one node. A host build folds its one-element transforms
    # into the response column (solprop SEP 25). A device build's hybrid policy keeps
    # the custom kernel under its shape guard (one mode).
    build = solprop_native.build_info()
    expected_provider = "folded" if build.device_backend == "none" else "custom"
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


@pytest.mark.parametrize("kind", ["radial", "free"])
def test_temporal_statistics_are_refused_on_a_spatial_problem(
    phase_a_case: PhaseACase, kind: str
) -> None:
    """Mirror numpy_ref before native entry: a peak power has no meaning on Hankel
    or DFT columns, and the Cartesian native statistics buffers hold one node
    chunk."""
    _, base_request, grid = phase_a_case
    space: RadialGrid | FreeGrid = (
        RadialGrid(1e-3, 8)
        if kind == "radial"
        else FreeGrid(1e-3, 4, 1.5e-3, 6, collar_fraction=0)
    )
    problem, run_request = assemble(
        grid,
        None,
        [],
        np.zeros((space.n_columns, 1, grid.n_omega), complex),
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=base_request.stepper.controller,
        save_planes=[1e-3],
        dz_max=1e-3,
        dz_init=1e-4,
    )
    backend = NativeBackend(problem, run_request, grid=grid)
    try:
        for name in ("peak_power", "fwhm_t", "energy_window", "omega0"):
            asked = run_request.for_run(stats=StatsSpec(names=(name,)))
            with pytest.raises(BackendError, match="not spatial saved-field"):
                backend.prepare(problem, asked)
    finally:
        backend.close()
    run_request = run_request.for_run(stats=StatsSpec(names=("energy",)))
    result = run(
        NativeBackend(problem, run_request, grid=grid), problem, run_request, owns=True
    )
    assert all(np.all(plane.stats["energy"] == 0) for plane in result.all_planes())


def test_packed_real_blas_radial_envelope_matches_the_reference_when_eligible() -> None:
    """The packed real GEMM is production policy for whole scalar radial envelopes of
    at least 256 nodes and 256 fine samples on HYBRID/TPL host builds; pin its
    selection, its ledger row and its agreement with the reference at that
    shape."""
    grid = EnvelopeGrid.from_limits(
        1e-4, 800e-9, (600e-9, 1.2e-6), 300e-15, thg=False, **LEGACY_GRID
    )
    assert grid.n_to == 256
    space = RadialGrid(1e-3, 256)
    start = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), 1e-4
    ).build(grid, space, medium=SpatialMedium())
    problem, run_request = assemble(
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
    assert plan_node_chunk(problem).chunk == problem.transform.n_nodes
    eps = np.finfo(float).eps
    # The ladder bound of the paired solprop tests: one fine FFT depth and two
    # length-n_r radial reductions per evaluation.
    bound = 16 * eps * math.ceil(math.log2(grid.n_to)) + 2 * (
        space.n_r * eps / (1 - space.n_r * eps)
    )
    reference = NumpyRefBackend(problem, run_request, grid=grid)
    try:
        reference.set_initial(problem.initial.field0, None)
        reference._hoist(0.0)  # pyright: ignore[reportPrivateUsage]
        expected = np.asarray(
            reference._nonlinear(  # pyright: ignore[reportPrivateUsage]
                reference._require_state(),  # pyright: ignore[reportPrivateUsage]
                reference._stage_scalars(0.0, 0.0),  # pyright: ignore[reportPrivateUsage]
                np.ones((1, *problem.shape), complex),
            )[0]
        )
    finally:
        reference.close()
    backend = NativeBackend(problem, run_request, grid=grid)
    try:
        state = load_native(backend, problem.initial.field0, clear_fsal=False)
        info = build_info()
        # The native rule: a host execution space, at this shape; the dev build
        # without TPL BLAS packs too.
        eligible = info.device_backend == "none"
        provider = str(backend.metadata()["synthesis_provider"])
        ledger = backend.memory_ledger()
        if eligible:
            assert provider.endswith("-packed-real"), provider
            assert ledger["radial_blas_buffers"] == 32 * space.n_r * grid.n_to
        else:
            assert not provider.endswith("-packed-real"), provider
            assert ledger.get("radial_blas_buffers", 0) == 0
        native_trial(backend, 0.0, 0.0)
        actual = np.asarray(state.stage_to_host(0)).reshape(problem.shape)
    finally:
        backend.close()
    scale = float(np.max(np.abs(expected)))
    assert scale > 0
    np.testing.assert_allclose(actual, expected, rtol=0, atol=bound * scale)


def test_envelope_raman_ignores_field_beat_flag(raman_case: RamanCase) -> None:
    """The envelope's complex quadrature always contributes to its Raman drive.

    The flag is ignored, so both runs are the same discrete computation on one
    prescribed step and agree to its rounding budget (solprop SEP 25): seven stages,
    each a transform round trip and the Raman convolution's on the doubled axis.
    Honouring the flag would drop the drive's quadrature and move the field by far more.
    """
    problem, run_request, grid = raman_case
    run_request = run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper, schedule=FixedSchedule.uniform(0.0, 1e-5, 1e-5)
        )
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
        backend = NativeBackend(candidate, run_request, grid=grid)
        try:
            fields.append(run(backend, candidate, run_request).fields[-1])
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
        wavelength=800e-9,
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
    payload = _lower(
        problem,
        simulation._run_request(),  # pyright: ignore[reportPrivateUsage]
        _LowerPlan(terms=terms(problem)),
    )
    assert payload.plasma.stash is stashed


def test_the_plasma_stash_is_off_under_the_split_step_absorber() -> None:
    """The split-step absorber changes the field after the last stage.

    The last stage's input is then not the committed field, so the plasma statistics
    are scanned afresh (the stash needs a first-same-as-last step).
    """
    payloads: dict[str, _core.Problem] = {}
    for scheme in ("nonlinear", "split_step"):
        simulation = Simulation.from_capillary(
            125e-6,
            8e-6,
            "Ar",
            1.0,
            wavelength=800e-9,
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
            noise=None,
        )
        simulation.settings = dataclasses.replace(
            simulation.settings, absorber_scheme=scheme
        )
        problem = simulation.translate()
        payloads[scheme] = _lower(
            problem,
            simulation._run_request(),  # pyright: ignore[reportPrivateUsage]
            _LowerPlan(terms=terms(problem)),
        )
    assert payloads["nonlinear"].plasma.stash is True
    assert payloads["split_step"].absorber.split_step is True
    assert payloads["split_step"].plasma.stash is False
