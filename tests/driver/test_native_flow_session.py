# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The coupled flow's exponential Adams march, resident wherever a CPU build runs it.

solprop SEP 47 phase 4 (section 6.3, the coupled-flow row): a coupled block wider than
16 channels, and a block with bins off its flow's eigenbasis route, step on the
resident Adams engine of a CPU or a device build, and the native session serves them,
where solprop's host engine stepped them over the backend's hooks before (that engine
was deleted with the inherited driver's fallbacks in phase 5).

The wide cases are lossless helium tapers with taper coupling and every bin on the
route, over ten fixed steps short enough that the formation bound of
``tests/flow_parity.py`` stays far below one; the cases off the route are
``flow_parity``'s eight, every bin or every other one routed to the augmented
exponential. Their ten steps against ``numpy_ref`` are rows of the case catalogue
(``tests/backends/flow_rows.py``, ``flow wide ...`` and ``flow off route ...``). What
stays here: the routing, the off-route run against the same backend on the eigenbasis
route, the session held to ``numpy_ref`` and the same run twice to the bit on one
thread, the block functions, the refusals and the resume.
"""

import dataclasses
import math
import os
import weakref
from typing import Any

import numpy as np
import pytest

from coupled_flow_cases import (
    OFF_ROUTES,
    STEPS,
    WIDE,
    WIDE_LENGTH,
    Case,
    off_route,
    on_adams,
    parity_bound,
    propagate,
    wide_simulation,
)
from flow_parity import (
    CASES,
    LENGTH,
    UNIT,
    adams_formation_bound,
    build,
    departure,
    flow_scales,
    run,
    ten_step_bound,
)
from sessions.executors import IDENTITY, Started, drive, entrance
from sessions.runs import bare, final_field, load_native, to_planes
from sessions.runs import run as run_session
from solprop.backends.events import ObservationReady, Outcome
from solprop.backends.exponential import carrier_reference
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.multistep import (
    CoupledBlock,
    coupled_moments,
    trial_tables,
)
from solprop.backends.registry import available
from solprop.backends.session import Start
from solprop.backends.sessions import start_session
from solprop.driver import Simulation
from solprop.exceptions import (
    BackendError,
    GrazingIncidenceWarning,
    MaterialRangeWarning,
)
from solprop.stepper import FixedSchedule
from solprop.stepper.multistep import Multistep
from solprop.translate.coupled import schur_eigenbasis

pytestmark = [
    pytest.mark.skipif("native" not in available(), reason="native is not installed"),
    pytest.mark.filterwarnings(
        f"ignore::{MaterialRangeWarning.__module__}.MaterialRangeWarning"
    ),
    pytest.mark.filterwarnings(
        f"ignore::{GrazingIncidenceWarning.__module__}.GrazingIncidenceWarning"
    ),
    pytest.mark.filterwarnings("ignore:the temporal absorber attenuates"),
]

backend_module = pytest.importorskip("solprop_native.backend")
native_session = pytest.importorskip("solprop_native.session")
NativeBackend = pytest.importorskip("solprop_native").NativeBackend

DEVICE = pytest.importorskip("solprop_native").build_info().device_backend != "none"
"""Whether the installed build executes on an accelerator, which holds no host channel
rows and refuses the carrier frame."""


CONTEXT: dict[int, dict[str, Any]] = {}
"""Each built case's z profiles and modes, which its plane statistics read."""


@pytest.fixture(scope="module")
def cases():
    """Build each width's case once for the module, on demand."""
    built: dict[int, Case] = {}

    def get(modes: int) -> Case:
        if modes not in built:
            simulation = wide_simulation(modes)
            built[modes] = Case(
                simulation.translate(),
                simulation._run_request(),
                simulation.grid,
                WIDE_LENGTH,
            )
            CONTEXT[modes] = {
                "profiles": simulation.profiles,
                "modes": simulation.modes,
            }
        return built[modes]

    return get


def _scale(case: Case) -> float:
    return float(np.max(np.abs(case.problem.initial.field0)))


@pytest.mark.parametrize("modes", WIDE)
def test_a_wide_flow_takes_the_resident_engine_and_the_session(cases, modes):
    """Past 16 channels the backend and the session step the flow natively, on a CPU
    or a device build."""
    case = cases(modes)
    problem, request = case.problem, case.run_request
    held = backend_module._host_flow(problem)
    assert held is not None and bool(np.all(held.route))
    assert problem.shape[0] == modes > backend_module.ADAMS_MAX_CHANNELS
    assert native_session.refusal(problem, request) is None
    backend = NativeBackend(problem, request, grid=case.grid)
    try:
        assert type(backend._resident).__name__ == "_ResidentAdams"
        # A CPU build's host passes' channel rows: one per bin, nine samples of every
        # channel; a device's passes hold none.
        if not DEVICE:
            rows = backend.memory_ledger()["adams_history"]
            assert rows >= problem.grids.n_omega * 9 * modes * 16
    finally:
        backend.close()


def _one_thread() -> bool:
    return "1" in {
        os.environ.get("SOLPROP_NATIVE_THREADS"),
        os.environ.get("SOLPROP_THREADS"),
    }


def session_run(
    case: Case, context: dict[str, Any] | None = None
) -> tuple[np.ndarray, Any]:
    """Run a case's prepared execution; return its last field and status.

    The preparation opens the native session, the only native executor. Every
    observation is exported with the identity recipe, the state as the backend holds
    it. `context` holds the profiles and modes a case's statistics read, a wide case's
    own by default.
    """
    backend = NativeBackend(
        case.problem,
        case.run_request,
        grid=case.grid,
        **(CONTEXT[case.problem.shape[0]] if context is None else context),
    )
    prepared = backend.prepare(case.problem, case.run_request)
    session = prepared.start(entrance(case.problem))
    try:
        assert isinstance(session, native_session.NativeSession)
        run = drive(Started(session), export=IDENTITY)
    finally:
        session.close()
    return run.fields[-1].reshape(case.problem.shape), run.status


@pytest.mark.parametrize(
    "modes",
    [
        pytest.param(17, marks=pytest.mark.medium),
        pytest.param(24, marks=pytest.mark.slow),
    ],
)
def test_the_session_runs_a_wide_flow_within_the_bound(cases, modes):
    """The session is ``numpy_ref``'s in bound, and repeatable.

    SEP 47's phase 4 evidence, kept past the bridge: the session is held to
    ``parity_bound`` against ``numpy_ref``, and two sessions are the same run to the
    bit on one thread.
    """
    case = cases(modes)
    session, status = session_run(case)
    again, repeated = session_run(case)
    expected = propagate(case, NumpyRefBackend)
    linear = 1e3 * propagate(case, NumpyRefBackend, case.problem.initial.field0 * 1e-3)
    bound = parity_bound(case, expected, linear)
    assert status.attempts == repeated.attempts == STEPS
    if _one_thread():
        np.testing.assert_array_equal(session, again)
    assert 0.0 < bound < 1e-6 * _scale(case)
    assert float(np.max(np.abs(session - expected))) <= bound


# --- Bins off the flow's eigenbasis route ------------------------------------------


@pytest.fixture(scope="module")
def parity_cases():
    """Build each ``flow_parity`` case once for the module, on demand."""
    built: dict[str, tuple[Simulation, Any]] = {}

    def get(name: str):
        if name not in built:
            simulation = build(name)
            built[name] = (simulation, simulation.translate())
        return built[name]

    return get


def _assert_routed(problem, route: str) -> None:
    held = backend_module._host_flow(problem)
    assert held is not None
    assert not np.all(held.route) and (route == "off") == (not np.any(held.route))
    # The native passes read the Schur factor's upper triangle alone.
    assert np.all(np.tril(held.triangular, -1) == 0.0)


@pytest.mark.medium
@pytest.mark.parametrize("route", OFF_ROUTES)
@pytest.mark.parametrize("name", CASES)
def test_bins_off_the_route_are_the_eigenbasis_route_s(
    parity_cases, name, route, monkeypatch, record_property
):
    """Ten fixed Adams steps with bins off the route, against the route's own run.

    The bound, stated first: ``flow_parity.ten_step_bound`` with its formation bound
    the larger of the actions' and the augmented exponential's
    (``adams_formation_bound``), against the resident engine with every bin on the
    eigenbasis route, a pair of implementations of the same ten steps. The run against
    ``numpy_ref`` (SciPy's ``expm`` on every bin) within the same bound is a row of the
    case catalogue (``tests/backends/flow_rows.py``, ``flow off route ...``).
    """
    simulation, problem = parity_cases(name)
    request, grid = on_adams(simulation._run_request()), simulation.grid
    on = run(NativeBackend, problem, request, grid)
    expected = run(NumpyRefBackend, problem, request, grid)
    off_route(monkeypatch, route)
    _assert_routed(problem, route)
    resident = run(NativeBackend, problem, request, grid)
    scales = flow_scales(problem, request, grid, expected)
    formation = max(scales.formation, adams_formation_bound(problem, request, grid))
    bound = ten_step_bound(dataclasses.replace(scales, formation=formation))
    eigenbasis = departure(resident, on)
    record_property("bound", bound)
    record_property("eigenbasis", eigenbasis)
    assert 0.0 < bound < 1e-8
    assert eigenbasis <= bound
    # The a priori bound is three to five orders above what two evaluations of the
    # same functions differ by; the route's own run is the sharper witness. The
    # off-route run is held to a small multiple of the on-route run's departure from
    # the reference, which the two formations' rounding both sit in.
    on_route = max(departure(on, expected), 32 * UNIT)
    record_property("on_route", on_route)
    assert eigenbasis <= 8 * on_route


@pytest.mark.medium
@pytest.mark.parametrize("route", OFF_ROUTES)
def test_the_session_runs_bins_off_the_route_within_the_bound(
    parity_cases, route, monkeypatch
):
    """The session admits the taper off the route, held to the reference's bound.

    ``numpy_ref`` within the bound of
    :func:`test_bins_off_the_route_are_the_reference_s`, and the same run twice to the
    bit on one thread.
    """
    simulation, problem = parity_cases("taper")
    request, grid = simulation._run_request(), simulation.grid
    expected = run(NumpyRefBackend, problem, request, grid)
    off_route(monkeypatch, route)
    _assert_routed(problem, route)
    assert native_session.refusal(problem, request) is None
    case = Case(problem, request, grid, float(np.sum(request.stepper.schedule.steps)))
    context = {"profiles": simulation.profiles, "modes": simulation.modes}
    session, status = session_run(case, context)
    again, repeated = session_run(case, context)
    scales = flow_scales(problem, request, grid, expected)
    formation = max(scales.formation, adams_formation_bound(problem, request, grid))
    bound = ten_step_bound(dataclasses.replace(scales, formation=formation))
    assert status.attempts == repeated.attempts
    if _one_thread():
        np.testing.assert_array_equal(session, again)
    assert departure(session, expected) <= bound


def test_a_bin_off_the_route_past_thirty_squarings_is_refused_by_name(
    parity_cases, monkeypatch
):
    """The augmented exponent's squarings are counted per trial and refused past 30.

    A phase increment of 1e15 rad a step, the same in every channel, puts the shifted
    block's row-sum norm far past ``0.5 2^30``: the trial refuses before it moves the
    state, in the words of the flow action's own refusal.
    """
    simulation, problem = parity_cases("scalar")
    off_route(monkeypatch, "off")
    linear = problem.linear
    assert linear.phi_constant is not None
    fast = dataclasses.replace(
        problem,
        linear=dataclasses.replace(linear, phi_constant=linear.phi_constant + 1e15j),
    )
    with pytest.raises(BackendError, match="would need more than 30 squarings"):
        run(NativeBackend, fast, simulation._run_request(), simulation.grid)


# --- The assembled block functions, one trial at a time -----------------------------


def _resident(simulation: Simulation, problem: Any, request: Any) -> Any:
    """Return a native backend whose resident Adams engine can take a trial directly.

    The engine's trial (``_ResidentAdams.exponential_trial``) is called with no march
    around it, as solprop's host march called it before SEP 47 phase 5 retired that
    march: the state is the launch, loaded into the native state, and every history
    row holds the launch's sample (the engine's own ``start``), so a trial of any
    order reads a full, finite history. A trial's flow and moments depend on its
    step, its width and the operator alone, not on the history it reads.
    """
    backend = NativeBackend(
        problem,
        request,
        grid=simulation.grid,
        profiles=simulation.profiles,
        modes=simulation.modes,
    )
    engine = backend._resident
    assert type(engine).__name__ == "_ResidentAdams"
    field = np.ascontiguousarray(problem.initial.field0, dtype=np.complex128)
    load_native(backend, field, clear_fsal=False)
    engine.start(0.0)
    rows = {
        name: np.broadcast_to(row[:1], row.shape).copy()
        for name, row in engine.snapshot_rows().items()
    }
    engine.load_rows(rows, engine.shape)
    return backend


def _stencil(order: int) -> tuple[list[float], np.ndarray]:
    """Return a trial's nodes and rows over `order` samples, newest first.

    ``adams_core.stencil``'s for a history taken at whole steps before the anchor.
    """
    nodes = [-float(index) for index in range(order)]
    return nodes, np.arange(order, dtype=np.int64)[::-1].copy()


def _block_departure(
    backend: Any,
    block: Any,
    shape: tuple[int, ...],
    z: float,
    dz: float,
    order: int,
    *,
    carrier: bool,
) -> tuple[int, float]:
    """Take one resident trial of `order`; return its count and its block's departure.

    The departure is the largest of each assembled matrix's 2-norm error against
    ``coupled_moments`` on the engine's own exponent and carrier shift, relative to
    that matrix's 2-norm, at least one.
    """
    engine, state = backend._resident, backend._native_state
    nodes, slots = _stencil(order)
    norms = engine.exponential_trial(z, dz, nodes, slots, order, [order])
    assert np.all(np.isfinite(norms))
    count = trial_tables(
        nodes, order, [order], compensated=engine.compensated, helpers=engine.helpers
    )[1].shape[0]
    shift = engine._trial_carrier
    assert bool(shift.size) is carrier
    expected = coupled_moments(
        state.adams_exponent_to_host().reshape(1, *shape),
        count,
        block,
        dz,
        carrier=np.full((1, 1, 1, 1), 1j * float(shift[0])) if carrier else None,
    )
    active = int(np.count_nonzero(block.active))
    native = state.adams_block_to_host().reshape(-1, active, shape[0], shape[0])
    pairs = [(native[0], expected.flow_matrix)] + [
        (native[k + 1], expected.matrices[k]) for k in range(count)
    ]
    worst = 0.0
    for got, want in pairs:
        error = np.linalg.norm(got - want, ord=2, axis=(-2, -1))
        scale = np.maximum(np.linalg.norm(want, ord=2, axis=(-2, -1)), 1.0)
        worst = max(worst, float(np.max(error / scale)))
    return count, worst


@pytest.mark.parametrize("carrier", [False, True], ids=["plain", "carrier"])
def test_the_off_route_block_functions_are_the_reference_s(
    parity_cases, carrier, monkeypatch
):
    """Each trial's assembled flow and moments off the route, against the reference.

    ``Z R_k Z^H`` with the ``(k - 1)!`` scale and, under a carrier frame, the
    ``exp(i sigma)`` on the flow, as the resident engine leaves them in its block after
    a trial (:func:`_resident`) -- of orders one to six at two positions and two
    steps, so counts two to seven -- against ``multistep.coupled_moments`` on the
    engine's own exponent: within 1e-12 of each matrix's 2-norm (at least one), which
    is ``adams_formation_bound``'s share at the case's step, about 1e-11 for both
    evaluations together, with the margin of its norm-wise slack.
    """
    if carrier and DEVICE:
        pytest.skip("device builds refuse the carrier frame by name")
    step = LENGTH / 40
    simulation = build(
        "te_tm",
        schedule=FixedSchedule(np.full(40, step)),
        dz_max=step,
        dz_init=step,
    )
    problem = simulation.translate()
    request = simulation._run_request()
    # A fixed sixth order, so that the trials reach seven moments.
    method = Multistep(
        order=6, variable_order=False, carrier_frame="scalar" if carrier else "off"
    )
    request = request.with_stepper(
        dataclasses.replace(request.stepper, integrator=method)
    )
    off_route(monkeypatch, "off")
    _assert_routed(problem, "off")
    held = backend_module._host_flow(problem)
    assert held is not None
    # One constant section, which every trial reads.
    assert held.vectors.shape[0] == 1
    block = CoupledBlock(
        active=held.active,
        vectors=held.vectors[0],
        triangular=held.triangular[0],
        section=0,
    )
    backend = _resident(simulation, problem, request)
    counts: set[int] = set()
    worst = 0.0
    try:
        for z in (0.0, 0.4 * LENGTH):
            for order in range(1, 7):
                for dz in (step, step / 2):
                    count, departure = _block_departure(
                        backend, block, problem.shape, z, dz, order, carrier=carrier
                    )
                    counts.add(count)
                    worst = max(worst, departure)
    finally:
        backend.close()
    assert counts == set(range(2, 8))
    assert worst <= 1e-12


# --- Refusals: the moments, a carry, a non-finite exponent --------------------------


def _carry_past_thirty(original):
    """Return ``_decomposed_flow`` with each factor a nilpotent ``3e8 / dz`` corner.

    A strictly upper-triangular ``T`` keeps every exponential a finite polynomial,
    however long the step: the moments' row-sum norm, ``|x| + 3e8 + 1``, needs 30
    squarings and passes, and a history sample's carry over two steps, ``6e8``, needs
    31 and is refused.
    """

    def decomposed(problem):
        flow = original(problem)
        schedule = 2e-4 / 10
        corner = np.zeros_like(flow.triangular)
        corner[..., 0, -1] = 3e8 / schedule
        return dataclasses.replace(flow, triangular=corner)

    return decomposed


def test_a_carry_past_thirty_squarings_is_refused_and_keeps_the_state(
    parity_cases, monkeypatch
):
    """A compensated trial's carry is counted too, refused by name before the state
    moves: the session ends failed at the committed state, and the resident engine's
    refused trial leaves the state and its history untouched."""
    simulation, problem = parity_cases("taper")
    request, grid = simulation._run_request(), simulation.grid
    off_route(monkeypatch, "off")
    monkeypatch.setattr(
        backend_module,
        "_decomposed_flow",
        _carry_past_thirty(backend_module._decomposed_flow),
    )
    context = {"profiles": simulation.profiles, "modes": simulation.modes}
    steps = request.stepper.schedule.steps
    # The native session: a failed terminal at the first step's end, by name.
    backend = NativeBackend(problem, request, grid=grid, **context)
    prepared = backend.prepare(problem, request)
    session = prepared.start(entrance(problem))
    try:
        assert isinstance(session, native_session.NativeSession)
        result = drive(Started(session), export=IDENTITY, raises=False)
        terminal = result.terminal
        assert terminal.outcome is Outcome.FAILED
        assert isinstance(terminal.error, BackendError)
        assert "a history sample's carry" in str(terminal.error)
        assert "would need more than 30 squarings" in str(terminal.error)
        assert terminal.z == float(steps[0])
        status = session.status()
        assert status.z == terminal.z and status.attempts == 1
        assert all(np.all(np.isfinite(field)) for field in result.fields)
    finally:
        session.close()
    # The resident engine's own trial (:func:`_resident`): the refused trial, whose
    # second sample carries over two steps, leaves the state and every history row as
    # it met them, and a trial over one step then forms.
    backend = _resident(simulation, problem, request)
    try:
        engine, state = backend._resident, backend._native_state
        assert engine.compensated
        dz = float(steps[0])
        field, rows = state.field_to_host().copy(), engine.snapshot_rows()
        nodes, slots = _stencil(2)
        with pytest.raises(BackendError, match="a history sample's carry"):
            engine.exponential_trial(dz, dz, nodes, slots, 2, [2])
        assert state.field_to_host().tobytes() == field.tobytes()
        for name, row in engine.snapshot_rows().items():
            assert row.tobytes() == rows[name].tobytes(), name
        nodes, slots = _stencil(1)
        norms = engine.exponential_trial(dz, dz, nodes, slots, 1, [1])
        assert np.all(np.isfinite(norms))
    finally:
        backend.close()


# --- More shapes of a flow off the route ---------------------------------------------


class _Sections:
    """``schur_eigenbasis`` with every even section off the route, odd ones on it."""

    def __init__(self):
        self.calls = 0

    def __call__(self, vectors, triangular):
        left, right, eigenvalues, route = schur_eigenbasis(vectors, triangular)
        if self.calls % 2 == 0:
            route = np.zeros_like(route)
            left = np.zeros_like(left)
            right = np.zeros_like(right)
        self.calls += 1
        return left, right, eigenvalues, route


def _held_to_the_bound(setup, runs, carrier=0.0):
    """Assert a run off the route against the reference and the route.

    `setup` is the problem, request and grid; `runs` the final states by name.
    ``flow_parity.ten_step_bound`` at ``adams_formation_bound``, and the eigenbasis
    route's run within eight times its own departure from the reference.
    """
    problem, request, grid = setup
    expected, routed = runs["expected"], runs["routed"]
    on = runs["on"]
    scales = flow_scales(problem, request, grid, expected)
    formation = max(
        scales.formation, adams_formation_bound(problem, request, grid, carrier)
    )
    bound = ten_step_bound(dataclasses.replace(scales, formation=formation))
    assert 0.0 < bound < 1e-8
    assert departure(routed, expected) <= bound
    assert departure(routed, on) <= 8 * max(departure(on, expected), 32 * UNIT)


@pytest.mark.medium
@pytest.mark.parametrize("name", ["taper", "lossless", "gradient"])
def test_sections_on_and_off_the_route_in_one_run(parity_cases, name, monkeypatch):
    """A flow whose sections alternate: every bin off the route, then every bin on."""
    simulation, problem = parity_cases(name)
    request, grid = on_adams(simulation._run_request()), simulation.grid
    on = run(NativeBackend, problem, request, grid)
    expected = run(NumpyRefBackend, problem, request, grid)
    monkeypatch.setattr(
        "solprop_native.backend._HOST_FLOWS", weakref.WeakKeyDictionary()
    )
    monkeypatch.setattr(backend_module, "schur_eigenbasis", _Sections())
    held = backend_module._host_flow(problem)
    assert held is not None and held.route.shape[0] > 1
    per_section = np.all(held.route, axis=1)
    assert np.any(per_section) and not np.all(per_section)
    assert np.array_equal(per_section, ~np.any(~held.route, axis=1))
    routed = run(NativeBackend, problem, request, grid)
    _held_to_the_bound(
        (problem, request, grid),
        {"expected": expected, "routed": routed, "on": on},
    )


@pytest.mark.medium
def test_a_carrier_frame_off_the_route(parity_cases, monkeypatch):
    """The carrier frame's shifted moments and ``exp(sigma)`` flow off the route."""
    if DEVICE:
        pytest.skip("device builds refuse the carrier frame by name")
    simulation, problem = parity_cases("te_tm")
    grid = simulation.grid
    method = Multistep(carrier_frame="scalar")
    request = simulation._run_request()
    request = request.with_stepper(
        dataclasses.replace(request.stepper, integrator=method)
    )
    reference = carrier_reference(problem, method)
    assert reference is not None and reference.active and reference.chord is not None
    steps = request.stepper.schedule.steps
    anchors = np.concatenate(([0.0], np.cumsum(steps)[:-1]))
    carrier = max(
        float(np.max(np.abs(np.imag(reference.chord.exponent(z, z + dz)))))
        for z, dz in zip(anchors, steps, strict=True)
    )
    assert carrier > 0.0
    on = run(NativeBackend, problem, request, grid)
    expected = run(NumpyRefBackend, problem, request, grid)
    off_route(monkeypatch, "off")
    _assert_routed(problem, "off")
    routed = run(NativeBackend, problem, request, grid)
    _held_to_the_bound(
        (problem, request, grid),
        {"expected": expected, "routed": routed, "on": on},
        carrier,
    )


@pytest.mark.medium
def test_a_wide_flow_off_the_route(cases, monkeypatch):
    """17 channels, every other bin off the route: wide rows and augmented bins."""
    case = cases(17)
    problem, request = case.problem, case.run_request
    on = propagate(case, NativeBackend)
    expected = propagate(case, NumpyRefBackend)
    off_route(monkeypatch, "mixed")
    _assert_routed(problem, "mixed")
    routed = propagate(case, NativeBackend)
    linear = 1e3 * propagate(case, NumpyRefBackend, problem.initial.field0 * 1e-3)
    formation = adams_formation_bound(problem, request, case.grid)
    bound = parity_bound(case, expected, linear, formation=formation)
    scale = _scale(case)
    assert 0.0 < bound < 1e-6 * scale
    assert float(np.max(np.abs(routed - expected))) <= bound
    assert float(np.max(np.abs(routed - on))) <= bound
    floor = max(float(np.max(np.abs(on - expected))), 32 * UNIT * scale)
    assert float(np.max(np.abs(routed - on))) <= 8 * floor


def _round_trip(simulation, problem, middle, end):
    """Return a run's state at `end`, and the same run snapshotted at `middle`,
    restored into a fresh backend's session and continued to `end`.

    Each run is a native session over a fixed schedule with planes at `middle` and
    `end`, both edges of it, so the planes take no step of their own.
    """
    request = to_planes(simulation._run_request(), [middle, end])
    context = {
        "grid": simulation.grid,
        "modes": simulation.modes,
        "profiles": simulation.profiles,
    }
    uninterrupted = final_field(
        NativeBackend(problem, request, **context),
        problem,
        bare(request),
        ended=True,
        owns=True,
    )
    first = NativeBackend(problem, request, **context)
    assert type(first._resident).__name__ == "_ResidentAdams"
    taken = []

    def take(session, batch) -> None:
        for event in batch.events:
            if isinstance(event, ObservationReady) and event.plane_index == 0:
                taken.append(session.march_snapshot(event.ticket))

    run_session(first, problem, bare(request), owns=True, between=take)
    [stored] = taken
    assert stored is not None and len(stored.identities) >= 2
    assert math.isclose(stored.z, middle, rel_tol=1e-12)
    second: Any = start_session(
        NativeBackend(problem, request, **context),
        problem,
        bare(request),
        Start(
            field=np.array(stored.state),
            z=stored.z,
            plane_index=1,
            plane_pending=False,
            resume=True,
            march=stored,
        ),
    )
    try:
        assert isinstance(second, native_session.NativeSession)
        assert second.march_refusal is None
        resumed = drive(Started(second), export=IDENTITY)
    finally:
        second.close()
    (last, *_) = reversed(resumed.observations)
    assert math.isclose(last.z, end, rel_tol=1e-12)
    return uninterrupted, resumed.fields[-1]


@pytest.mark.medium
@pytest.mark.parametrize("kind", ["off route", "wide"])
def test_a_snapshot_off_the_route_or_wide_resumes_to_the_bit(
    parity_cases, cases, kind, monkeypatch
):
    """The resident history of a wide or off-route flow, snapshotted and restored.

    The engine's rows round-trip losslessly, so a resumed run takes the uninterrupted
    one's steps on the same arithmetic: the same state to the bit in one process.
    """
    if kind == "wide":
        case = cases(17)
        simulation = wide_simulation(17)
        problem = case.problem
        steps = case.run_request.stepper.schedule.steps
        middle, end = float(np.sum(steps[:5])), float(np.sum(steps))
    else:
        simulation, problem = parity_cases("gradient")
        off_route(monkeypatch, "off")
        _assert_routed(problem, "off")
        steps = simulation._run_request().stepper.schedule.steps
        middle, end = float(np.sum(steps[:5])), float(np.sum(steps))
    uninterrupted, resumed = _round_trip(simulation, problem, middle, end)
    np.testing.assert_array_equal(resumed, uninterrupted)
