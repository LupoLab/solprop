# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The coupled flow's exponential Adams march, resident wherever a CPU build runs it.

solprop SEP 47 phase 4 (section 6.3, the coupled-flow row): a coupled block wider than
16 channels, and a block with bins off its flow's eigenbasis route, step on the
resident Adams engine of a CPU or a device build, and the native session serves them,
where solprop's host engine stepped them over the backend's hooks before.

The wide cases are lossless helium tapers with taper coupling and every bin on the
route, over ten fixed steps short enough that the formation bound of
``tests/flow_parity.py`` stays far below one; the resident engine is held to
``numpy_ref`` and to solprop's host engine on the same backend within the ten-step
bound of ``tests/driver/test_native_coupled_flow.py`` (``parity_bound``). The cases off
the route are ``flow_parity``'s eight, every bin or every other one routed to the
augmented exponential, held to ``numpy_ref``, the host engine and the same backend on
the eigenbasis route within ``flow_parity.ten_step_bound`` at
``adams_formation_bound``. Every bound is written before any measurement; the session
is held to ``numpy_ref`` and to the host engine within the same bound, and is the same
run twice to the bit on one thread.
"""

import dataclasses
import os
import weakref
from typing import Any

import numpy as np
import pytest

from driver.test_coherent_capillary import GRID, STEPS, WEAK_COLLAR
from driver.test_native_coupled_flow import (
    ENERGY,
    PRESSURE,
    Case,
    parity_bound,
    propagate,
)
from flow_parity import (
    CASES,
    UNIT,
    adams_formation_bound,
    build,
    departure,
    flow_scales,
    run,
    ten_step_bound,
)
from sessions.executors import IDENTITY, Started, drive, entrance
from solprop.backends.events import Outcome
from solprop.backends.exponential import carrier_reference
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import available
from solprop.driver import Simulation
from solprop.exceptions import (
    BackendError,
    GrazingIncidenceWarning,
    MaterialRangeWarning,
)
from solprop.profiles import PiecewiseLinear
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

WIDE_LENGTH = 2e-5
"""Length of the wide cases, m: ten steps of 2 um.

At 24 channels ``||h T||`` stays near two, so that the scaling and squaring share of
the formation bound, ``exp(2 ||h T||)`` times the rounding, stays below a millionth
(a step of 20 um puts it above one)."""

CONTEXT: dict[int, dict[str, Any]] = {}
"""Each built case's z profiles and modes, which its plane statistics read."""

WIDE = (17, 24)
"""Channel counts past the 16 a host task holds on its stack."""


def wide_simulation(modes: int, backend: str = "numpy_ref") -> Simulation:
    """Return the lossless taper with `modes` channels, ten fixed steps.

    Parameters
    ----------
    modes : int
        The channels.
    backend : str, optional
        The backend the simulation names.

    Returns
    -------
    solprop.driver.Simulation
        The simulation.
    """
    length = WIDE_LENGTH
    return Simulation.from_capillary(
        PiecewiseLinear.from_stages((0.0, length), [50e-6, 48e-6]),
        length,
        "He",
        PRESSURE,
        wall=None,
        loss=False,
        cladding=1.45,
        modes=modes,
        wavelength=800e-9,
        grid=GRID,
        absorber=WEAK_COLLAR,
        fwhm=30e-15,
        energy=ENERGY,
        n_save=2,
        backend=backend,
        dz_max=length / STEPS,
        dz_init=length / STEPS,
        schedule=FixedSchedule(np.full(STEPS, length / STEPS)),
        plasma=False,
        raman=False,
        noise=None,
    )


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


def on_host_engine(monkeypatch) -> None:
    """Make the backend step every coupled flow on solprop's host engine."""
    monkeypatch.setattr(
        backend_module, "_flow_on_host", lambda *_: "the test's host engine"
    )


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
        assert type(backend._adams._engine).__name__ == "_ResidentAdams"
        assert backend._adams.native
        assert backend._adams.refusal is None
        # A CPU build's host passes' channel rows: one per bin, nine samples of every
        # channel; a device's passes hold none.
        if not DEVICE:
            rows = backend.memory_ledger()["adams_history"]
            assert rows >= problem.grids.n_omega * 9 * modes * 16
    finally:
        backend.close()


# The host engine runs solprop's march, with the warning
# tests/backends/test_native_march_admission.py asserts (SEP 39).
@pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning")
@pytest.mark.parametrize(
    "modes",
    [
        pytest.param(17, marks=pytest.mark.medium),
        pytest.param(24, marks=pytest.mark.slow),
    ],
)
def test_ten_wide_steps_are_the_reference_s_and_the_host_engine_s(
    cases, modes, monkeypatch, record_property
):
    """The resident engine against ``numpy_ref`` and the host engine, within the bound.

    The bound (``parity_bound``), stated first: the flow's formation bound ``eps_form``
    at the case's step, the carries' ``2 n kappa u + kappa eps_form`` and the shared
    work's rounding, per step and per stage, over ten steps. The host engine runs on
    the same backend's hooks, so it differs from the resident engine by no more than
    either differs from ``numpy_ref``: the same bound holds both.
    """
    case = cases(modes)
    expected = propagate(case, NumpyRefBackend)
    linear = 1e3 * propagate(case, NumpyRefBackend, case.problem.initial.field0 * 1e-3)
    bound = parity_bound(case, expected, linear)
    resident = propagate(case, NativeBackend)
    on_host_engine(monkeypatch)
    host = propagate(case, NativeBackend)
    scale = _scale(case)
    rho = float(np.max(np.abs(expected - linear))) / scale / STEPS
    assert rho > 1e-5, "the nonlinear half must be exercised through the flow"
    reference = float(np.max(np.abs(resident - expected)))
    engine = float(np.max(np.abs(resident - host)))
    record_property("bound", bound / scale)
    record_property("reference", reference / scale)
    record_property("host_engine", engine / scale)
    assert 0.0 < bound < 1e-6 * scale
    assert reference <= bound
    assert engine <= bound


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


# The host engine runs solprop's march, with the warning
# tests/backends/test_native_march_admission.py asserts (SEP 39).
@pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning")
@pytest.mark.medium
@pytest.mark.parametrize("modes", WIDE)
def test_the_session_runs_a_wide_flow_within_the_bound(cases, modes, monkeypatch):
    """The session is ``numpy_ref``'s and the host engine's in bound, and repeatable.

    SEP 47's phase 4 evidence, kept past the bridge: the session is held to
    ``parity_bound`` against ``numpy_ref`` and against solprop's host engine on the
    same backend, and two sessions are the same run to the bit on one thread.
    """
    case = cases(modes)
    session, status = session_run(case)
    again, repeated = session_run(case)
    expected = propagate(case, NumpyRefBackend)
    linear = 1e3 * propagate(case, NumpyRefBackend, case.problem.initial.field0 * 1e-3)
    bound = parity_bound(case, expected, linear)
    with monkeypatch.context() as patched:
        on_host_engine(patched)
        host = propagate(case, NativeBackend)
    assert status.attempts == repeated.attempts == STEPS
    if _one_thread():
        np.testing.assert_array_equal(session, again)
    assert 0.0 < bound < 1e-6 * _scale(case)
    assert float(np.max(np.abs(session - expected))) <= bound
    assert float(np.max(np.abs(session - host))) <= bound


# --- Bins off the flow's eigenbasis route ------------------------------------------

ROUTES = ("off", "mixed")
"""Every bin off the eigenbasis route, and every other active bin of every section."""


def _every_other(vectors, triangular):
    """``schur_eigenbasis`` with every other bin taken off the route."""
    left, right, eigenvalues, route = schur_eigenbasis(vectors, triangular)
    route = route.copy()
    route[1::2] = False
    left[~route] = 0.0
    right[~route] = 0.0
    return left, right, eigenvalues, route


def off_route(monkeypatch, route: str) -> None:
    """Route a problem's bins off the eigenbasis, all of them or every other one.

    Each problem's decomposition is kept while it lives, routed as it was when formed:
    a fresh cache forms this route's.
    """
    monkeypatch.setattr(
        "solprop_native.backend._HOST_FLOWS", weakref.WeakKeyDictionary()
    )
    if route == "off":
        monkeypatch.setattr(
            "solprop.translate.coupled.eigenbasis_limit", lambda n_modes: 0.0
        )
    else:
        monkeypatch.setattr(backend_module, "schur_eigenbasis", _every_other)


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


def on_adams(request):
    """Return a request on the exponential Adams march, the split block's included."""
    if isinstance(request.stepper.integrator, Multistep):
        return request
    return request.with_stepper(
        dataclasses.replace(request.stepper, integrator=Multistep())
    )


def _assert_routed(problem, route: str) -> None:
    held = backend_module._host_flow(problem)
    assert held is not None
    assert not np.all(held.route) and (route == "off") == (not np.any(held.route))
    # The native passes read the Schur factor's upper triangle alone.
    assert np.all(np.tril(held.triangular, -1) == 0.0)


@pytest.mark.medium
@pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning")
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("name", CASES)
def test_bins_off_the_route_are_the_reference_s_and_the_host_engine_s(
    parity_cases, name, route, monkeypatch, record_property
):
    """Ten fixed Adams steps with bins off the route, within ``B_10``.

    The bound, stated first: ``flow_parity.ten_step_bound`` with its formation bound
    the larger of the actions' and the augmented exponential's
    (``adams_formation_bound``), against ``numpy_ref`` (SciPy's ``expm`` on every
    bin), against solprop's host engine on the same backend (the eigenbasis on the
    route's bins, ``expm`` off it), and against the resident engine with every bin on
    the eigenbasis route, each a pair of implementations of the same ten steps.
    """
    simulation, problem = parity_cases(name)
    request, grid = on_adams(simulation._run_request()), simulation.grid
    on = run(NativeBackend, problem, request, grid)
    expected = run(NumpyRefBackend, problem, request, grid)
    off_route(monkeypatch, route)
    _assert_routed(problem, route)
    backend = NativeBackend(problem, request, grid=grid)
    try:
        assert type(backend._adams._engine).__name__ == "_ResidentAdams"
    finally:
        backend.close()
    resident = run(NativeBackend, problem, request, grid)
    with monkeypatch.context() as patched:
        on_host_engine(patched)
        host = run(NativeBackend, problem, request, grid)
    scales = flow_scales(problem, request, grid, expected)
    formation = max(scales.formation, adams_formation_bound(problem, request, grid))
    bound = ten_step_bound(dataclasses.replace(scales, formation=formation))
    measured = {
        "reference": departure(resident, expected),
        "host_engine": departure(resident, host),
        "eigenbasis": departure(resident, on),
    }
    record_property("bound", bound)
    for key, value in measured.items():
        record_property(key, value)
    assert 0.0 < bound < 1e-8
    assert max(measured.values()) <= bound
    # The a priori bound is three to five orders above what two evaluations of the
    # same functions differ by; the route's own run is the sharper witness. The
    # off-route run is held to a small multiple of the on-route run's departure from
    # the reference, which the two formations' rounding both sit in.
    on_route = max(departure(on, expected), 32 * UNIT)
    record_property("on_route", on_route)
    assert measured["eigenbasis"] <= 8 * on_route


@pytest.mark.medium
@pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning")
@pytest.mark.parametrize("route", ROUTES)
def test_the_session_runs_bins_off_the_route_within_the_bound(
    parity_cases, route, monkeypatch
):
    """The session admits the taper off the route, held to the reference's bound.

    ``numpy_ref`` and the host engine within the bound of
    :func:`test_bins_off_the_route_are_the_reference_s_and_the_host_engine_s`, and the
    same run twice to the bit on one thread.
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
    with monkeypatch.context() as patched:
        on_host_engine(patched)
        host = run(NativeBackend, problem, request, grid)
    scales = flow_scales(problem, request, grid, expected)
    formation = max(scales.formation, adams_formation_bound(problem, request, grid))
    bound = ten_step_bound(dataclasses.replace(scales, formation=formation))
    assert status.attempts == repeated.attempts
    if _one_thread():
        np.testing.assert_array_equal(session, again)
    assert departure(session, expected) <= bound
    assert departure(session, host) <= bound


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
    moves, and the session stays usable at the committed state."""
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
        result = drive(Started(session), export=IDENTITY)
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
    """Assert a run off the route against the reference, the host engine and the route.

    `setup` is the problem, request and grid; `runs` the final states by name.
    ``flow_parity.ten_step_bound`` at ``adams_formation_bound``, and the eigenbasis
    route's run within eight times its own departure from the reference.
    """
    problem, request, grid = setup
    expected, routed = runs["expected"], runs["routed"]
    on, host = runs["on"], runs["host"]
    scales = flow_scales(problem, request, grid, expected)
    formation = max(
        scales.formation, adams_formation_bound(problem, request, grid, carrier)
    )
    bound = ten_step_bound(dataclasses.replace(scales, formation=formation))
    assert 0.0 < bound < 1e-8
    assert departure(routed, expected) <= bound
    assert departure(routed, host) <= bound
    assert departure(routed, on) <= 8 * max(departure(on, expected), 32 * UNIT)


@pytest.mark.medium
@pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning")
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
    with monkeypatch.context() as patched:
        on_host_engine(patched)
        host = run(NativeBackend, problem, request, grid)
    _held_to_the_bound(
        (problem, request, grid),
        {"expected": expected, "routed": routed, "on": on, "host": host},
    )


@pytest.mark.medium
@pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning")
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
    with monkeypatch.context() as patched:
        on_host_engine(patched)
        host = run(NativeBackend, problem, request, grid)
    _held_to_the_bound(
        (problem, request, grid),
        {"expected": expected, "routed": routed, "on": on, "host": host},
        carrier,
    )


@pytest.mark.medium
@pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning")
def test_a_wide_flow_off_the_route(cases, monkeypatch):
    """17 channels, every other bin off the route: wide rows and augmented bins."""
    case = cases(17)
    problem, request = case.problem, case.run_request
    on = propagate(case, NativeBackend)
    with monkeypatch.context() as patched:
        on_host_engine(patched)
        host_on = propagate(case, NativeBackend)
    off_route(monkeypatch, "mixed")
    _assert_routed(problem, "mixed")
    routed = propagate(case, NativeBackend)
    with monkeypatch.context() as patched:
        on_host_engine(patched)
        host = propagate(case, NativeBackend)
    linear = 1e3 * propagate(case, NativeBackend, problem.initial.field0 * 1e-3)
    formation = adams_formation_bound(problem, request, case.grid)
    bound = parity_bound(case, host, linear, formation=formation)
    scale = _scale(case)
    assert 0.0 < bound < 1e-6 * scale
    assert float(np.max(np.abs(routed - host))) <= bound
    assert float(np.max(np.abs(routed - on))) <= bound
    floor = max(float(np.max(np.abs(on - host_on))), 32 * UNIT * scale)
    assert float(np.max(np.abs(routed - on))) <= 8 * floor


def _round_trip(simulation, problem, middle, end):
    """Return a run's state at `end`, and the same run snapshotted at `middle`,
    restored into a fresh backend and continued to `end`."""
    request = simulation._run_request()
    context = {
        "grid": simulation.grid,
        "modes": simulation.modes,
        "profiles": simulation.profiles,
    }
    whole = NativeBackend(problem, request, **context)
    try:
        whole.set_initial(problem.initial.field0, None)
        whole.propagate_to(end)
        uninterrupted = whole.observe().require_field().copy()
    finally:
        whole.close()
    first = NativeBackend(problem, request, **context)
    try:
        assert type(first._adams._engine).__name__ == "_ResidentAdams"
        first.set_initial(problem.initial.field0, None)
        first.propagate_to(middle)
        stored = first.march_snapshot()
    finally:
        first.close()
    assert stored is not None and len(stored.identities) >= 2
    second = NativeBackend(problem, request, **context)
    try:
        second.set_initial(problem.initial.field0, None, z=stored.z)
        assert second.restore_march(stored) is None
        second.propagate_to(end)
        resumed = second.observe().require_field().copy()
    finally:
        second.close()
    return uninterrupted, resumed


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
