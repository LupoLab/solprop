# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 45 phase 2: a Runge-Kutta trial's stage rows and its norm formed natively.

Where every row a trial reads is one of the uploaded stage tables', the native state
evaluates each stage's scalars at ``z + c_i dz`` itself (``stage::rows``); the
host-packed trial it was measured against bit for bit was deleted in SEP 47's phase 5,
and the stage scalars are held to solprop's bundle in ``test_native_stage_tables.py``.
The norm is reduced natively for solprop's own
weak scale, its sums in order (Q1), and for a weighted one (phase 4): each agrees with
the host's NumPy reduction to rounding.

The trials are the native state's own (``rk_trial``), called on the backend's state
directly, and a run's norms are what its session reports for each attempt: SEP 47 phase
5 retired the backend's driver calls.
"""

import dataclasses
import math
import warnings

import numpy as np
import pytest

from capability_routes import ROUTES
from native_flow_guards import check_flow_conditioning, flow_squarings
from sessions.runs import load_native, native_trial, run
from solprop.backends import available
from solprop.exceptions import BackendError
from solprop.stepper.controller import PIController
from solprop.translate.stage import scalars

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif("native" not in available(), reason="native is not installed"),
]

native = pytest.importorskip("solprop_native")
backend_module = pytest.importorskip("solprop_native.backend")
core = pytest.importorskip("solprop_native._core")

ROWS = (
    "gnlse",
    "capillary (mode-averaged)",
    "capillary (envelope)",
    "capillary (multimode)",
    "capillary (pressure gradient)",
    "general guide (constant)",
    "radial",
    "cartesian",
    "line",
    "planar",
    "general guide (moving)",
    "capillary (coherent wall)",
    "capillary (taper coupling)",
)
"""The routes whose trial rows are all the stage tables': every operator and prefactor
form but a z table's, and beside a coupled flow (SEP 45 phase 4), whose actions the
state forms behind the transcribed guards."""

FLOWS = ("general guide (moving)", "capillary (coherent wall)")
"""Routes beside a coupled flow, with a frozen residual and without."""


def runge_kutta(route: str, controller: PIController | None = None):
    """Return a route's backend on the Runge-Kutta march, launched, with its problem
    and request."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = ROUTES[route](None)
    return launched(simulation, controller)


def launched(simulation, controller: PIController | None = None):
    """Return a simulation's backend on the Runge-Kutta march, launched, with its
    problem and request: the launch is loaded into the native state."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        settings = dataclasses.replace(simulation.settings, integrator=None)
        if controller is not None:
            settings = dataclasses.replace(settings, controller=controller)
        simulation.settings = settings
        problem = simulation.translate()
        run_request = simulation._run_request()
    backend = native.NativeBackend(
        problem,
        run_request,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    load_native(backend, problem.initial.field0)
    return backend, problem, run_request


def _nodes(run_request) -> list[float]:
    return [float(node) for node in run_request.stepper.tableau.nodes]


def steps(problem, run_request) -> list[tuple[float, float]]:
    """Return a few anchors and steps across the run, a breakpoint among them."""
    start, end = float(problem.initial.z), float(run_request.stepper.z_end)
    span = end - start
    edges = [float(edge) for edge in problem.linear.phi_breakpoints[1:-1]]
    anchors = [start, start + 0.37 * span, *(edge - 1e-3 * span for edge in edges[:2])]
    return [(z, min(0.05 * span, end - z)) for z in anchors if z < end]


def _refusal(call) -> str:
    # The native state raises its refusal as the extension's own class; the backend
    # gives it solprop's name with the same message.
    with pytest.raises((BackendError, core.BackendRefusal)) as caught:
        call()
    return str(caught.value)


@pytest.mark.parametrize("route", FLOWS)
def test_the_native_flow_guards_refuse_as_the_host_s(route, monkeypatch):
    """Each transcribed guard refuses where solprop's does, with its message.

    The guards are reached by handing the state a certificate, or a Taylor scale, the
    step exceeds: the conditioning measured and the squarings counted are the host's.
    """
    backend, problem, run_request = runge_kutta(route)
    try:
        backend._staged_rows()
        flow = backend._flow
        state = backend._native_state
        z, dz = steps(problem, run_request)[0]
        nodes = _nodes(run_request)
        modes = int(flow.vectors.shape[-1])
        allowance = backend_module._flow_allowance(modes)
        section = int(scalars(problem, z, z + dz, helpers=backend.helpers).flow_section)
        fractions = np.array([*nodes, 1.0])
        measured = np.asarray(state.form_flow_actions(section, dz, fractions))[0]
        if measured.size:
            halved = dataclasses.replace(flow, condition_bound=flow.condition_bound / 2)
            expected = _refusal(
                lambda: check_flow_conditioning(
                    backend_module, halved, section, measured, dz
                )
            )
            state.flow_guard_configure(
                [float(v) for v in flow.row_sum],
                [float(v) for v in np.ravel(halved.condition_bound)],
                backend_module._TAYLOR_SCALE,
                backend_module._MAX_FLOW_SQUARINGS,
                allowance,
            )
            assert _refusal(lambda: native_trial(backend, z, dz)) == expected
        tiny = 1e-12
        # Both refuse at the first stage length past the scale, in the nodes' order.
        first = next(
            float(f) * dz * float(flow.row_sum[section]) for f in fractions if f
        )
        monkeypatch.setattr(backend_module, "_TAYLOR_SCALE", tiny)
        expected = _refusal(lambda: flow_squarings(backend_module, first))
        state.flow_guard_configure(
            [float(v) for v in flow.row_sum],
            [float(v) for v in np.ravel(flow.condition_bound)],
            tiny,
            backend_module._MAX_FLOW_SQUARINGS,
            allowance,
        )
        assert _refusal(lambda: native_trial(backend, z, dz)) == expected
    finally:
        backend.close()


def _host_norm(partials: np.ndarray, rtol: float, atol: float) -> float:
    """The host's reduction, before SEP 45 phase 2."""
    old = math.sqrt(float(np.asarray(partials[0]).sum()))
    new = math.sqrt(float(np.asarray(partials[1]).sum()))
    residual = math.sqrt(float(np.asarray(partials[2]).sum()))
    return residual / (rtol * max(old, new) + atol)


@pytest.mark.parametrize("slots", [1, 3, 8, 37])
@pytest.mark.parametrize("atol", [0.0, 1e-20, 0.3])
def test_the_native_norm_is_the_host_s_to_rounding(slots, atol):
    rng = np.random.default_rng(slots)
    partials = np.ascontiguousarray(rng.uniform(0.0, 2.0, size=(3, slots)))
    for rtol in (1e-8, 1e-3):
        found = core.rk_score(partials, rtol, atol)
        expected = _host_norm(partials, rtol, atol)
        # Only the order of each row's sum differs: a few ulps of each sum.
        assert found == pytest.approx(expected, rel=8 * slots * np.finfo(float).eps)
        if slots < 8:
            # Below NumPy's pairwise blocking its sum is sequential too.
            assert found == expected


def _first_attempt(route: str, controller: PIController):
    """Run a route in its session; return its first attempt's trial partials, step
    and the norm the session reported for it.

    The first attempt starts at the launch, so its trial is retaken on the loaded
    launch: the same state, position and step.
    """
    backend, problem, run_request = runge_kutta(route, controller)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            landed = run(backend, problem, run_request)
        history = next(one for one in landed.histories if one.n_steps)
        dz, norm = float(history.dz[0]), float(history.error_norm[0])
        start = float(problem.initial.z)
        assert float(history.z[0]) == start + dz
        load_native(backend, problem.initial.field0)
        partials = native_trial(backend, start, dz)
        weighted = None
        if controller.weights is not None:
            old = math.sqrt(float(partials[0].sum()))
            new = math.sqrt(float(partials[1].sum()))
            weighted = np.asarray(
                backend._native_state.weighted_error(
                    np.ascontiguousarray(controller.scale(old, new), dtype=np.float64)
                )
            )
    finally:
        backend.close()
    return partials, weighted, norm


def test_the_weak_scale_is_reduced_natively():
    """A run's norm, PIController's scale, is the host's reduction to its rounding.

    A controller with a scale of its own is refused where the backend is built
    (solprop SEP 47 section 6.3, `test_native_rk_admission`), so every trial's norm is
    reduced natively: here the session's own norm of its first attempt, against the
    host's NumPy reduction of that trial's partials.
    """
    controller = PIController(rtol=1e-6, atol=0.0, weights=None)
    partials, _, norm = _first_attempt("gnlse", controller)
    expected = _host_norm(partials, 1e-6, 0.0)
    assert expected > 0.0
    assert norm == pytest.approx(expected, rel=1e-14)
    assert core.rk_score(np.ascontiguousarray(partials), 1e-6, 0.0) == pytest.approx(
        expected, rel=1e-14
    )


def test_a_weighted_norm_is_the_host_s_to_rounding():
    """The weighted scale and its reduction natively (SEP 45 phase 4), in order.

    The session's norm of its first attempt, against the host's sum of the trial's
    residual divided bin by bin by the controller's own scale.
    """
    backend, problem, _ = runge_kutta("gnlse")
    n_omega = problem.grids.n_omega
    backend.close()
    weights = np.linspace(0.05, 1.0, n_omega)
    controller = PIController(rtol=1e-6, atol=1e-12, weights=weights)
    _, weighted, norm = _first_attempt("gnlse", controller)
    assert weighted is not None
    expected = math.sqrt(float(weighted.sum()))
    assert expected > 0.0
    assert norm == pytest.approx(expected, rel=1e-13)
