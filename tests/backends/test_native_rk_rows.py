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
"""

import dataclasses
import math
import warnings

import numpy as np
import pytest

from capability_routes import ROUTES
from native_flow_guards import check_flow_conditioning, flow_squarings
from solprop.backends import available
from solprop.exceptions import BackendError
from solprop.stepper.controller import PIController

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
    """Return a route's backend on the Runge-Kutta march, launched."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = ROUTES[route](None)
    return launched(simulation, controller)


def launched(simulation, controller: PIController | None = None):
    """Return a simulation's backend on the Runge-Kutta march, launched."""
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
    backend.set_initial(problem.initial.field0, None, z=problem.initial.z)
    return backend, problem


def steps(problem, run_request) -> list[tuple[float, float]]:
    """Return a few anchors and steps across the run, a breakpoint among them."""
    start, end = float(problem.initial.z), float(run_request.stepper.z_end)
    span = end - start
    edges = [float(edge) for edge in problem.linear.phi_breakpoints[1:-1]]
    anchors = [start, start + 0.37 * span, *(edge - 1e-3 * span for edge in edges[:2])]
    return [(z, min(0.05 * span, end - z)) for z in anchors if z < end]


def _refusal(call) -> str:
    with pytest.raises(BackendError) as caught:
        call()
    return str(caught.value)


@pytest.mark.parametrize("route", FLOWS)
def test_the_native_flow_guards_refuse_as_the_host_s(route, monkeypatch):
    """Each transcribed guard refuses where solprop's does, with its message.

    The guards are reached by handing the state a certificate, or a Taylor scale, the
    step exceeds: the conditioning measured and the squarings counted are the host's.
    """
    backend, problem = runge_kutta(route)
    try:
        backend._staged_rows()
        flow = backend._flow
        state = backend._native_state
        z, dz = steps(problem, backend.request)[0]
        nodes = backend._rk_nodes
        modes = int(flow.vectors.shape[-1])
        allowance = backend_module._flow_allowance(modes)
        section = int(backend._stage_scalars(z, z + dz).flow_section)
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
            assert _refusal(lambda: backend._prepare_trial(z, dz)) == expected
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
        assert _refusal(lambda: backend._prepare_trial(z, dz)) == expected
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


def test_the_weak_scale_is_reduced_natively(monkeypatch):
    # A controller with a scale of its own is refused where the backend is built
    # (solprop SEP 47 section 6.3, `test_native_rk_admission`), so every trial's norm
    # is reduced natively, by PIController's scale, to the host's reduction's rounding.
    controller = PIController(rtol=1e-6, atol=0.0, weights=None)
    backend, problem = runge_kutta("gnlse", controller)
    calls: list[int] = []
    score = core.rk_score

    def counted(partials, rtol, atol):
        calls.append(1)
        return score(partials, rtol, atol)

    try:
        monkeypatch.setattr(core, "rk_score", counted)
        z, dz = steps(problem, backend.request)[0]
        partials = backend._prepare_trial(z, dz)
        norm = backend._score_trial(partials)
        expected = _host_norm(np.asarray(partials), 1e-6, 0.0)
        assert calls
        assert norm == pytest.approx(expected, rel=1e-14)
    finally:
        backend.close()


def test_a_weighted_norm_is_the_host_s_to_rounding():
    """The weighted scale and its reduction natively (SEP 45 phase 4), in order."""
    backend, problem = runge_kutta("gnlse")
    n_omega = problem.grids.n_omega
    weights = np.linspace(0.05, 1.0, n_omega)
    controller = PIController(rtol=1e-6, atol=1e-12, weights=weights)
    backend.close()
    backend, problem = runge_kutta("gnlse", controller)
    try:
        z, dz = steps(problem, backend.request)[0]
        partials = np.asarray(backend._prepare_trial(z, dz))
        found = backend._score_trial(partials)
        old = math.sqrt(float(partials[0].sum()))
        new = math.sqrt(float(partials[1].sum()))
        scale = controller.scale(old, new)
        weighted = backend._native_state.weighted_error(
            np.ascontiguousarray(scale, dtype=np.float64)
        )
        expected = math.sqrt(float(np.asarray(weighted).sum()))
        assert found == pytest.approx(expected, rel=1e-13)
    finally:
        backend.close()
