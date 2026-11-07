# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 45 phase 2: a Runge-Kutta trial's stage rows and its norm formed natively.

Where every row a trial reads is one of the uploaded stage tables', the native state
evaluates each stage's scalars at ``z + c_i dz`` itself (``stage::rows``), as
``_trial_arrays`` packs the bundles of ``_stage_bundles``: the trial's partials are the
host-packed trial's bit for bit. The norm is reduced natively for solprop's own weak
scale, its sums in order (Q1): it agrees with the host's NumPy reduction to rounding,
and a weighted controller, or one with its own ``scale``, keeps the host's.
"""

import dataclasses
import math
import warnings

import numpy as np
import pytest

from capability_routes import ROUTES
from solprop.backends import available
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
)
"""The routes whose trial rows are all the stage tables': every operator and prefactor
form but a z table's, with no coupled flow."""

HOST = (
    "general guide (moving)",
    "capillary (coherent wall)",
    "capillary (taper coupling)",
)
"""The routes beside a coupled flow, whose rows the host forms."""


def runge_kutta(route: str, controller: PIController | None = None):
    """Return a route's backend on the Runge-Kutta march, launched."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = ROUTES[route](None)
        settings = dataclasses.replace(simulation.settings, integrator=None)
        if controller is not None:
            settings = dataclasses.replace(settings, controller=controller)
        simulation.settings = settings
        problem = simulation.translate()
    backend = native.NativeBackend(
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    backend.set_initial(problem.initial.field0, None, z=problem.initial.z)
    return backend, problem


def steps(problem) -> list[tuple[float, float]]:
    """Return a few anchors and steps across the run, a breakpoint among them."""
    start, end = float(problem.initial.z), float(problem.stepper.z_end)
    span = end - start
    edges = [float(edge) for edge in problem.linear.phi_breakpoints[1:-1]]
    anchors = [start, start + 0.37 * span, *(edge - 1e-3 * span for edge in edges[:2])]
    return [(z, min(0.05 * span, end - z)) for z in anchors if z < end]


@pytest.mark.parametrize("route", ROWS)
def test_a_staged_trial_is_the_host_packed_trial(route, monkeypatch):
    backend, problem = runge_kutta(route)
    try:
        assert backend._staged_rows()
        for z, dz in steps(problem):
            staged = np.asarray(backend._prepare_trial(z, dz))
            with monkeypatch.context() as patch:
                patch.setattr(backend_module, "_RK_ROWS", "host")
                packed = np.asarray(backend._prepare_trial(z, dz))
            assert np.array_equal(staged, packed), (z, dz)
    finally:
        backend.close()


@pytest.mark.parametrize("route", HOST)
def test_a_coupled_flow_s_rows_stay_the_host_s(route):
    backend, _ = runge_kutta(route)
    try:
        assert not backend._staged_rows()
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


class _Scaled(PIController):
    """A controller whose scale is its own."""

    def scale(self, norm_y, norm_y_new):
        return 2.0 * super().scale(norm_y, norm_y_new)


@pytest.mark.parametrize(
    "controller",
    [
        _Scaled(rtol=1e-6, atol=0.0),
        PIController(rtol=1e-6, atol=0.0, weights=None),
    ],
    ids=["own_scale", "stock"],
)
def test_only_solprop_s_weak_scale_is_reduced_natively(controller, monkeypatch):
    backend, problem = runge_kutta("gnlse", controller)
    calls: list[int] = []
    score = core.rk_score

    def counted(partials, rtol, atol):
        calls.append(1)
        return score(partials, rtol, atol)

    try:
        monkeypatch.setattr(core, "rk_score", counted)
        z, dz = steps(problem)[0]
        partials = backend._prepare_trial(z, dz)
        norm = backend._score_trial(partials)
        expected = _host_norm(np.asarray(partials), 1e-6, 0.0)
        if isinstance(controller, _Scaled):
            assert not calls
            assert norm == expected / 2.0
        else:
            assert calls
            assert norm == pytest.approx(expected, rel=1e-14)
    finally:
        backend.close()
