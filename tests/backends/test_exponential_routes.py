# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 16 phase 2: the exponential formulation on every constant-operator route.

The only route-specific part of the exponential formulation is its source sample,
``StepDriver._adams_lab_source``: one evaluation of the route's own source pipeline,
untransported. On each route it must be the source the transported formulation
evaluates when the anchor is the sample's own position, where the propagator is one;
and a short run must reproduce the Runge-Kutta march's field. The routes are the
route builders (``tests/capability_routes.py``), short and weak, so the
runs check the linear flow, the separable and modal moments and the lifecycle on
each route; the nonlinear integration is checked on the soliton and scalar problems.
"""

from typing import cast

import numpy as np
import pytest

from bounds import (
    AccuracyEnvelope,
    assert_within,
    departure,
    device_backend,
    fft_bound,
)
from capability_routes import ROUTES
from solprop.backends import available
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.stepper import PIController
from solprop.stepper.multistep import Multistep
from solprop.stepper.runge_kutta import RungeKutta

pytestmark = pytest.mark.medium

CONSTANT_ROUTES = [
    "gnlse",
    "capillary (mode-averaged)",
    "capillary (multimode)",
    "general guide (constant)",
    "radial",
    "cartesian",
    "line",
    "planar",
]


def translated(route: str, integrator, rtol: float):
    """Return a route's simulation and problem with an integrator and tolerance."""
    simulation = ROUTES[route](None)
    settings = simulation.settings
    simulation.settings = type(settings)(
        **{
            **{name: getattr(settings, name) for name in settings.__dataclass_fields__},
            "controller": PIController(
                rtol=rtol, atol=cast("PIController", settings.controller).atol
            ),
            "integrator": integrator,
        }
    )
    return simulation, simulation.translate()


@pytest.mark.parametrize("route", CONSTANT_ROUTES)
def test_the_lab_frame_sample_is_the_route_source(route):
    simulation, problem = translated(route, Multistep(formulation="exponential"), 1e-6)
    backend = NumpyRefBackend(problem, grid=simulation.grid, modes=simulation.modes)
    try:
        rng = np.random.default_rng(5)
        field = problem.initial.field0 * (1.0 + 0.1 * rng.standard_normal())
        position = 0.3 * float(problem.stepper.save_planes[-1])
        lab = backend._adams_lab_source(field, position)
        transported, propagator = backend._adams_source(field, position, position)
    finally:
        backend.close()
    # At equal anchor and position the transported source is multiplied by
    # exp(-0) = 1 exactly, a structural identity; the two samples are the same
    # evaluation repeated, within one evaluation's transform round trip.
    np.testing.assert_array_equal(propagator, np.ones_like(propagator))
    assert_within(transported, lab, fft_bound(problem.grids.n_to), what="source")


@pytest.mark.parametrize("route", CONSTANT_ROUTES)
def test_a_short_run_reproduces_runge_kutta(route):
    def final(integrator, rtol):
        simulation, problem = translated(route, integrator, rtol)
        backend = NumpyRefBackend(problem, grid=simulation.grid, modes=simulation.modes)
        try:
            backend.set_initial(problem.initial.field0, None)
            for plane in problem.stepper.save_planes:
                backend.propagate_to(float(plane))
            return backend.observe().require_field()
        finally:
            backend.close()

    reference = final(RungeKutta(), 1e-11)
    for integrator in (
        Multistep(formulation="interaction"),
        Multistep(formulation="exponential"),
    ):
        field = final(integrator, 1e-9)
        error = np.linalg.norm(field - reference) / np.linalg.norm(reference)
        # A local tolerance of 1e-9 against a tighter reference: measured at most
        # 4.4e-7 (the interaction formulation on the general guide, whose in-stage
        # collar keeps its steps short and many).
        assert error < 1e-6, (integrator.name, error)


def _march_with(backend_class, route, integrator, rtol=1e-8):
    """Run a route at `rtol`; return its final field, call total and steps."""
    simulation, problem = translated(route, integrator, rtol)
    backend = backend_class(problem, grid=simulation.grid, modes=simulation.modes)
    try:
        backend.set_initial(problem.initial.field0, None)
        steps, field = [], None
        for plane in problem.stepper.save_planes:
            backend.propagate_to(float(plane))
            observed = backend.observe()
            steps += list(observed.step_history.dz)
            field = observed.field
        march = backend._adams
        assert march is not None and field is not None
        return field, march.ledger.total, np.array(steps)
    finally:
        backend.close()


COMPILED_ADAMS: dict[str, type] = {"numba_cpu": NumbaCpuBackend}
if "native" in available():
    from solprop_native import NativeBackend  # pyright: ignore[reportMissingImports]

    COMPILED_ADAMS["native"] = NativeBackend


@pytest.mark.parametrize("formulation", ["interaction", "exponential"])
@pytest.mark.parametrize("route", CONSTANT_ROUTES)
@pytest.mark.parametrize("backend_name", sorted(COMPILED_ADAMS))
def test_a_compiled_backend_follows_the_reference(backend_name, route, formulation):
    integrator = Multistep(formulation=formulation)  # pyright: ignore[reportArgumentType]
    reference, calls, steps = _march_with(NumpyRefBackend, route, integrator)
    field, compiled_calls, compiled_steps = _march_with(
        COMPILED_ADAMS[backend_name], route, integrator
    )
    # The same decisions -- every call -- and step sizes within the rounding the
    # controller passes on: measured up to 1e-6 relative on numba_cpu and 1.8e-4 on
    # Kokkos over the 690 steps of the general guide, whose in-stage collar keeps the
    # controller near its stability limit, where a rounding-level change of the norm
    # moves the next step most. Fields within the accumulated rounding of the two
    # evaluation orders: measured 1e-18 to 5e-14 of the peak on the GNLSE,
    # free-space and general routes and up to 1.1e-9 on the gas-filled
    # capillaries, whose ionisation rate amplifies rounding at the peak (Runge-Kutta's
    # own difference there: 1.3e-11; the exponential formulation's 1.2e-10 before its
    # order choice weighed the in-stage collar, which adds a step on the multimode
    # capillary). A device build's own evaluation order measured 3.5e-9 to 3.8e-9 on
    # two capillary rows on an NVIDIA Hopper device, not from multiply-add contraction
    # (a CPU build contracting every multiply-add stays within 9e-10): its bound is
    # 5e-9.
    bound = 3e-9 if device_backend(backend_name) == "none" else 5e-9
    if compiled_calls != calls:
        # The two adaptive runs forked: a rounding-level change of the error norm
        # moved one accept or order decision, which SEP 25 section 4.4 allows (seen
        # on an NVIDIA Hopper device). They are then compared as accuracy (class T):
        # each against the reference method at a hundred times tighter tolerance,
        # within four times the reference run's own error there, plus the rounding
        # bound below (a route whose two reference runs agree exactly has no error).
        tight, _, _ = _march_with(NumpyRefBackend, route, integrator, rtol=1e-10)
        envelope = AccuracyEnvelope(
            "numpy_ref at rtol 1e-10", error=departure(reference, tight), factor=4.0
        )
        assert departure(field, tight) <= envelope.bound + bound, (
            compiled_calls,
            calls,
        )
        return
    np.testing.assert_allclose(compiled_steps, steps, rtol=1e-3)
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(field, reference, rtol=0, atol=bound * scale)
