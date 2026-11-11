# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Which loop steps a native Runge-Kutta run (solprop SEP 45 phase 3).

The native march steps an adaptive run on staged rows with ``PIController``'s norm, and
the run's metadata records which loop stepped it; no switch selects solprop's loop any
more (solprop SEP 47 section 6.3 retired the test interface with the transitional
bridge). A controller that overrides one of ``PIController``'s decisions is a
reference-only experiment (solprop SEP 47 section 6.3): the backend refuses it by name,
for every method and on CPU and device builds alike, before it allocates anything.
"""

import dataclasses
import warnings

import numpy as np
import pytest

from backends._session_probe_b2 import run_session
from capability_routes import ROUTES
from solprop.backends import available
from solprop.backends.events import Outcome
from solprop.exceptions import BackendError, HostMarchWarning
from solprop.stepper.controller import PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)

native = pytest.importorskip("solprop_native")
backend_module = pytest.importorskip("solprop_native.backend")


ROUTES_NATIVE = (
    "gnlse",
    "capillary (multimode)",
    "radial",
    "capillary (coherent wall)",
)
FLOWS = ("capillary (coherent wall)",)


def launched(route: str, **settings):
    """Return a route's backend on the Runge-Kutta march, its problem and request."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = ROUTES[route](None)
        simulation.settings = dataclasses.replace(
            simulation.settings, integrator=RungeKutta(), **settings
        )
        problem = simulation.translate()
        run_request = simulation._run_request()
    backend = native.NativeBackend(
        problem,
        run_request,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    return backend, problem, run_request


@pytest.mark.parametrize(
    ("route", "dense"),
    [
        *((route, False) for route in ROUTES_NATIVE),
        # Translation refuses interpolated planes beside a coupled flow.
        *((route, True) for route in ROUTES_NATIVE if route not in FLOWS),
    ],
)
def test_the_default_loop_is_native_without_a_warning(route, dense):
    """The backend's session steps the whole run natively, warning of no host march."""
    backend, problem, run_request = launched(route, interpolate_planes=dense)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.simplefilter("error", HostMarchWarning)
            landed = run_session(backend, problem, run_request)
        assert landed.run.terminal.outcome is Outcome.COMPLETED
        assert landed.n_accepted > 0
        assert backend.metadata()["rk_march"] == "native"
    finally:
        backend.close()


DECISIONS = (
    "scale",
    "accepts",
    "propose",
    "rejection_factor",
    "start_factor",
    "equivalent_norm",
)
"""Every ``PIController`` method the native controller implements itself."""


def _overriding(name: str) -> type[PIController]:
    """Return a ``PIController`` subclass that overrides `name` and nothing else."""

    def inherited(self, *args, **kwargs):
        return getattr(super(type(self), self), name)(*args, **kwargs)

    return type(f"Own_{name}", (PIController,), {name: inherited})


@pytest.mark.parametrize("name", DECISIONS)
@pytest.mark.parametrize("method", ["runge-kutta", "adams", "low-storage"])
@pytest.mark.parametrize("device", [False, True], ids=["cpu", "device"])
def test_a_controller_that_overrides_a_decision_is_refused_before_allocation(
    name, method, device, monkeypatch
):
    # solprop SEP 47 section 6.3: a controller that decides in Python is reference-only
    # and refused by name at admission, whatever the method, on CPU and device builds
    # alike (the device patched in: the decision is the controller's alone). The
    # override defers to PIController's own method, so only its presence is refused.
    # The native state, the first allocation, is never built.
    simulation = ROUTES["gnlse"](None)
    settings = {
        "controller": _overriding(name)(rtol=1e-6, atol=0.0),
        "integrator": {
            "runge-kutta": RungeKutta(),
            "adams": Multistep(),
            "low-storage": LowStorageRungeKutta(),
        }[method],
    }
    if method == "low-storage":
        stepper = ROUTES["gnlse"](None)._run_request().stepper
        settings["schedule"] = FixedSchedule.uniform(
            0.0,
            float(stepper.z_end),
            # Below the temporal collar's stable step, which a replay must respect.
            float(stepper.z_end) / 400,
            save_planes=np.asarray(stepper.save_planes).tolist(),
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation.settings = dataclasses.replace(simulation.settings, **settings)
        problem = simulation.translate()
        run_request = simulation._run_request()
    monkeypatch.setattr(backend_module, "_on_device", lambda: device)
    built_states = []
    monkeypatch.setattr(
        backend_module._core, "State", lambda *args: built_states.append(args)
    )
    with pytest.raises(BackendError, match="reference-only") as refused:
        native.NativeBackend(
            problem, run_request, grid=simulation.grid, modes=simulation.modes
        )
    message = str(refused.value)
    assert f"the controller Own_{name}, which overrides PIController's {name}:" in (
        message
    )
    assert "backend='numpy_ref'" in message
    assert built_states == []
