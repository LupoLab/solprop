# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The coupled flow's exponential Adams march, resident wherever a CPU build runs it.

solprop SEP 47 phase 4 (section 6.3, the coupled-flow row): a coupled block wider than
16 channels steps on the resident Adams engine of a CPU build, and the native session
serves it, where solprop's host engine stepped it over the backend's hooks before and
the session left it to the transitional bridge. A device build keeps today's
admission: the host engine and the bridge.

Each case is a lossless helium taper with taper coupling and every bin on the flow's
eigenbasis route, over ten fixed steps short enough that the formation bound of
``tests/flow_parity.py`` stays far below one. The resident engine is held to
``numpy_ref`` and to solprop's host engine on the same backend within the ten-step
bound of ``tests/driver/test_native_coupled_flow.py`` (``parity_bound``), written
before any measurement; the session is the bridge over the resident engine to the bit
on one thread and the bridge over the host engine within the same bound.
"""

import os
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
from sessions.executors import IDENTITY, Started, drive, entrance
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import available
from solprop.driver import Simulation
from solprop.exceptions import GrazingIncidenceWarning, MaterialRangeWarning
from solprop.profiles import PiecewiseLinear
from solprop.stepper import FixedSchedule

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
"""Whether the installed build executes on an accelerator, which keeps the host
engine for a wide block."""

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
    monkeypatch.setattr(backend_module, "_flow_on_host", lambda *_: True)


def _scale(case: Case) -> float:
    return float(np.max(np.abs(case.problem.initial.field0)))


@pytest.mark.parametrize("modes", WIDE)
def test_a_wide_flow_takes_the_resident_engine_and_the_session(cases, modes):
    """Past 16 channels a CPU build's backend and session step the flow natively."""
    case = cases(modes)
    problem, request = case.problem, case.run_request
    held = backend_module._host_flow(problem)
    assert held is not None and bool(np.all(held.route))
    assert problem.shape[0] == modes > backend_module.ADAMS_MAX_CHANNELS
    refused = native_session.refusal(problem, request)
    if DEVICE:
        assert refused is not None and "host engine" in refused
        return
    assert refused is None
    backend = NativeBackend(problem, request, grid=case.grid)
    try:
        assert type(backend._adams._engine).__name__ == "_ResidentAdams"
        assert backend._adams.native
        assert native_session.backend_refusal(backend) is None
        # The host passes' channel rows: one per bin, nine samples of every channel.
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
    if DEVICE:
        pytest.skip("a device build steps a wide block on the host engine")
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


def session_run(case: Case, monkeypatch, path: str) -> tuple[np.ndarray, Any]:
    """Run a case's prepared execution on one path; return its last field and status.

    ``"session"`` is whatever the preparation chooses, ``"bridge"`` the transitional
    session over the backend's own march and ``"host"`` that bridge over solprop's host
    engine. Every observation is exported with the identity recipe, the state as the
    backend holds it.
    """
    with monkeypatch.context() as patched:
        if path != "session":
            patched.setattr(
                "solprop_native.session.session_refusal",
                lambda *_: "the test's bridge",
            )
        if path == "host":
            on_host_engine(patched)
        backend = NativeBackend(
            case.problem,
            case.run_request,
            grid=case.grid,
            **CONTEXT[case.problem.shape[0]],
        )
        prepared = backend.prepare(case.problem, case.run_request)
        session = prepared.start(entrance(case.problem))
        try:
            run = drive(Started(session), export=IDENTITY)
        finally:
            session.close()
        if path == "session":
            assert prepared.refused is None
    return run.fields[-1].reshape(case.problem.shape), run.status


# The host engine runs solprop's march, with the warning
# tests/backends/test_native_march_admission.py asserts (SEP 39).
@pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning")
@pytest.mark.medium
@pytest.mark.parametrize("modes", WIDE)
def test_the_session_runs_a_wide_flow_as_the_bridges_do(cases, modes, monkeypatch):
    """The session is the resident bridge's to the bit and the host engine's in bound.

    The evidence of SEP 47's phase 4: where the bridge stepped natively the session
    is bitwise the same on one thread; where it stepped on the host engine, the
    session is held to ``parity_bound``, its ``rho`` measured on the native runs.
    """
    if DEVICE:
        pytest.skip("a device build steps a wide block on the host engine")
    case = cases(modes)
    session, status = session_run(case, monkeypatch, "session")
    bridge, bridged = session_run(case, monkeypatch, "bridge")
    host, hosted = session_run(case, monkeypatch, "host")
    expected = propagate(case, NativeBackend)
    linear = 1e3 * propagate(case, NativeBackend, case.problem.initial.field0 * 1e-3)
    bound = parity_bound(case, expected, linear)
    assert status.attempts == bridged.attempts == hosted.attempts == STEPS
    if _one_thread():
        np.testing.assert_array_equal(session, bridge)
    assert float(np.max(np.abs(session - bridge))) <= bound
    assert float(np.max(np.abs(session - host))) <= bound
