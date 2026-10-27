# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 16 phase 3b: the Adams history resident in Kokkos's engine (ADR-043).

Kokkos returns its own engine from ``StepDriver._adams_engine`` for the interaction
formulation on the modal routes: the history's rows, the trial arithmetic, the norms,
adoption, transport and dense observation stay in the native state. These tests pin
which engine each case takes and check the resident one against the reference
host engine over the same backend's hooks, so that only the engine differs, on the
paths the shared parity suites do not reach: both dense interpolants, a budget stop and
a replayed schedule. The parity of either against the reference backend is the shared
suites' (``test_exponential_routes.py``, ``test_compiled_multistep.py``).
"""

import math
from typing import cast

import numpy as np
import pytest

from backends.conftest import WAVELENGTH
from capability_routes import ROUTES
from legacy_geometry import LEGACY_GRID
from solprop.backends import available
from solprop.backends.multistep import HostHistory
from solprop.exceptions import ResponseBudgetError
from solprop.grid import EnvelopeGrid
from solprop.stepper import FixedSchedule, PIController
from solprop.stepper.multistep import ComparisonMultistep, Multistep

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif("native" not in available(), reason="native is not installed"),
]

NativeBackend = pytest.importorskip("solprop_native").NativeBackend

# The soliton fixture's period (tests/backends/conftest.py).
SOLITON_PERIOD = math.pi / 2 * (100e-15) ** 2 / 11.83e-27
PLANES = list(np.linspace(SOLITON_PERIOD / 16, SOLITON_PERIOD / 2, 8))


def engine_of(route: str, integrator: Multistep) -> str:
    """Return the name of the engine Kokkos's march takes on a route."""
    simulation = ROUTES[route](None)
    simulation.settings = type(simulation.settings)(
        **{
            **{
                name: getattr(simulation.settings, name)
                for name in simulation.settings.__dataclass_fields__
            },
            "integrator": integrator,
        }
    )
    problem = simulation.translate()
    backend = NativeBackend(problem, grid=simulation.grid, modes=simulation.modes)
    try:
        assert backend._adams is not None
        return type(backend._adams._engine).__name__
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("route", "resident"),
    [
        ("gnlse", True),
        ("capillary (mode-averaged)", True),
        ("capillary (multimode)", True),
        ("general guide (constant)", True),
        ("radial", True),
        ("cartesian", True),
        ("line", True),
        ("planar", True),
    ],
)
def test_both_formulations_are_resident_on_every_constant_route(route, resident):
    # Every route here has a constant or separable operator.
    expected = "_ResidentAdams" if resident else HostHistory.__name__
    assert engine_of(route, Multistep(formulation="interaction")) == expected
    assert engine_of(route, Multistep(formulation="exponential")) == expected


def test_a_varying_operator_is_resident_in_both_formulations():
    route = "capillary (pressure gradient)"
    assert engine_of(route, Multistep(formulation="interaction")) == "_ResidentAdams"
    assert engine_of(route, Multistep(formulation="exponential")) == "_ResidentAdams"


def run(problem, grid, monkeypatch, *, resident):
    """Propagate on ``native`` through every plane, with or without the resident engine.

    Without it the same backend takes the host engine over its hooks, so the two runs
    differ only in the engine: the native source, adoption and statistics are shared.
    """
    with monkeypatch.context() as patch:
        if not resident:
            patch.setattr(NativeBackend, "_adams_engine", lambda *_: None)
        backend = NativeBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    fields, steps = [], []
    try:
        for plane in problem.stepper.save_planes:
            backend.propagate_to(float(plane))
            observed = backend.observe()
            fields.append(observed.field.copy())
            steps += list(observed.step_history.dz)
    finally:
        backend.close()
    march = backend._adams
    assert march is not None
    assert (type(march._engine).__name__ == "_ResidentAdams") == resident
    return np.array(fields), march, np.array(steps)


def assert_same_run(first, second):
    """Assert two runs took the same calls and steps and agree to rounding."""
    (fields, march, steps), (other, other_march, other_steps) = first, second
    assert march.ledger.total == other_march.ledger.total
    assert march.ledger.dense_observations == other_march.ledger.dense_observations
    # The engines differ only in rounding order: the resident one forms the predictor
    # in the anchor's frame without the propagator's round trip, and reduces the norms
    # in its own order. A defect is a small difference of large terms, so the steps
    # chosen from it move most: measured at most 2.1e-8 relative over 566 calls, the
    # fields at most 9e-15 of the peak.
    np.testing.assert_allclose(steps, other_steps, rtol=1e-6)
    scale = np.max(np.abs(other))
    np.testing.assert_allclose(fields, other, rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize(
    "integrator",
    [
        Multistep(formulation="interaction"),
        ComparisonMultistep(
            corrector="pec", order=4, corrector_order=5, interpolant="hermite"
        ),
        Multistep(formulation="exponential"),
    ],
    ids=["adams", "hermite", "exponential"],
)
def test_resident_dense_observation_follows_the_host_engine(
    soliton_problem, integrator, monkeypatch
):
    # The narrow fixture's window as it was built before SEP 21 (1024 bins): the
    # steps' 1e-6 agreement below was measured on it, and the comparison PEC method's
    # step choices part by 4.9e-5 on the 768-bin grid the fixture now builds.
    narrow_grid = EnvelopeGrid.from_limits(
        1.0, WAVELENGTH, (500e-9, 2e-6), 1e-12, **LEGACY_GRID
    )
    problem = soliton_problem(
        narrow_grid,
        2,
        save_planes=PLANES,
        controller=PIController(rtol=1e-7, atol=1e-18),
        dz_max=SOLITON_PERIOD,
        interpolate_planes=True,
        integrator=integrator,
    )
    resident = run(problem, narrow_grid, monkeypatch, resident=True)
    assert resident[1].ledger.dense_observations > 0
    assert_same_run(resident, run(problem, narrow_grid, monkeypatch, resident=False))


@pytest.mark.parametrize("formulation", ["interaction", "exponential"])
@pytest.mark.parametrize("oversized_first_step", [False, True])
def test_a_budget_stop_reports_the_resident_state(
    narrow_grid, soliton_problem, monkeypatch, formulation, oversized_first_step
):
    problem = soliton_problem(
        narrow_grid,
        2,
        save_planes=[SOLITON_PERIOD / 2],
        controller=PIController(rtol=1e-7, atol=1e-18),
        integrator=Multistep(max_calls=12, formulation=formulation),
        **({"dz_init": SOLITON_PERIOD / 4} if oversized_first_step else {}),
    )
    stopped = {}
    for resident in (True, False):
        with monkeypatch.context() as patch:
            if not resident:
                patch.setattr(NativeBackend, "_adams_engine", lambda *_: None)
            backend = NativeBackend(problem, grid=narrow_grid)
        backend.set_initial(problem.initial.field0, None)
        with pytest.raises(ResponseBudgetError) as caught:
            backend.propagate_to(SOLITON_PERIOD / 2)
        if oversized_first_step:
            assert backend._adams is not None
            assert backend._adams.ledger.rejected_steps > 0
        stopped[resident] = caught.value
        backend.close()
    # The positions sum steps chosen from norms that differ in rounding: measured
    # 2.2e-10 relative, and the stopped fields, a little apart, 1.4e-11 of the peak.
    assert stopped[True].z == pytest.approx(stopped[False].z, rel=1e-8)
    assert stopped[True].calls == stopped[False].calls
    host = np.asarray(stopped[False].field)
    np.testing.assert_allclose(
        np.asarray(stopped[True].field), host, rtol=0, atol=1e-9 * np.max(np.abs(host))
    )


@pytest.mark.parametrize("formulation", ["interaction", "exponential"])
def test_a_replayed_schedule_follows_the_host_engine(
    narrow_grid, soliton_problem, monkeypatch, formulation
):
    length = SOLITON_PERIOD / 4
    planes = [length / 2, length]
    problem = soliton_problem(
        narrow_grid,
        2,
        save_planes=planes,
        schedule=FixedSchedule.uniform(0.0, length, length / 23, save_planes=planes),
        dz_max=length / 23,
        dz_init=length / 23,
        integrator=Multistep(formulation=formulation),
    )
    assert_same_run(
        run(problem, narrow_grid, monkeypatch, resident=True),
        run(problem, narrow_grid, monkeypatch, resident=False),
    )


@pytest.mark.parametrize("formulation", ["interaction", "exponential"])
@pytest.mark.parametrize("route", ["cartesian", "planar"])
def test_a_separable_route_follows_the_host_engine(route, formulation, monkeypatch):
    # The separable routes form each element's exponent from the step's length and
    # the route's own operator, as the engine's stage does.
    integrator = Multistep(formulation=formulation)  # pyright: ignore[reportArgumentType]
    simulation = ROUTES[route](None)
    simulation.settings = type(simulation.settings)(
        **{
            **{
                name: getattr(simulation.settings, name)
                for name in simulation.settings.__dataclass_fields__
            },
            "integrator": integrator,
            "controller": PIController(
                rtol=1e-8,
                atol=cast("PIController", simulation.settings.controller).atol,
            ),
        }
    )
    problem = simulation.translate()

    def march(resident):
        with monkeypatch.context() as patch:
            if not resident:
                patch.setattr(NativeBackend, "_adams_engine", lambda *_: None)
            backend = NativeBackend(
                problem, grid=simulation.grid, modes=simulation.modes
            )
        try:
            backend.set_initial(problem.initial.field0, None)
            for plane in problem.stepper.save_planes:
                backend.propagate_to(float(plane))
            observed = backend.observe()
            march = backend._adams
            assert march is not None
            return observed.field, march.ledger.total, type(march._engine).__name__
        finally:
            backend.close()

    field, calls, engine = march(True)
    host, host_calls, host_engine = march(False)
    assert (engine, host_engine) == ("_ResidentAdams", HostHistory.__name__)
    assert calls == host_calls
    np.testing.assert_allclose(field, host, rtol=0, atol=1e-12 * np.max(np.abs(host)))


@pytest.mark.parametrize("dense", [False, True], ids=["landed", "dense"])
def test_the_compensated_reference_follows_the_host_engine(dense, monkeypatch):
    # A pressure gradient: the exponential formulation's transported compensation
    # (each sample's state and rate, the carried rows, the endpoint term and, densely,
    # the guarded interpolant) runs in the engine.
    from backends.test_exponential_varying import LENGTH, gradient  # noqa: PLC0415

    simulation = gradient(
        save_planes=list(np.linspace(LENGTH / 8, LENGTH, 8)),
        interpolate_planes=dense,
        rtol=1e-7,
        # Dense observation needs the in-stage collar; it holds the steps short.
        absorber_scheme="nonlinear" if dense else "split_step",
    )
    problem = simulation.translate()

    def march(resident):
        with monkeypatch.context() as patch:
            if not resident:
                patch.setattr(NativeBackend, "_adams_engine", lambda *_: None)
            backend = NativeBackend(
                problem, grid=simulation.grid, modes=simulation.modes
            )
        try:
            backend.set_initial(problem.initial.field0, None)
            fields = []
            for plane in problem.stepper.save_planes:
                backend.propagate_to(float(plane))
                fields.append(backend.observe().require_field().copy())
            march = backend._adams
            assert march is not None
            engine = march._engine
            return np.array(fields), march.ledger, type(engine).__name__, engine
        finally:
            backend.close()

    fields, ledger, name, engine = march(True)
    host, host_ledger, host_name, _ = march(False)
    assert (name, host_name) == ("_ResidentAdams", HostHistory.__name__)
    assert engine.compensated
    # Landed, the engines take the same steps. Densely the controller passes rounding
    # on (below), and a build that rounds differently can end a step apart: the
    # Windows Clang build took 207 accepted steps against the host engine's 206, each
    # 6.0e-9 of the peak from an rtol 1e-10 reference and 2.5e-9 apart (SEP 29 phase
    # 0). The field bound below, not the count, carries the agreement.
    assert abs(ledger.total - host_ledger.total) <= (2 if dense else 0)
    assert ledger.dense_observations == host_ledger.dense_observations
    assert (ledger.dense_observations > 0) == dense
    # The same arithmetic in another order: measured 1e-15 of the peak landed, and
    # 1.5e-10 densely, where the in-stage collar holds every step at its stability
    # limit and the step controller passes the rounding on (steps up to 4 % apart over
    # the 206; 4.5e-12 before the exponential formulation's order choice weighed the
    # collar, at 0.4 %). With the collar scored by the error norm (kernel schema 23)
    # the controller reads the collar's bins too: 1.8e-10 on every isolated run, and
    # 3.7e-9 once in a full parallel suite, at five bins of one plane next to the
    # band's edge, about 1e-7 of their own amplitude, with the same calls.
    scale = np.max(np.abs(host))
    np.testing.assert_allclose(fields, host, rtol=0, atol=1e-8 * scale)
