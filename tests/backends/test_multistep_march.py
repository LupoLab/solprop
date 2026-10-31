# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Adams march (SEP 13) on the reference: order, exact flow, calls, refusals.

These are the tests SEP 13's phases 1--3 need to trust the method comparison; the full
verification table of SEP 13 section 7 belongs to the selected method at delivery.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from legacy_geometry import LEGACY_GRID
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import resolve_name
from solprop.driver.simulation import Simulation, SimulationSettings
from solprop.exceptions import (
    BackendError,
    DriverError,
    ResponseBudgetError,
    SolpropWarning,
)
from solprop.gasflow import steady_fill
from solprop.grid import EnvelopeGrid
from solprop.stepper import TSIT5, FixedSchedule, PIController, RungeKutta
from solprop.stepper.multistep import ComparisonMultistep, Multistep
from solprop.translate.coupled import constant_schur_flow
from solprop.translate.linop import delta_phi

# The soliton fixture's period: pi/2 times the dispersion length of a 100 fs sech in
# its -11.83 fs^2/mm fibre (tests/backends/conftest.py).
SOLITON_PERIOD = math.pi / 2 * (100e-15) ** 2 / 11.83e-27

PEC4 = ComparisonMultistep(corrector="pec", order=4)
PECE4 = ComparisonMultistep(corrector="pece", order=4)
AB4 = ComparisonMultistep(corrector="none", order=4)
RICHTER = ComparisonMultistep(corrector="pec", order=1, corrector_order=2)


@pytest.fixture(scope="session")
def narrow_grid():
    """Return the conftest's narrow grid as built before SEP 21.

    The measurements here were fixed on it, and their subject is not the boundaries.
    """
    return EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 1e-12, **LEGACY_GRID)


def run(problem, grid):
    """Propagate a problem to its last plane; return the field and the call ledger."""
    backend = NumpyRefBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    backend.propagate_to(float(problem.stepper.save_planes[-1]))
    field = backend.observe().require_field()[0, 0, 0]
    ledger = None if backend._adams is None else backend._adams.ledger
    backend.close()
    return field, ledger


@pytest.mark.medium
@pytest.mark.parametrize(
    ("method", "order"), [(AB4, 4), (PEC4, 4), (PECE4, 4), (RICHTER, 2)]
)
def test_adams_converges_at_its_order(narrow_grid, soliton_problem, method, order):
    # A fundamental soliton over one period. dz_max sets the mature step; the tolerance
    # is a loose safety net that never binds, and the startup ramp starts a thousandth
    # of a step down so its low-order error stays below the mature steps'.
    def solve(n, integrator, rtol=1e-2):
        problem = soliton_problem(
            narrow_grid,
            1,
            save_planes=[SOLITON_PERIOD],
            controller=PIController(rtol=rtol, atol=1e-18),
            dz_max=SOLITON_PERIOD / n,
            dz_init=SOLITON_PERIOD / n * 1e-3,
            integrator=integrator,
        )
        return run(problem, narrow_grid)[0]

    reference = solve(1, None, rtol=1e-12)
    errors = [
        np.linalg.norm(solve(n, method) - reference) / np.linalg.norm(reference)
        for n in (20, 80)
    ]
    observed = math.log(errors[0] / errors[1]) / math.log(4)
    # Three levels of the measurement: 3.7-5.4 for the order-four methods, 2.0 for
    # Richter's; the margin admits the irregular startup and landing steps.
    assert observed > order - 0.6


def test_adams_is_the_exact_linear_flow_without_a_source(narrow_grid, linear_problem):
    length = 0.5
    problem = linear_problem(narrow_grid, save_planes=[length], integrator=PECE4)
    field, ledger = run(problem, narrow_grid)
    start = problem.initial.field0[0, 0, 0] * problem.grids.active_mask
    exact = np.exp(delta_phi(problem.linear, 0.0, length))[0, 0, 0] * start
    # A zero source leaves only the propagators' products: round-off per step.
    np.testing.assert_allclose(field, exact, rtol=0, atol=1e-12 * np.abs(start).max())
    assert ledger is not None and ledger.rejected_steps == 0


@pytest.mark.parametrize(
    ("method", "per_step"),
    [
        (Multistep(formulation="interaction"), 1),
        (Multistep(order=1, formulation="interaction"), 1),
        (Multistep(interval_cap=True, formulation="interaction"), 1),
        (AB4, 1),
        (PEC4, 1),
        (PECE4, 2),
    ],
)
def test_adams_call_ledger(narrow_grid, soliton_problem, method, per_step):
    problem = soliton_problem(
        narrow_grid,
        1,
        save_planes=[SOLITON_PERIOD / 4, SOLITON_PERIOD / 2],
        controller=PIController(rtol=1e-4, atol=1e-18),
        integrator=method,
    )
    _, ledger = run(problem, narrow_grid)
    assert ledger is not None
    # One startup call at the entrance; the save plane keeps the history.
    assert ledger.startup == 1
    assert ledger.restarts == 0
    accepted_cost = ledger.total - ledger.startup - ledger.rejected
    assert accepted_cost == per_step * ledger.accepted_steps
    if not method.variable_order:
        assert max(ledger.orders) == method.order
    assert max(ledger.orders) <= method.order


def test_settings_refuse_an_integrator_elsewhere():
    planes = [1.0]
    common = {"save_planes": planes, "dz_max": 0.1, "dz_init": 1e-3}
    control = PIController(rtol=1e-6, atol=1e-12)
    with pytest.raises(DriverError, match="RungeKutta"):
        SimulationSettings(control, backend="cuda", integrator=PEC4, **common)
    with pytest.raises(DriverError, match="RungeKutta"):
        SimulationSettings(control, backend="cuda", **common)
    settings = SimulationSettings(control, integrator=PEC4, **common)
    # Resolved as for the Runge-Kutta march (SEP 16 phase 3).
    assert settings.backend == resolve_name("auto")
    # Any backend steps the Runge-Kutta march.
    assert SimulationSettings(
        control, backend="cuda", integrator=RungeKutta(TSIT5), **common
    ).integrator == RungeKutta(TSIT5)


def test_the_default_integrator_is_exponential_adams():
    settings = SimulationSettings(
        PIController(rtol=1e-6, atol=0.0), save_planes=[1.0], dz_max=0.1, dz_init=1e-3
    )
    assert settings.integrator == Multistep(formulation="exponential")


@pytest.mark.parametrize("rtol", [5e-2, 0.1])
def test_a_loose_adams_tolerance_is_warned_on(rtol):
    common = {"save_planes": [1.0], "dz_max": 0.1, "dz_init": 1e-3}
    with pytest.warns(SolpropWarning, match="loose for the Adams method"):
        SimulationSettings(PIController(rtol=rtol, atol=0.0), **common)
    # Not for the Runge-Kutta march, nor a replayed schedule, which ignores rtol.
    SimulationSettings(
        PIController(rtol=rtol, atol=0.0), integrator=RungeKutta(), **common
    )
    SimulationSettings(
        PIController(rtol=rtol, atol=0.0),
        schedule=FixedSchedule.uniform(0.0, 1.0, 0.1),
        **common,
    )


def test_contract_admits_an_integrator_with_a_schedule(narrow_grid, soliton_problem):
    # SEP 16 phase 2: a schedule fixes an Adams run's accepted positions.
    schedule = FixedSchedule.uniform(0.0, 0.1, 0.01, save_planes=[0.1])
    problem = soliton_problem(
        narrow_grid, 1, save_planes=[0.1], schedule=schedule, integrator=PEC4
    )
    assert problem.stepper.schedule is schedule


@pytest.mark.parametrize("dense", ["adams", "hermite"])
def test_dense_observation_is_the_exact_linear_flow_without_a_source(
    narrow_grid, linear_problem, dense
):
    planes = [0.1, 0.2, 0.3, 0.5]
    method = ComparisonMultistep(
        corrector="pec", order=3, corrector_order=4, interpolant=dense
    )
    problem = linear_problem(
        narrow_grid, save_planes=planes, integrator=method, interpolate_planes=True
    )
    backend = NumpyRefBackend(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    start = problem.initial.field0[0, 0, 0] * problem.grids.active_mask
    for plane in planes:
        backend.propagate_to(plane)
        field = backend.observe().require_field()[0, 0, 0]
        exact = np.exp(delta_phi(problem.linear, 0.0, plane))[0, 0, 0] * start
        # With no source the interaction variable is constant across the step, so
        # every interpolant reproduces it: round-off in the propagators only.
        np.testing.assert_allclose(
            field, exact, rtol=0, atol=1e-12 * np.abs(start).max()
        )
    assert backend._adams is not None
    # One step crosses every plane short of the end, which is landed on.
    assert backend._adams.ledger.dense_observations == len(planes) - 1
    backend.close()


@pytest.mark.medium
@pytest.mark.parametrize("dense", ["adams", "hermite"])
def test_dense_observation_matches_landing(narrow_grid, soliton_problem, dense):
    planes = list(np.linspace(0.05, 1.0, 20) * SOLITON_PERIOD)

    def fields(method, *, interpolate=False):
        problem = soliton_problem(
            narrow_grid,
            2,
            save_planes=planes,
            controller=PIController(rtol=1e-7, atol=1e-18),
            integrator=method,
            interpolate_planes=interpolate,
        )
        backend = NumpyRefBackend(problem, grid=narrow_grid)
        backend.set_initial(problem.initial.field0, None)
        observed = []
        for plane in planes:
            backend.propagate_to(plane)
            observed.append(backend.observe().require_field()[0, 0, 0])
        ledger = backend._adams.ledger if backend._adams is not None else None
        backend.close()
        return np.array(observed), ledger

    landed, _ = fields(PECE4)
    interpolated, ledger = fields(
        ComparisonMultistep(corrector="pece", order=4, interpolant=dense),
        interpolate=True,
    )
    assert ledger is not None and ledger.dense_observations > 0
    scale = np.linalg.norm(landed, axis=1)
    error = np.linalg.norm(interpolated - landed, axis=1) / scale
    # The two runs take different steps, so they differ by their own global errors at
    # rtol 1e-7: 2.0e-7 measured, identical for both interpolants, which therefore add
    # less than that. A factor of ten above the measurement.
    assert error.max() < 2e-6


@pytest.mark.medium
def test_variable_order_converges_and_uses_its_orders(narrow_grid, soliton_problem):
    def solve(integrator, rtol):
        problem = soliton_problem(
            narrow_grid,
            2,
            save_planes=[SOLITON_PERIOD / 2],
            controller=PIController(rtol=rtol, atol=1e-18),
            integrator=integrator,
        )
        return run(problem, narrow_grid)

    reference, _ = solve(None, 1e-11)
    method = Multistep(formulation="interaction")
    field, ledger = solve(method, 1e-7)
    assert ledger is not None
    # The order starts at one and is chosen within [1, 5]; a smooth soliton half-period
    # at a tight tolerance climbs above three.
    assert min(ledger.orders) == 1
    assert 3 < max(ledger.orders) <= 5
    error = np.linalg.norm(field - reference) / np.linalg.norm(reference)
    # 6.9e-8 measured over 477 steps at a local tolerance of 1e-7; the bound is a
    # factor of ten above the measurement.
    assert error < 7e-7


def test_call_budget_stops_before_overspending(narrow_grid, soliton_problem):
    budget = 25
    method = ComparisonMultistep(corrector="pece", order=3, max_calls=budget)
    problem = soliton_problem(
        narrow_grid,
        1,
        save_planes=[SOLITON_PERIOD],
        controller=PIController(rtol=1e-6, atol=1e-18),
        integrator=method,
    )
    backend = NumpyRefBackend(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    with pytest.raises(ResponseBudgetError, match="budget of 25") as caught:
        backend.propagate_to(SOLITON_PERIOD)
    assert backend._adams is not None
    ledger = backend._adams.ledger
    assert ledger.total == caught.value.calls <= budget
    # PECE needs two calls a trial: the stop leaves at most one unspent.
    assert budget - ledger.total < 2
    assert 0.0 < caught.value.z < SOLITON_PERIOD
    np.testing.assert_array_equal(caught.value.field, backend._state)
    backend.close()


def test_resume_restarts_the_history_and_charges_its_startup(
    narrow_grid, soliton_problem
):
    method = ComparisonMultistep(corrector="pec", order=3, corrector_order=4)
    problem = soliton_problem(
        narrow_grid,
        1,
        save_planes=[SOLITON_PERIOD / 2, SOLITON_PERIOD],
        controller=PIController(rtol=1e-7, atol=1e-18),
        integrator=method,
    )
    whole = NumpyRefBackend(problem, grid=narrow_grid)
    whole.set_initial(problem.initial.field0, None)
    whole.propagate_to(SOLITON_PERIOD / 2)
    halfway = whole.observe().require_field()
    whole.propagate_to(SOLITON_PERIOD)
    uninterrupted = whole.observe().require_field()
    whole.close()

    resumed = NumpyRefBackend(problem, grid=narrow_grid)
    resumed.set_initial(halfway, None, z=SOLITON_PERIOD / 2)
    resumed.propagate_to(SOLITON_PERIOD)
    restarted = resumed.observe().require_field()
    assert resumed._adams is not None
    # A fresh start: one startup call and the order ramp again from one.
    assert resumed._adams.ledger.startup == 1
    assert resumed._adams.ledger.orders[0] == 1
    resumed.close()
    # Two trajectories each converged to rtol 1e-7: 1.2e-8 apart measured, bounded at
    # ten times that.
    error = np.linalg.norm(restarted - uninterrupted) / np.linalg.norm(uninterrupted)
    assert error < 1.2e-7


def test_capillary_constructor_takes_the_collar_scheme_and_integrator():
    method = ComparisonMultistep(corrector="pec", order=3, corrector_order=4)
    simulation = Simulation.from_capillary(
        150e-6,
        0.1,
        "He",
        1.0,
        lambda0=800e-9,
        fwhm=10e-15,
        energy=1e-5,
        trange=100e-15,
        wavelength_limits=(200e-9, 2e-6),
        absorber_scheme="split_step",
        integrator=method,
        noise=None,
    )
    assert simulation.settings.absorber_scheme == "split_step"
    assert simulation.settings.integrator is method
    assert simulation.settings.backend == resolve_name("auto")


def test_the_record_names_the_method_and_keys_on_it():
    # The run record holds the Adams method in full and hashes it, so a resume can
    # never continue a run with a different stepping method.
    def metadata(integrator):
        simulation = Simulation.from_capillary(
            150e-6,
            0.1,
            "He",
            1.0,
            lambda0=800e-9,
            fwhm=10e-15,
            energy=1e-5,
            trange=100e-15,
            wavelength_limits=(200e-9, 2e-6),
            integrator=integrator,
            backend="numpy_ref",
            noise=None,
        )
        return simulation.metadata()

    runge_kutta = metadata(RungeKutta())
    adams = metadata(None)
    assert runge_kutta.stepper.method is None
    # The default integrator, recorded with its formulation.
    assert adams.stepper.method == repr(Multistep(formulation="exponential"))
    assert runge_kutta.configuration_hash() != adams.configuration_hash()


def test_an_interior_kink_restarts_the_history():
    # A steady fill's central port is a hard kink: the march lands on it and starts its
    # history afresh on the far side (SEP 13 section 6.2), at one extra startup call.
    fill = steady_fill("He", [0.0, 0.1, 0.2], [1.0, 2.0, 1.0], radius=125e-6)
    simulation = Simulation.from_capillary(
        125e-6,
        0.2,
        "He",
        fill,
        lambda0=800e-9,
        fwhm=10e-15,
        energy=1e-6,
        trange=100e-15,
        wavelength_limits=(200e-9, 2e-6),
        integrator=Multistep(formulation="interaction"),
        n_save=2,
        rtol=1e-4,
        noise=None,
    )
    problem = simulation.translate()
    assert 0.1 in problem.linear.hard_kinks
    backend = NumpyRefBackend(problem, grid=simulation.grid, modes=simulation.modes)
    backend.set_initial(problem.initial.field0, None)
    backend.propagate_to(0.2)
    history = backend.observe().step_history
    assert backend._adams is not None
    ledger = backend._adams.ledger
    backend.close()
    # The kink is an accepted step's end (the history records each attempt's end), and
    # the history restarts there at order one.
    ends = history.z[history.accepted]
    assert np.any(np.isclose(ends, 0.1, rtol=0, atol=1e-12))
    assert ledger.restarts >= 1
    assert ledger.startup == 1 + ledger.restarts
    assert ledger.orders.count(1) >= 2


@pytest.mark.parametrize("backend_class", [NumpyRefBackend, NumbaCpuBackend])
def test_the_backend_refuses_what_the_march_cannot_step(
    narrow_grid, soliton_problem, backend_class
):
    # The modified noise is admitted since SEP 16 phase 2
    # (tests/backends/test_exponential_march.py); a coupled flow is not, in the
    # interaction formulation, and both backends say so on construction.
    problem = soliton_problem(
        narrow_grid,
        1,
        save_planes=[0.1],
        integrator=Multistep(formulation="interaction"),
    )
    # A zero constant-Schur flow on the envelope's frequencies, sorted as a flow's are:
    # the refusal reads only that a coupled flow is present.
    flow = constant_schur_flow(
        np.sort(problem.grids.omega),
        problem.linear.phi_breakpoints[[0, -1]],
        np.zeros((1, 1, problem.grids.omega.size), dtype=complex),
        dz_max=problem.stepper.dz_max,
        condition_limit=1e6,
        gauge_id="test",
        certificate_id="constant-schur",
    )
    coupled = replace(problem, linear=replace(problem.linear, coupled_flow=flow))
    with pytest.raises(BackendError, match="a coupled linear flow"):
        backend_class(coupled, grid=narrow_grid)


def test_repeated_rejections_restart_the_history(narrow_grid, soliton_problem):
    # A parasitic mode of the recurrence survives every smaller step, so the third
    # rejection in a row restarts at order one (SEP 13 phase 2). Driven directly: no
    # in-tree problem rejects three times running at a mature order.
    problem = soliton_problem(
        narrow_grid,
        1,
        save_planes=[SOLITON_PERIOD / 4],
        controller=PIController(rtol=1e-4, atol=1e-18),
        integrator=Multistep(formulation="interaction"),
    )
    backend = NumpyRefBackend(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    backend.propagate_to(SOLITON_PERIOD / 4)
    march = backend._adams
    assert march is not None
    assert len(march._history) > 1
    for _ in range(3):
        march._reject(1e-3, 2.0, 2)
    assert march.ledger.failure_restarts == 1
    assert march._history == []
    assert march._rejections == 0
    backend.close()


def test_variable_order_pece_one_order_higher_takes_the_general_path(
    narrow_grid, soliton_problem
):
    # The selected family (one-order-higher PEC) takes the fused path; PECE one order
    # higher with variable order takes the general one, scoring the neighbouring
    # orders' defects (SEP 13 section 4.4) and still converging on the soliton.
    planes = [SOLITON_PERIOD / 4]
    method = ComparisonMultistep(
        corrector="pece", order=4, corrector_order=5, variable_order=True
    )

    def final(integrator, rtol):
        problem = soliton_problem(
            narrow_grid,
            2,
            save_planes=planes,
            controller=PIController(rtol=rtol, atol=1e-18),
            integrator=integrator,
        )
        backend = NumpyRefBackend(problem, grid=narrow_grid)
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(planes[-1])
        field = np.array(backend.observe().require_field())
        march = backend._adams
        backend.close()
        return field, march

    reference, _ = final(None, 1e-11)
    field, march = final(method, 1e-6)
    assert march is not None
    assert march._fused is False
    assert len(set(march.ledger.orders)) > 2
    scale = np.max(np.abs(reference))
    # Measured 1.1e-7 of the peak over orders one to four.
    assert np.max(np.abs(field - reference)) < 1e-6 * scale
