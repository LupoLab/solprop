# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 16 phase 2: the exponential Adams formulation on the reference backend.

The formulation shares the Adams march's lifecycle (``solprop.backends.multistep``):
these tests check what is its own -- the record, exactness on the linear flow, order,
agreement with the Runge-Kutta march and the transported formulation, dense
observation, the call ledger and budget -- and the two capabilities SEP 16 phase 2
adds to both formulations: the ``modified`` noise and prescribed schedules.
"""

import dataclasses
import math
import warnings
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from bounds import assert_within, gamma, step_evaluations
from legacy_geometry import LEGACY_GRID
from solprop._helpers import PLAIN
from solprop.backends.batch import LockstepRunner
from solprop.backends.driver import StepDriver
from solprop.backends.exponential import refuse_formulation
from solprop.backends.multistep import (
    AdamsMarch,
    compensation_tables,
    exponential_tables,
    fused_tables,
    trial_tables,
)
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.exceptions import (
    BackendError,
    ConvergenceError,
    ResponseBudgetError,
    SolpropWarning,
    StepperError,
    TranslationError,
)
from solprop.grid import EnvelopeGrid
from solprop.noise import modified_noise
from solprop.stepper import FixedSchedule, PIController, RungeKutta
from solprop.stepper.multistep import (
    PEC_REAL_INTERVAL,
    ComparisonMultistep,
    Multistep,
    march_policy,
)
from solprop.translate.coupled import constant_schur_flow
from solprop.translate.linop import delta_phi

# The soliton fixture's period (tests/backends/conftest.py).
SOLITON_PERIOD = math.pi / 2 * (100e-15) ** 2 / 11.83e-27

EXPONENTIAL = Multistep(formulation="exponential")
INTERACTION = Multistep(formulation="interaction")
FORMULATIONS = [INTERACTION, EXPONENTIAL]


@pytest.fixture(scope="session")
def narrow_grid():
    """Return the conftest's narrow grid as built before SEP 21.

    The measurements here were fixed on it, and their subject is not the boundaries.
    """
    return EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 1e-12, **LEGACY_GRID)


def fixed(method: Multistep) -> Multistep:
    """Return `method` held at its top order, for a matched comparison of two runs.

    On one schedule a variable-order march still chooses each step's order from its
    defects, which rounding can tip; at a fixed order two runs share every decision
    and differ by rounding alone (SEP 25 section 4.4).
    """
    return dataclasses.replace(method, variable_order=False)


def replay_bound(problem, steps: int) -> float:
    """Bound two matched replays of `steps` steps, relative to the field.

    Each step carries its evaluation's rounding through the predictor's and
    corrector's weights (:func:`bounds.step_evaluations`), 16 eps an evaluation
    (the one-step rung's budget), added over the steps.
    """
    return steps * step_evaluations(problem) * 16 * np.finfo(float).eps


def run(problem, grid, noise=None):
    """Propagate to the last plane; return the final field, the march and its steps."""
    backend = NumpyRefBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, noise)
    steps, field = [], None
    for plane in problem.stepper.save_planes:
        backend.propagate_to(float(plane))
        observed = backend.observe()
        steps += list(observed.step_history.dz)
        field = observed.require_field()[0, 0, 0]
    march = backend._adams
    backend.close()
    assert field is not None
    return field, march, np.array(steps)


def ledger_of(march):
    """Return a march's call ledger; the problems here all carry an integrator."""
    assert march is not None
    return march.ledger


def test_the_record_names_the_formulation():
    # repr(Multistep()) is hashed into every Adams run's record, and it names the
    # formulation, the default exponential one included.
    assert repr(Multistep()) == (
        "Multistep(order=5, variable_order=True, max_calls=None, interval_cap=False, "
        "equal_approach=True, max_growth=2.0, min_factor=0.2, safety=0.8, "
        "formulation='exponential', carrier_frame='off')"
    )
    assert repr(ComparisonMultistep(corrector="pece", order=4)) == (
        "ComparisonMultistep(order=4, variable_order=False, max_calls=None, "
        "interval_cap=False, equal_approach=True, max_growth=2.0, min_factor=0.2, "
        "safety=0.8, formulation='interaction', carrier_frame='off', corrector='pece', "
        "corrector_order=None, interpolant='adams')"
    )
    assert repr(INTERACTION).endswith(
        "safety=0.8, formulation='interaction', carrier_frame='off')"
    )
    assert EXPONENTIAL.name == "exponential-adams-pec5/6-variable"
    assert Multistep() == EXPONENTIAL
    assert INTERACTION != EXPONENTIAL


def test_comparison_variants_keep_the_interaction_formulation():
    with pytest.raises(StepperError, match="interaction formulation only"):
        ComparisonMultistep(corrector="pece", order=4, formulation="exponential")


@pytest.mark.parametrize("responses", ["none", "zero kerr"])
def test_exponential_is_the_exact_linear_flow_without_a_source(
    narrow_grid, linear_problem, zero_kerr, responses
):
    length = 0.5
    problem = linear_problem(
        narrow_grid,
        save_planes=[length / 3, length],
        responses=() if responses == "none" else (zero_kerr,),
        integrator=EXPONENTIAL,
    )
    field, march, _ = run(problem, narrow_grid)
    start = problem.initial.field0[0, 0, 0] * problem.grids.active_mask
    exact = np.exp(delta_phi(problem.linear, 0.0, length))[0, 0, 0] * start
    # A zero source leaves the flows' products: round-off per step.
    np.testing.assert_allclose(field, exact, rtol=0, atol=1e-12 * np.abs(start).max())
    assert ledger_of(march).rejected_steps == 0


@pytest.mark.medium
@pytest.mark.parametrize("order", [2, 4])
def test_exponential_converges_at_its_order_on_a_schedule(
    narrow_grid, soliton_problem, order
):
    # A second-order soliton over half a period on uniform prescribed steps: three
    # levels against a tight Runge-Kutta reference.
    length = SOLITON_PERIOD / 2
    method = Multistep(order=order, variable_order=False, formulation="exponential")

    def solve(steps, integrator):
        # The history starts at order one: a startup ramp of steps halving back from
        # the uniform one keeps its low-order local errors below the mature steps'.
        h = length / steps
        ramp = h * 2.0 ** -np.arange(1, 13)[::-1]
        rest = length - ramp.sum()
        count = math.ceil(rest / h)
        schedule = FixedSchedule(np.concatenate((ramp, np.full(count, rest / count))))
        problem = soliton_problem(
            narrow_grid,
            2,
            save_planes=[length],
            schedule=schedule,
            dz_max=length / steps,
            dz_init=length / steps,
            integrator=integrator,
        )
        return run(problem, narrow_grid)[0]

    reference = run(
        soliton_problem(
            narrow_grid,
            2,
            save_planes=[length],
            controller=PIController(rtol=1e-12, atol=1e-18),
        ),
        narrow_grid,
    )[0]
    # The populated bins, above 1e-8 of the peak power. The weak wings below hold
    # free radiation under the soliton's cross-phase modulation, a source linear in
    # each bin's own field that rotates with its free phase: there the exponential
    # formulation converges at order 0.6 only, once |Im D| h exceeds about one (SEP 16
    # section 4.4; measured in the phase 2 report), while carrying 1e-2 of their own
    # norm, which is 1e-9 of the field's.
    power = np.abs(reference) ** 2
    strong = power > 1e-8 * power.max()
    errors = [
        np.linalg.norm((solve(n, method) - reference)[strong])
        / np.linalg.norm(reference[strong])
        for n in (100, 200, 400)
    ]
    observed = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
    # PEC q/(q+1) is order q + 1 once the ramp has raised the order: measured 3.12,
    # 3.07 (q = 2) and 5.00, 5.01 (q = 4); the margin admits a pre-asymptotic level.
    assert all(order + 1 - 0.3 <= one <= order + 2 for one in observed), observed


@pytest.mark.medium
def test_the_default_leaves_its_excess_error_in_the_dark_wings(
    narrow_grid, soliton_problem
):
    """Bound where the adaptive default puts the error it does not control (SEP 16 Q18).

    A fundamental soliton over a period at ``rtol=1e-8``. Its wings below 1e-10 of the
    peak power carry free radiation under the soliton's cross-phase modulation, a
    source term that rotates with each bin's free phase and that the exponential
    formulation interpolates in the lab frame: where ``|Im D| h`` exceeds about one it
    converges there at order 0.6 only (the test above). Scoring the whole active band
    keeps that error bounded by the wings' own content, while the populated bins follow
    the tolerance.
    """
    length = SOLITON_PERIOD
    reference = run(
        soliton_problem(
            narrow_grid,
            1,
            save_planes=[length],
            controller=PIController(rtol=1e-12, atol=1e-18),
        ),
        narrow_grid,
    )[0]
    field, march, _ = run(
        soliton_problem(
            narrow_grid,
            1,
            save_planes=[length],
            controller=PIController(rtol=1e-8, atol=1e-18),
            integrator=Multistep(),
        ),
        narrow_grid,
    )
    power = np.abs(reference) ** 2
    lit = power >= 1e-10 * power.max()
    error = field - reference
    norm = np.linalg.norm(reference)
    # Measured: 2.7e-10 of the field in the populated bins, in 336 calls; 3.3e-7 in the
    # wings, 0.03 of their content (1.2e-5 of the field). The interaction formulation
    # leaves 1.6e-8 there in 820 calls.
    assert np.linalg.norm(error[lit]) < 3e-9 * norm
    assert np.linalg.norm(error[~lit]) < 0.1 * np.linalg.norm(reference[~lit])
    assert ledger_of(march).total < 500


@pytest.mark.medium
@pytest.mark.parametrize("method", FORMULATIONS, ids=["interaction", "exponential"])
def test_both_formulations_agree_with_runge_kutta(narrow_grid, soliton_problem, method):
    planes = [SOLITON_PERIOD / 4, SOLITON_PERIOD / 2]
    reference = run(
        soliton_problem(
            narrow_grid,
            3,
            save_planes=planes,
            controller=PIController(rtol=1e-11, atol=1e-18),
        ),
        narrow_grid,
    )[0]
    field, march, _ = run(
        soliton_problem(
            narrow_grid,
            3,
            save_planes=planes,
            controller=PIController(rtol=1e-8, atol=1e-18),
            integrator=method,
        ),
        narrow_grid,
    )
    error = np.linalg.norm(field - reference) / np.linalg.norm(reference)
    # A local tolerance of 1e-8 over a few hundred steps of an N = 3 soliton: the
    # global error is a few hundred times the tolerance (measured 2e-7 to 2e-6).
    assert error < 1e-5
    assert ledger_of(march).startup == 1


@pytest.mark.parametrize("method", FORMULATIONS, ids=["interaction", "exponential"])
def test_one_call_per_step_and_the_budget(narrow_grid, soliton_problem, method):
    problem = soliton_problem(
        narrow_grid,
        1,
        save_planes=[SOLITON_PERIOD / 4, SOLITON_PERIOD / 2],
        controller=PIController(rtol=1e-5, atol=1e-18),
        integrator=method,
    )
    _, march, _ = run(problem, narrow_grid)
    ledger = ledger_of(march)
    assert ledger.startup == 1
    assert ledger.total - ledger.startup == ledger.accepted_steps + ledger.rejected
    budget = dataclasses.replace(method, max_calls=ledger.total // 2)
    with pytest.raises(ResponseBudgetError) as caught:
        run(
            soliton_problem(
                narrow_grid,
                1,
                save_planes=[SOLITON_PERIOD / 4, SOLITON_PERIOD / 2],
                controller=PIController(rtol=1e-5, atol=1e-18),
                integrator=budget,
            ),
            narrow_grid,
        )
    assert caught.value.calls <= ledger.total // 2


@pytest.mark.medium
def test_exponential_dense_observation_matches_landing(narrow_grid, soliton_problem):
    planes = list(np.linspace(SOLITON_PERIOD / 8, SOLITON_PERIOD / 2, 4))

    def solve(interpolate):
        problem = soliton_problem(
            narrow_grid,
            2,
            save_planes=planes,
            controller=PIController(rtol=1e-9, atol=1e-18),
            integrator=EXPONENTIAL,
            interpolate_planes=interpolate,
        )
        backend = NumpyRefBackend(problem, grid=narrow_grid)
        backend.set_initial(problem.initial.field0, None)
        fields = []
        for plane in planes:
            backend.propagate_to(float(plane))
            fields.append(backend.observe().require_field()[0, 0, 0])
        ledger = ledger_of(backend._adams)
        backend.close()
        return np.array(fields), ledger

    landed, _ = solve(interpolate=False)
    observed, ledger = solve(interpolate=True)
    assert ledger.dense_observations == len(planes) - 1
    for one, two in zip(observed, landed, strict=True):
        # Both are within the run's accuracy of the truth (rtol 1e-9 per step; the
        # interpolant has the step's own order): measured below 1e-7.
        assert np.linalg.norm(one - two) / np.linalg.norm(two) < 1e-6


def test_exponential_dense_observation_is_the_linear_flow_without_a_source(
    narrow_grid, linear_problem
):
    planes = [0.13, 0.5]
    problem = linear_problem(
        narrow_grid,
        save_planes=planes,
        integrator=EXPONENTIAL,
        interpolate_planes=True,
        dz_max=0.3,
    )
    backend = NumpyRefBackend(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    backend.propagate_to(planes[0])
    observed = backend.observe().require_field()[0, 0, 0]
    backend.close()
    start = problem.initial.field0[0, 0, 0] * problem.grids.active_mask
    exact = np.exp(delta_phi(problem.linear, 0.0, planes[0]))[0, 0, 0] * start
    np.testing.assert_allclose(
        observed, exact, rtol=0, atol=1e-12 * np.abs(start).max()
    )


@pytest.mark.medium
@pytest.mark.parametrize("method", FORMULATIONS, ids=["interaction", "exponential"])
def test_modified_noise_with_the_same_draw_as_runge_kutta(
    narrow_grid, soliton_problem, method
):
    # One draw for the run, conditional on which the source is deterministic in z:
    # both Adams formulations converge to the Runge-Kutta solution with the same draw.
    planes = [SOLITON_PERIOD / 8, SOLITON_PERIOD / 4]
    noise = modified_noise(narrow_grid, np.random.default_rng(7))

    def solve(integrator, rtol):
        problem = soliton_problem(
            narrow_grid,
            2,
            save_planes=planes,
            noise=noise,
            controller=PIController(rtol=rtol, atol=1e-18),
            integrator=integrator,
        )
        return run(problem, narrow_grid, noise=noise)

    reference, _, _ = solve(None, 1e-11)
    field, march, _ = solve(method, 1e-8)
    error = np.linalg.norm(field - reference) / np.linalg.norm(reference)
    # As without noise (test_both_formulations_agree_with_runge_kutta): the draw is
    # part of the deterministic source, so the same accuracy holds.
    assert error < 1e-5
    # A different draw moves the weak bins far more than the tolerance: the noise is
    # really in the source.
    other = modified_noise(narrow_grid, np.random.default_rng(8))
    problem = soliton_problem(
        narrow_grid,
        2,
        save_planes=planes,
        noise=other,
        controller=PIController(rtol=1e-8, atol=1e-18),
        integrator=method,
    )
    moved, _, _ = run(problem, narrow_grid, noise=other)
    assert np.linalg.norm(moved - field) > 10 * np.linalg.norm(field - reference)
    assert ledger_of(march).rejected_steps < ledger_of(march).accepted_steps


@pytest.mark.parametrize("method", FORMULATIONS, ids=["interaction", "exponential"])
def test_a_schedule_is_replayed_exactly_and_reports_its_norms(
    narrow_grid, soliton_problem, method
):
    length = SOLITON_PERIOD / 4
    schedule = FixedSchedule.uniform(
        0.0, length, length / 37, save_planes=[length / 2, length]
    )

    def problem_of(integrator):
        return soliton_problem(
            narrow_grid,
            1,
            save_planes=[length / 2, length],
            schedule=schedule,
            controller=PIController(rtol=1e-6, atol=1e-18),
            dz_max=length / 37,
            dz_init=length / 37,
            integrator=integrator,
        )

    _, march, steps = run(problem_of(method), narrow_grid)
    # The steps are the schedule's, every one accepted, one call each after startup:
    # accounting for a prescribed schedule, exact.
    np.testing.assert_array_equal(steps, schedule.steps)
    assert ledger_of(march).accepted_steps == schedule.n_steps
    assert ledger_of(march).total == schedule.n_steps + 1
    # A replay at a fixed order is repeatable to its rounding.
    matched = problem_of(fixed(method))
    first, _, _ = run(matched, narrow_grid)
    second, _, _ = run(matched, narrow_grid)
    assert_within(second, first, replay_bound(matched, schedule.n_steps))


@pytest.mark.parametrize("method", FORMULATIONS, ids=["interaction", "exponential"])
def test_a_lockstep_batch_steps_each_member_with_its_own_history(
    narrow_grid, soliton_problem, method
):
    length = SOLITON_PERIOD / 8
    schedule = FixedSchedule.uniform(0.0, length, length / 11, save_planes=[length])

    def problem_of(order):
        return soliton_problem(
            narrow_grid,
            order,
            save_planes=[length],
            schedule=schedule,
            dz_max=length / 11,
            dz_init=length / 11,
            integrator=fixed(method),
        )

    singles = [run(problem_of(order), narrow_grid)[0] for order in (1, 2)]
    members = []
    for order in (1, 2):
        member = NumpyRefBackend(problem_of(order), grid=narrow_grid)
        member.set_initial(problem_of(order).initial.field0, None)
        members.append(member)
    runner = LockstepRunner(members)
    runner.propagate_to(length)
    observed = runner.observe_members()
    runner.close()
    # A member and its single run: one fixed-order method on one schedule.
    bound = replay_bound(problem_of(1), 11)
    for observation, single in zip(observed, singles, strict=True):
        assert_within(observation.require_field()[0, 0, 0], single, bound)


@pytest.mark.parametrize("method", FORMULATIONS, ids=["interaction", "exponential"])
def test_a_rejected_trial_is_charged_and_leaves_the_history(
    narrow_grid, soliton_problem, method
):
    # A first step of a quarter period at a tight tolerance is far too long: the
    # trials it takes are rejected and charged, and the accepted run still converges.
    planes = [SOLITON_PERIOD / 4]
    problem = soliton_problem(
        narrow_grid,
        2,
        save_planes=planes,
        controller=PIController(rtol=1e-9, atol=1e-18),
        dz_init=SOLITON_PERIOD / 4,
        integrator=method,
    )
    field, march, _ = run(problem, narrow_grid)
    ledger = ledger_of(march)
    assert ledger.rejected_steps >= 1
    assert ledger.rejected == ledger.rejected_steps
    assert ledger.total == ledger.startup + ledger.accepted_steps + ledger.rejected
    reference, _, _ = run(
        soliton_problem(
            narrow_grid,
            2,
            save_planes=planes,
            controller=PIController(rtol=1e-11, atol=1e-18),
        ),
        narrow_grid,
    )
    # As in test_both_formulations_agree_with_runge_kutta, a tighter tolerance.
    assert np.linalg.norm(field - reference) / np.linalg.norm(reference) < 1e-6


@pytest.mark.parametrize(
    "method", [RungeKutta(), *FORMULATIONS], ids=["rk", "interaction", "exponential"]
)
def test_a_schedule_past_the_methods_collar_limit_is_refused(method):
    # The in-stage temporal collar is a damping rate integrated explicitly: DOPRI5
    # keeps h alpha / 2 below 3.3, PEC5/6 below 0.0877. A replayed step between the
    # two limits is safe for the Runge-Kutta march and refused for an Adams one,
    # which it would diverge.
    import warnings  # noqa: PLC0415 - local to this check

    from solprop.boundaries import stable_step  # noqa: PLC0415
    from solprop.driver.simulation import Simulation  # noqa: PLC0415
    from solprop.exceptions import SolpropWarning  # noqa: PLC0415

    def problem_with(schedule):
        simulation = Simulation.from_capillary(
            100e-6,
            0.01,
            "He",
            1.0,
            wavelength=800e-9,
            fwhm=10e-15,
            energy=1e-9,
            trange=100e-15,
            wavelength_limits=(300e-9, 2e-6),
            backend="numpy_ref",
            integrator=method,
            noise=None,
            save_planes=[0.01],
            schedule=schedule,
        )
        return simulation.translate()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SolpropWarning)
        rate = problem_with(None).responses.temporal_absorber
    assert rate is not None
    step = 0.1 * stable_step(rate)  # h alpha / 2 = 0.33: past every PEC's interval
    count = math.ceil(0.01 / step)
    schedule = FixedSchedule(np.full(count, 0.01 / count))
    if isinstance(method, RungeKutta):
        with warnings.catch_warnings():
            warnings.simplefilter("error", SolpropWarning)
            problem_with(schedule)
    else:
        with pytest.raises(TranslationError, match=r"below 0\.0877"):
            problem_with(schedule)


def scheduled(soliton_problem, grid, method, schedule, length, **overrides):
    """Return a soliton problem replaying `schedule` to `length` with `method`."""
    return soliton_problem(
        grid,
        1,
        save_planes=[length],
        schedule=schedule,
        dz_max=float(schedule.steps.max()),
        dz_init=float(schedule.steps[0]),
        integrator=method,
        **overrides,
    )


def lockstep_of(problems, grid):
    """Return a lockstep runner over fresh reference members of `problems`."""
    members = []
    for problem in problems:
        member = NumpyRefBackend(problem, grid=grid)
        member.set_initial(problem.initial.field0, None)
        members.append(member)
    return LockstepRunner(members)


@pytest.mark.parametrize("method", FORMULATIONS, ids=["interaction", "exponential"])
@pytest.mark.parametrize("batch", [False, True], ids=["single", "lockstep"])
def test_a_replayed_schedule_stops_to_save_and_refuses_to_run_out(
    narrow_grid, soliton_problem, method, batch
):
    length = SOLITON_PERIOD / 8
    schedule = FixedSchedule.uniform(0.0, length, length / 9)

    def start(**overrides):
        problem = scheduled(
            soliton_problem, narrow_grid, fixed(method), schedule, length, **overrides
        )
        if batch:
            return lockstep_of([problem], narrow_grid)
        backend = NumpyRefBackend(problem, grid=narrow_grid)
        backend.set_initial(problem.initial.field0, None)
        return backend

    def field_of(runner):
        observed = runner.observe_members()[0] if batch else runner.observe()
        return observed.field[0, 0, 0]

    # Every third step's end is saved: each call stops there, and the steps, so the
    # field, are those of the uninterrupted replay.
    whole = start()
    whole.propagate_to(length)
    stopping = start(step_planes=3)
    calls = 0
    while stopping._z < length * (1 - 1e-12):
        stopping.propagate_to(length)
        calls += 1
    assert calls == 3
    matched = scheduled(soliton_problem, narrow_grid, fixed(method), schedule, length)
    assert_within(field_of(stopping), field_of(whole), replay_bound(matched, 9))
    # Past its last edge the schedule cannot take the march.
    with pytest.raises(BackendError, match="cannot reach"):
        whole.propagate_to(1.01 * length)
    whole.close()
    stopping.close()


@pytest.mark.parametrize("method", FORMULATIONS, ids=["interaction", "exponential"])
def test_a_replayed_step_with_a_non_finite_norm_is_refused(
    narrow_grid, soliton_problem, method, monkeypatch
):
    length = SOLITON_PERIOD / 8
    schedule = FixedSchedule.uniform(0.0, length, length / 9)
    problem = scheduled(soliton_problem, narrow_grid, method, schedule, length)
    backend = NumpyRefBackend(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    # A diverged field: a replay cannot reject the step, so it stops and says why.
    monkeypatch.setattr(
        backend._adams, "_attempt", lambda dz, *, fixed, landing: (None, math.nan)
    )
    with pytest.raises(ConvergenceError, match="non-finite error norm"):
        backend.propagate_to(length)
    backend.close()


def test_lockstep_members_must_share_the_integrator(narrow_grid, soliton_problem):
    length = SOLITON_PERIOD / 8
    schedule = FixedSchedule.uniform(0.0, length, length / 9)
    problems = [
        scheduled(soliton_problem, narrow_grid, method, schedule, length)
        for method in FORMULATIONS
    ]
    with pytest.raises(BackendError, match="same integrator"):
        lockstep_of(problems, narrow_grid)


def test_a_backend_without_the_lab_frame_source_refuses_the_formulation():
    # Every shipped backend implements the hook; an external one that does not is
    # told which formulation to use instead.
    stub = SimpleNamespace(name="external")
    lab = np.zeros((1, 1, 1, 4), dtype=complex)
    with pytest.raises(BackendError, match="external does not execute"):
        StepDriver._adams_lab_source(stub, lab, 0.0, purpose="startup")  # type: ignore[arg-type]


def test_a_backend_without_a_coupled_flow_hands_the_march_no_block():
    # The default hook: every shipped backend that steps a coupled flow returns its
    # block; an external one that holds none gives the march nothing to step.
    stub = SimpleNamespace(name="external")
    assert StepDriver._adams_block(stub, 0.0, 1e-3) is None  # type: ignore[arg-type]


def test_the_formulation_refuses_a_cf4_section(narrow_grid, soliton_problem):
    # SEP 16 Q3: the exponential formulation steps constant-Schur and frozen-residual
    # sections (phase 5); a CF4 section is excluded, and the shared check names it.
    problem = soliton_problem(narrow_grid, 1, save_planes=[0.1], integrator=EXPONENTIAL)
    flow = constant_schur_flow(
        np.sort(problem.grids.omega),
        problem.linear.phi_breakpoints[[0, -1]],
        np.zeros((1, 1, problem.grids.omega.size), dtype=complex),
        dz_max=problem.stepper.dz_max,
        condition_limit=1e6,
        gauge_id="test",
        certificate_id="constant-schur",
    )
    coupled = dataclasses.replace(
        problem, linear=dataclasses.replace(problem.linear, coupled_flow=flow)
    )
    refuse_formulation(coupled, "probe")  # a constant-Schur section is stepped
    flow = object.__new__(type(flow))
    object.__setattr__(flow, "section_methods", ("cf4",))
    coupled = dataclasses.replace(
        problem, linear=dataclasses.replace(problem.linear, coupled_flow=flow)
    )
    with pytest.raises(BackendError, match="'cf4' section"):
        refuse_formulation(coupled, "probe")


def _chooser(method, collar_rate, *, quiet=False):
    """Return a bare march holding only what the order choice reads.

    Its backend stands in for the one flag the march reads from it: whether the stages
    leave a quiet collar out.
    """
    march = object.__new__(AdamsMarch)
    march._method = method
    march._policy = march_policy(
        method,
        dz_min=0.0,
        dz_max=1.0,
        stretch=1.1,
        z_end=1.0,
        save_planes=[],
        hard_kinks=[],
        breakpoints=[0.0, 1.0],
        interpolate=False,
        collar_rate=collar_rate,
    )
    march._order = 5
    # A stand-in, not a StepDriver: the order choice reads one flag from it.
    march._backend = cast("Any", SimpleNamespace(_collar_quiet=quiet))
    march._collar_was_quiet = False
    return march


def test_the_exponential_order_choice_weighs_the_in_stage_collar():
    # Order 5's defect allows the longer step (0.8 * 1e-3 ** (-1/6) = 2.53 against
    # order 4's 2.01), but a collar at which order 5 is stable only up to the step just
    # taken (PEC5/6's interval over h alpha / 2 = 1) leaves order 4 stable to 1.86 of
    # it: the exponential formulation takes order 4; without a collar it keeps 5.
    dz = 1e-4
    rate = PEC_REAL_INTERVAL[4] / dz
    scores = {5: 1e-3, 4: 1e-2}
    assert _chooser(EXPONENTIAL, rate)._choose_order(5, scores, dz) == (4, 1e-2)
    assert _chooser(EXPONENTIAL, 0.0)._choose_order(5, scores, dz) == (5, 1e-3)


def test_a_resting_collar_no_longer_weighs_the_order_choice():
    # While the stages leave a quiet collar out (an opted-in BoundaryCheck) there is no
    # stiff term to weigh the orders by: the same scores keep order 5.
    dz = 1e-4
    rate = PEC_REAL_INTERVAL[4] / dz
    scores = {5: 1e-3, 4: 1e-2}
    assert _chooser(EXPONENTIAL, rate, quiet=True)._choose_order(5, scores, dz) == (
        5,
        1e-3,
    )


def test_the_step_after_a_collar_returns_is_held_to_its_stable_step():
    # A long order-5 proposal made while the collar rested is held, on the first attempt
    # with the term back, to the step at which order 5 is stable for the collar; later
    # attempts and a march whose collar never rested pass the proposal through.
    rate = 1.0e3
    stable = PEC_REAL_INTERVAL[4] / rate
    march = _chooser(EXPONENTIAL, rate, quiet=True)
    assert march._collar_reentry(1.0) == 1.0
    march._backend._collar_quiet = False
    assert march._collar_reentry(1.0) == pytest.approx(stable)
    assert march._collar_reentry(1.0) == 1.0
    assert _chooser(EXPONENTIAL, 0.0)._collar_reentry(1.0) == 1.0


def test_a_new_run_clears_the_collar_flag_and_a_pending_floor():
    # A backend reused for another run starts it as a fresh one would (SEP 39 R9): the
    # collar flag of the last run's last attempt does not cap the new run's first
    # step after a collar returns.
    march = _chooser(EXPONENTIAL, 1.0e3, quiet=True)
    march._collar_was_quiet = True
    march._last_rejected = True
    march._floor = ConvergenceError("pending")
    march.begin_run()
    assert march._collar_was_quiet is False
    assert march._last_rejected is False
    assert march._floor is None
    march._raise_floor()


def test_a_resumed_collar_flag_caps_the_step_after_the_collar_returns():
    # The flag a snapshot restores decides the re-entry cap exactly as the
    # uninterrupted march's would: restored quiet, the first attempt with the term
    # back is held to the stable step; restored active, it is not.
    rate = 1.0e3
    stable = PEC_REAL_INTERVAL[4] / rate
    for restored, expected in ((True, stable), (False, 1.0)):
        march = _chooser(EXPONENTIAL, rate, quiet=False)
        march._collar_was_quiet = restored
        assert march._collar_reentry(1.0) == pytest.approx(expected)


@pytest.mark.parametrize("integrator", FORMULATIONS, ids=["interaction", "exponential"])
def test_only_the_exponential_formulation_reads_the_collar(integrator):
    # In the exponential formulation the in-stage collar is the one stiff term the
    # explicit history integrates; the interaction formulation's history is carried
    # through the propagators, whose stiffness the collar alone does not describe.
    import flow_parity  # noqa: PLC0415 - the capillary cases' builder

    with warnings.catch_warnings():
        # The capillary cases' weak collar and their materials' advisories.
        for category in (*flow_parity.QUIET, SolpropWarning):
            warnings.simplefilter("ignore", category)
        simulation = flow_parity.build("scalar", wall=None)
        simulation.settings = dataclasses.replace(
            simulation.settings, integrator=integrator
        )
        problem = simulation.translate()
    backend = NumpyRefBackend(problem, grid=simulation.grid, modes=simulation.modes)
    try:
        march = backend._adams
        assert march is not None
        collar = problem.responses.temporal_absorber
        assert collar is not None
        expected = 0.5 * float(np.max(collar))
        assert march._policy.collar_rate == (
            expected if integrator.formulation == "exponential" else 0.0
        )
    finally:
        backend.close()


def test_the_resident_engines_tables_are_the_adams_quadratures():
    """The tables a resident engine receives are the host march's quadratures.

    A resident engine (the native plugin's) takes its coefficients from these instead
    of forming them: the predictor integrates over the step every polynomial its
    ``order`` newest samples interpolate, the corrector every one through the new
    sample as well, and each scored order's defect -- its corrector less its predictor
    -- vanishes on the polynomials both integrate. The exponential tables are the
    Lagrange bases of the same nodes, and their factored defect's divided difference
    vanishes below its order and is one at it. Each identity is a Vandermonde solve's
    and is held to its condition number times ``gamma`` of its operations.
    """
    nodes = [0.0, -0.8, -1.7, -2.5, -3.1]
    order, orders = 3, [3, 2, 4]
    width = 1 + max(orders)
    points = np.array([1.0, *nodes[: width - 1]])

    def vandermonde(at):
        return np.vander(np.asarray(at), increasing=True).T

    def integrates(weights, at):
        powers = vandermonde(at)
        targets = 1.0 / np.arange(1, len(at) + 1)
        bound = float(np.linalg.cond(powers)) * gamma(2 * len(at))
        assert_within(powers @ weights, targets, bound, what="quadrature")

    fused = fused_tables(nodes, order, orders, helpers=PLAIN)
    compensated_tables = compensation_tables(nodes, order, orders, helpers=PLAIN)
    np.testing.assert_array_equal(
        compensated_tables[0], 1.0 - np.array(nodes[: width - 1])
    )
    for predictor, corrector, rows in (fused, compensated_tables[1:]):
        integrates(predictor, nodes[:order])
        integrates(corrector[: order + 1], points[: order + 1])
        assert not np.any(corrector[order + 1 :])
        for row, scored in zip(rows, orders, strict=True):
            powers = vandermonde(points[: scored + 1])[:scored]
            bound = float(np.linalg.cond(vandermonde(points[: scored + 1]))) * gamma(
                2 * width
            )
            assert_within(
                powers @ row[: scored + 1], np.zeros(scored), bound, scale=1.0
            )
            assert not np.any(row[scored + 1 :])

    predictor, corrector, factored = exponential_tables(
        nodes, order, orders, helpers=PLAIN
    )
    identity = np.eye(order)
    bound = float(np.linalg.cond(vandermonde(nodes[:order]))) * gamma(2 * order)
    assert_within(predictor @ vandermonde(nodes[:order]), identity, bound)
    through = points[: order + 1]
    bound = float(np.linalg.cond(vandermonde(through))) * gamma(2 * (order + 1))
    assert_within(
        corrector[: order + 1, : order + 1] @ vandermonde(through),
        np.eye(order + 1),
        bound,
    )
    assert not np.any(corrector[order + 1 :]) and not np.any(corrector[:, order + 1 :])
    for row, scored in zip(factored, orders, strict=True):
        powers = vandermonde(points[: scored + 1])
        expected = np.zeros(scored + 1)
        expected[scored] = 1.0
        bound = float(np.linalg.cond(powers)) * gamma(2 * width)
        assert_within(powers @ row[: scored + 1], expected, bound)
    # The engine reads all of them through one call: these, the compensation's when
    # it carries one, and empty arrays when it does not.
    for compensated in (False, True):
        tables = trial_tables(
            nodes, order, orders, compensated=compensated, helpers=PLAIN
        )
        parts = (predictor, corrector, factored)
        rest = compensated_tables if compensated else (np.empty(0),) * 3
        for table, part in zip(tables, (*parts, *rest), strict=False):
            np.testing.assert_array_equal(table, part)
        assert tables[-1].shape == ((len(orders), width) if compensated else (0, 0))
