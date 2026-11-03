# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""One thread and several on ``native``, with every pool forced to its width (SEP 43).

These hold the obligations of the ``numba_cpu`` thread tests SEP 43 retired, on their
inputs and bounds: the default Adams march, the z-varying capillary and its statistics
rows, the dispersive path's node reductions, the planar stage, the modified noise's
draw, and the transverse transform's pool. The retired tests forced their parallel
kernels with ``regime="large"``. Native's equivalent is an explicit pool
(``forced_threads.py``), and each test asserts from ``execution_sizing`` that the
threads ran, because a size rule that kept them serial would otherwise pass.

**What a witness here is weaker in.** The retired tests held a serial closure against a
parallel one, kernel by kernel, or held a few element roundings. Native has one kernel
body, so its threads are compared over whole runs, where the rest of the run's rounding
is in the budget too. Each test keeps the retired bound or counts a run budget from the
operations, stated before measuring, never from the difference it then accepts (SEP 25).
"""

import importlib.util
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from backends.dispersive_cases import capillary as dispersive_capillary
from backends.dispersive_cases import evaluation_bound as dispersive_evaluation
from backends.forced_thread_cases import (
    CAPILLARY_ROW_NAMES,
    accept_problem,
    field_grid,
    linear_cartesian,
    linear_flow_problem,
    noisy_problem,
    route_adams_problem,
    ten_steps,
    wide_transverse_problem,
    with_adams,
)
from backends.forced_threads import POOLS, forced_run, widest
from backends.low_storage_cases import parity_bound
from backends.matched import fixed_order
from backends.planar_cases import planar_case
from bounds import (
    assert_within,
    departure,
    gamma,
    higham_gamma,
    ladder_bound,
    step_evaluations,
)

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("solprop_native") is None,
        reason="solprop_native is not installed",
    ),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.GrazingIncidenceWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
]

EPS = np.finfo(float).eps

WIDTH = 4
"""The several threads compared with one, as the retired tests' four."""

ADAPTIVE_BOUND = 3e-9
"""The retired test's bound on an adaptive Adams march, of the peak.

The free-space routes' parity with the reference, ``test_exponential_routes.py``. This
is not a matched computation's rounding. Two adaptive runs can make the same decisions
and differ in step length, since the error norm's rounding moves the step the controller
chooses, and a threaded FFTW plan (or a measured one, process to process) rounds
differently: on the multimode capillary at one thread, repeats differ by 1.3e-10 under
``measure`` planning with their steps 1.9e-12 apart, and by nothing under ``estimate``.
"""

FORKED_BOUND = 100 * 1e-8
"""What two runs at ``rtol`` 1e-8 may differ by when they fork (SEP 25 section 4.4).

A rounding-level change of the error norm can move one accept or order decision. The
two are then different trajectories, each within its tolerance's error of the solution,
so this is accuracy (class T), a hundred times ``rtol``, not a rounding comparison.
"""


def assert_threads_ran(one: dict, several: dict, *operations: str) -> None:
    """Assert the first run took one thread and the second `WIDTH`, per operation."""
    for operation in operations:
        assert widest(one["sizing"], operation) == 1, operation
        assert widest(several["sizing"], operation) == WIDTH, operation


# --- the default integrator -------------------------------------------------------


@pytest.mark.medium
@pytest.mark.parametrize("formulation", ["interaction", "exponential"])
def test_an_adaptive_adams_march_makes_the_same_decisions_on_its_threads(
    tmp_path, formulation
):
    """``numba_cpu``'s ``test_a_large_problem_marches_on_the_parallel_passes``.

    Adams is the default integrator. On the retired test's input, the Cartesian route at
    ``rtol`` 1e-8: the same calls, and fields within :data:`ADAPTIVE_BOUND`. Its run is
    two steps, so :func:`test_ten_matched_adams_steps_agree_across_threads` and the
    routes below carry the longer witness.
    """
    case = f"adams_route:cartesian|{formulation}"
    one = forced_run(case, 1, tmp_path)
    several = forced_run(case, WIDTH, tmp_path)
    assert_threads_ran(one, several, "kernels.adams", "kernels.stage", "fft.fft")
    assert int(several["calls"]) == int(one["calls"])
    assert_within(
        several["field"],
        one["field"],
        ADAPTIVE_BOUND,
        scale=float(np.max(np.abs(one["field"]))),
        what="field",
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("route", "formulation"),
    [
        ("capillary (mode-averaged)", "interaction"),
        ("capillary (multimode)", "interaction"),
        ("capillary (pressure gradient)", "interaction"),
        ("radial", "exponential"),
        ("general guide (constant)", "exponential"),
    ],
)
def test_adaptive_adams_agrees_across_threads_on_the_other_routes(
    tmp_path, route, formulation
):
    """What the Cartesian route has not: a gas, a varying operator, hundreds of calls.

    At the retired test's bound, which ``test_exponential_routes.py`` holds these routes
    to against the reference, or at :data:`FORKED_BOUND` where the two forked. The
    exponential runs of the capillaries are left out: their in-stage collar holds them
    at its stable step, where rounding sets the steps (``COLLAR_HELD`` there; on the
    pressure gradient, 4 % apart, 2026-11-03), so they are not one discrete march.
    """
    case = f"adams_route:{route}|{formulation}"
    one = forced_run(case, 1, tmp_path)
    several = forced_run(case, WIDTH, tmp_path)
    assert_threads_ran(one, several, "kernels.stage", "fft.fft")
    assert widest(several["sizing"], "kernels.adams") in (0, WIDTH)
    same = int(several["calls"]) == int(one["calls"])
    assert_within(
        several["field"],
        one["field"],
        ADAPTIVE_BOUND if same else FORKED_BOUND,
        scale=float(np.max(np.abs(one["field"]))),
        what="field",
    )


def _adams_free_bound(problem) -> float:
    """Bound a matched Adams run on the Cartesian route, relatively.

    The Cartesian ladder's ``32 eps`` per FFT level, per evaluation
    (``test_free_ladders.py``), over every sample's weight in the step's predictor and
    corrector (:func:`bounds.step_evaluations`); and the exponent's ``2 eps k_max L``,
    the separable kinds' principal root, up to ``order + 1`` times in each of the two.
    """
    depth = math.ceil(math.log2(problem.grids.n_to)) + math.ceil(
        math.log2(problem.transform.n_nodes)
    )
    steps = problem.stepper.schedule.n_steps
    evaluations = steps * step_evaluations(problem)
    separable = problem.linear.separable
    root = math.sqrt(float(np.max(np.abs(separable.k_squared[separable.active]))))
    operator = 2 * EPS * root * float(problem.stepper.save_planes[-1])
    return evaluations * 32 * EPS * depth + operator * 2 * (
        problem.stepper.integrator.order + 1
    )


@pytest.mark.slow
@pytest.mark.parametrize("formulation", ["interaction", "exponential"])
def test_ten_matched_adams_steps_agree_across_threads(tmp_path, formulation):
    """Ten fixed steps at a fixed order: one discrete march, differing by rounding.

    The retired test's route on a matched computation, which holds the threads to a
    rounding budget where an adaptive march can only be held to ``3e-9``.
    """
    _, problem = route_adams_problem("cartesian", formulation)
    problem = fixed_order(ten_steps(problem))
    bound = _adams_free_bound(problem)
    case = f"adams_matched:cartesian|{formulation}"
    one = forced_run(case, 1, tmp_path)
    several = forced_run(case, WIDTH, tmp_path)
    assert_threads_ran(one, several, "kernels.adams", "kernels.stage", "fft.fft")
    assert_within(several["field"], one["field"], bound, what="field")
    assert departure(one["field"], problem.initial.field0) > 1e3 * bound


# --- the z-varying capillary ------------------------------------------------------


CAPILLARY_BOUND = 200 * 16 * EPS
"""The capillary rung's budget for the 200 steps to 0.1 m, 16 eps a step."""


@pytest.fixture(scope="module")
def capillary_runs(tmp_path_factory):
    """Replay the capillary at one and at four threads, once for both of its tests."""
    directory = tmp_path_factory.mktemp("capillary")
    return (
        forced_run("capillary", 1, directory),
        forced_run("capillary", WIDTH, directory),
    )


@pytest.mark.medium
def test_the_varying_capillary_agrees_across_threads(capillary_runs):
    """``numba_cpu``'s ``test_one_thread_and_all_of_them_agree``: its input and bound.

    A two-point gradient, a hard kink at the fill's boundary and the in-stage absorber,
    200 replayed steps. The rung's own budget against the reference holds here too
    (``test_a_capillary_run_with_an_absorber_and_a_kink_agrees``), with a device's
    larger envelope through :func:`bounds.ladder_bound`.
    """
    one, several = capillary_runs
    assert_threads_ran(one, several, "kernels.stage", "fft.fft")
    assert int(one["steps"]) == int(several["steps"]) == 200
    bound = ladder_bound("native", CAPILLARY_BOUND, field_grid().n_to, 200 * 7)
    assert_within(several["field"], one["field"], bound, what="four threads")


@pytest.mark.medium
def test_the_varying_capillary_s_rows_agree_across_threads(capillary_runs):
    """``numba_cpu``'s test of the statistics rows, on its input and bound.

    On a prescribed schedule the rows stand at the same positions and steps, exactly.
    Their values are squared fields and their sums, twice the field's budget and a
    reduction over the samples, with a factor of four for ``fwhm_t``'s interpolation.
    """
    one, several = capillary_runs
    assert one["z"].size > 0
    np.testing.assert_array_equal(several["z"], one["z"])
    np.testing.assert_array_equal(several["step"], one["step"])
    samples = field_grid().n_to
    field_bound = ladder_bound("native", CAPILLARY_BOUND, samples, 200 * 7)
    bound = 4 * (2 * field_bound + gamma(samples))
    for name in CAPILLARY_ROW_NAMES:
        assert_allclose(
            several[f"row_{name}"], one[f"row_{name}"], rtol=bound, err_msg=name
        )


# --- the dispersive path's node reductions ----------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("raman", [False, True], ids=["argon", "nitrogen, Raman"])
@pytest.mark.parametrize("evaluator", ["symmetric", "polynomial"])
def test_the_fused_dispersive_path_agrees_across_threads(tmp_path, evaluator, raman):
    """``numba_cpu``'s ``test_one_thread_and_four_agree``, on its 33 nodes.

    Native's namesake runs the mode-averaged route, one node, so it reduces over no
    nodes. These are the three-mode capillary's 33, in argon and in nitrogen with its
    Raman response: ten fixed-order steps, at the window's rounding budget, and a repeat
    at the same width to the same budget.
    """
    simulation = dispersive_capillary(
        evaluator, gas="N2" if raman else "Ar", raman=raman, stepped=True
    )
    problem = fixed_order(simulation.translate())
    bound = 10 * step_evaluations(problem) * dispersive_evaluation(problem)
    case = f"dispersive:{evaluator}-{'raman' if raman else 'plain'}"
    one = forced_run(case, 1, tmp_path)
    several = forced_run(case, WIDTH, tmp_path)
    operations = ["kernels.response", "kernels.adams", "fft.fft"]
    if raman:
        operations.append("raman_fft.raman")
    assert_threads_ran(one, several, *operations)
    assert_within(several["field"], one["field"], bound, what="four threads")
    assert_within(several["again"], several["field"], bound, what="repeat run")


# --- the planar stage -------------------------------------------------------------


def _planar_bound(grid, space, problem, steps: int) -> float:
    """Return ``numba_cpu``'s planar ladder bound of `steps` replayed Kerr steps.

    Per evaluation, ``32 eps`` per FFT level (time and y) and ``2 n_p gamma(n_x)`` for
    the slab rule's two real contractions. Over the run the operator's
    ``2 eps k_max L``, the principal root's, and under Adams that up to ``order + 1``
    times in each of the predictor and the corrector.
    """
    depth = math.ceil(math.log2(grid.n_to)) + math.ceil(math.log2(space.n_y))
    evaluation = 32 * EPS * depth + 2 * space.n_p * gamma(space.x.size)
    separable = problem.linear.separable
    root = math.sqrt(float(np.max(np.abs(separable.k_squared[separable.active]))))
    operator = 2 * EPS * root * grid.length
    method = problem.stepper.integrator
    if method is not None:
        operator *= 2 * (method.order + 1)
    return operator + max(1.0, step_evaluations(problem) * steps) * evaluation


@pytest.mark.slow
@pytest.mark.parametrize("march", ["rk", "adams"])
def test_the_planar_in_stage_collar_agrees_across_threads(tmp_path, march):
    """``numba_cpu``'s ``test_one_thread_and_four_agree`` and ``test_two_runs_agree``.

    ``kerr-collar``, with the collar in the stage, at the retired ladder bound; the
    case takes the Runge-Kutta march unless it is given a method, and ``"adams"`` gives
    it the default one, which native's planar thread test has not run. A repeat at the
    same width holds the second retired test, at one thread and at four.
    """
    grid, space, problem = planar_case("kerr-collar")
    if march == "adams":
        problem = with_adams(problem)
    problem = fixed_order(problem)
    bound = _planar_bound(grid, space, problem, problem.stepper.schedule.n_steps)
    case = f"planar:{march}"
    one = forced_run(case, 1, tmp_path)
    several = forced_run(case, WIDTH, tmp_path)
    operations = ["kernels.response", "kernels.stage", "fft.fft"]
    if march == "adams":
        operations.append("kernels.adams")
    assert_threads_ran(one, several, *operations)
    assert_within(several["field"], one["field"], bound, what="four threads")
    assert_within(one["again"], one["field"], bound, what="repeat at one thread")
    assert_within(several["again"], several["field"], bound, what="repeat at four")


# --- the modified noise -----------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("march", ["march", "rk", "adams"])
def test_the_modified_noise_draw_agrees_across_threads(tmp_path, march):
    """``test_spatial_execution.py``'s parallel combine, at a run's budget.

    The retired test held the stage combine's noise term, 16 eps, against its serial
    loop. Native's combine is not reachable alone, so this is a beam beside a draw over
    a short run, whose budget (:func:`parity_bound` for the march, the free ladder's for
    the others) is far looser. It is weaker in that, and is here to catch the draw's
    term depending on the thread count: the draw moves the answer by more than that.
    """
    problem, grid, _ = noisy_problem(march)
    transverse = problem.transform.n_nodes
    if march == "march":
        bound = parity_bound(problem, transverse=transverse)
    else:
        depth = math.ceil(math.log2(grid.n_to)) + math.ceil(math.log2(transverse))
        steps = problem.stepper.schedule.n_steps
        bound = steps * step_evaluations(problem) * 32 * EPS * depth
    one = forced_run(f"noisy:{march}", 1, tmp_path)
    several = forced_run(f"noisy:{march}", WIDTH, tmp_path)
    assert_threads_ran(one, several, "kernels.stage", "kernels.response", "fft.fft")
    assert_within(several["field"], one["field"], bound, what="four threads")
    assert departure(one["field"], one["quiet"]) > bound


# --- the transverse transform's pool (#322) ---------------------------------------


@pytest.mark.medium
@pytest.mark.parametrize("threads", [2, 4])
def test_the_transverse_pool_threads_its_transform_and_agrees(tmp_path, threads):
    """``numba_cpu``'s ``test_whole_runs_agree_at_1_2_and_4_threads`` with a used pool.

    The march on a plane whose transverse transform is above
    ``threaded_fft_min_samples``, so that the automatic rule threads it, at the retired
    bound. :func:`parity_bound` counts a transverse pair per evaluation. An FFT plan's
    rounding can depend on its width, so ``execution_sizing`` must show the plan
    threaded. The rule's other pools stay automatic here; the next test varies this one.
    """
    problem, _ = wide_transverse_problem()
    samples = math.prod(problem.shape)
    key = f"fft.fft.{samples}"
    bound = parity_bound(problem, transverse=problem.transform.n_nodes)
    one = forced_run("wide_transverse", 1, tmp_path, pools={})
    several = forced_run("wide_transverse", threads, tmp_path, pools={})
    assert one["sizing"][key]["threads"] == 1
    assert several["sizing"][key]["threads"] == threads
    assert_within(several["field"], one["field"], bound, what=f"{threads} threads")


@pytest.mark.medium
@pytest.mark.parametrize("march", ["march", "adams"])
def test_the_transverse_pool_varied_alone_agrees(tmp_path, march):
    """The transform's pool at four threads with the kernels held at one (#322).

    What the retired test's three pools moved together could not separate: the plan's
    width alone, reported by the sizing, at the same bound. The Adams row is the march
    #322 recommends beside it, the default one, on its own rounding budget.
    """
    problem, _ = wide_transverse_problem(march)
    key = f"fft.fft.{math.prod(problem.shape)}"
    bound = (
        parity_bound(problem, transverse=problem.transform.n_nodes)
        if march == "march"
        else _adams_free_bound(problem)
    )
    pools = dict.fromkeys(POOLS, 1) | {"SOLPROP_NATIVE_FFT_THREADS": WIDTH}
    one = forced_run(f"wide_transverse:{march}", 1, tmp_path)
    several = forced_run(f"wide_transverse:{march}", WIDTH, tmp_path, pools=pools)
    assert one["sizing"][key]["threads"] == 1
    assert several["sizing"][key]["threads"] == WIDTH
    assert widest(several["sizing"], "kernels.stage") == 1
    assert_within(several["field"], one["field"], bound, what="the FFT pool alone")


# --- the passes of a trial, one at a time --------------------------------------------


@pytest.mark.medium
def test_a_threaded_accept_reduces_the_state_it_wrote(tmp_path):
    """``numba_cpu``'s ``test_the_accept_pass_is_the_reference_on_a_threaded_state``.

    The accept's partials are native's own sums: the old and new state's squared norm
    per column, over the norm mask. Each is held, per column and at each width, to the
    retired ``2 gamma(n_omega + 2)`` of a host sum over the array the pass wrote, which
    leaves the transforms out and so keeps the retired test's strength. The advanced
    state cannot be read apart from the stages behind it: it and the residual's norm
    are held to one trial's budget, weaker than the retired ``2 eps`` on the pass alone.
    """
    _, problem = accept_problem()
    n_omega = problem.grids.n_omega
    depth = math.ceil(math.log2(problem.grids.n_to)) + math.ceil(
        math.log2(problem.transform.n_nodes)
    )
    state_bound = 7 * 32 * EPS * depth
    one = forced_run("accept", 1, tmp_path)
    several = forced_run("accept", WIDTH, tmp_path)
    assert_threads_ran(one, several, "kernels.stage", "fft.fft")
    mask = one["mask"]
    assert np.count_nonzero(~mask) > 0
    assert one["partials"].shape == (3, math.prod(problem.shape[:-1]))
    bound = 2 * higham_gamma(n_omega + 2)
    for run in (one, several):
        for row, state in ((0, run["launch"]), (1, run["advanced"])):
            expected = np.sum(np.abs(state[:, mask]) ** 2, axis=1)
            assert_within(run["partials"][row], expected, bound, what=f"partial {row}")
    assert_within(several["advanced"], one["advanced"], state_bound, what="state")
    # Norms move by at most the norm of what moved their operand, and the transforms'
    # rounding is normwise across the columns: over the columns, one trial's budget of
    # the whole state, the residual being a difference of the stages. Exact per column
    # would not be a bound, since a quiet column's transverse rounding comes from a loud
    # one's.
    first, second = np.sqrt(one["partials"][2]), np.sqrt(several["partials"][2])
    assert np.any(first > 0)
    scale = float(np.sqrt(np.sum(one["partials"][0])))
    assert float(np.linalg.norm(second - first)) <= state_bound * scale


@pytest.mark.medium
@pytest.mark.parametrize(
    ("mode", "threads"), [("tabulate", 2), ("tabulate", 4), ("recompute", 4)]
)
def test_the_low_storage_folds_agree_across_threads(tmp_path, mode, threads):
    """``numba_cpu``'s ``test_the_parallel_kernels_agree_with_...``, through a run.

    With no response a pass is its fold alone, with no transform, so its rounding is
    what is compared: the retired ``4 eps`` per fold over the run's four folds a step,
    and the engine's squared norm per step to ``n_columns eps``. Weaker in that the
    retired test drove every fold mode and flow treatment on random registers, each at
    once; here the march chooses them, on a beam, and there is no source to fold.
    """
    problem, _ = linear_cartesian(mode)
    folds = 4 * problem.stepper.schedule.n_steps
    one = forced_run(f"folds:{mode}", 1, tmp_path)
    several = forced_run(f"folds:{mode}", threads, tmp_path)
    assert widest(several["sizing"], "kernels.stage") == threads
    assert_within(several["field"], one["field"], folds * 4 * EPS, what="field")
    assert_allclose(
        several["norms"], one["norms"], rtol=math.prod(problem.shape[:-1]) * EPS, atol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", ["scalar", "te_tm", "taper", "split"])
def test_the_threaded_flow_passes_agree_across_threads(tmp_path, name):
    """``numba_cpu``'s ``test_the_threaded_flow_passes_are_the_serial_ones``, one trial.

    With no source a linear trial is the flow's passes alone, each of the seven stages'
    combinations and the accept at the retired ``8 eps`` an output, and each partial at
    ``2 gamma(n_omega + 2)`` beside twice the state's. Weaker in that no source reaches
    the accumulation, and only the routed bins and the masks the case has are driven;
    the retired test drew each operand at random. The residual's norm is exactly zero
    on the capillaries, which take a constant flow.
    """
    problem, _, _ = linear_flow_problem(name)
    state_bound = (problem.stepper.tableau.n_stages + 1) * 8 * EPS
    one = forced_run(f"flow:{name}", 1, tmp_path)
    several = forced_run(f"flow:{name}", WIDTH, tmp_path)
    assert_threads_ran(one, several, "kernels.flow", "kernels.stage")
    assert_within(several["field"], one["field"], state_bound, what="state")
    bound = 2 * higham_gamma(problem.grids.n_omega + 2) + 2 * state_bound
    for row in (0, 1):
        assert_within(
            several["partials"][row], one["partials"][row], bound, what=f"partial {row}"
        )


@pytest.mark.slow
@pytest.mark.parametrize("name", ["scalar_wall", "taper_gradient"])
def test_the_flow_with_its_source_agrees_on_forced_pools(tmp_path, name):
    """Ten steps of the flow beside its source, the flow's own pool forced.

    ``test_the_flow_is_invariant_under_the_thread_count`` runs these sizes under the
    size rule, which keeps their ``kernels.flow`` passes serial below 32,768 elements,
    so it does not thread them. This is its test with the pools forced, at its bound.
    """
    from driver.test_native_coupled_flow import (  # noqa: PLC0415
        CASES as FLOWS,
    )
    from driver.test_native_coupled_flow import (  # noqa: PLC0415
        parity_bound as flow_parity_bound,
    )
    from driver.test_native_coupled_flow import reference_and_linear  # noqa: PLC0415

    case = FLOWS[name]()
    bound = flow_parity_bound(case, *reference_and_linear(case))
    one = forced_run(f"flow_kerr:{name}", 1, tmp_path)
    several = forced_run(f"flow_kerr:{name}", WIDTH, tmp_path)
    assert_threads_ran(one, several, "kernels.flow", "kernels.stage")
    assert float(np.max(np.abs(several["field"] - one["field"]))) <= bound
