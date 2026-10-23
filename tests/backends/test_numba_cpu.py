# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The CPU backend against the reference, and against its own promises.

The validation ladder of ``docs/dev/backends.md`` section 3.7 -- one linear application,
one nonlinear evaluation, one step, eight steps, each statistic, a full save interval, a
full run -- plus the claims that are this backend's alone: that two runs of it, and one
thread and all of them, agree on a fixed schedule within the rung's rounding budget,
that every buffer is named and inside the budget, and that a run resumed on it
continues where it stopped.

**The bounds are derived, not fitted.** The reference and this backend perform the same
arithmetic in the same order wherever they can, and where they cannot the difference has
a measured cause: numpy's complex multiply contracts into a fused multiply-add on this
host and numba's does not (1.42 eps per product), numpy forms a modulus as
``np.abs(z) ** 2`` where a kernel sums squares (1.4 eps), and pocketfft and FFTW differ
by ``log2(n) eps`` over a transform. A step is a few of each, so a step agrees to a few
eps and a run of ``n`` steps to a few ``n`` eps at worst -- and, because the two
trajectories are the same to that accuracy, the controller takes the same steps and the
errors do not compound the way an independent integration's would. Every bound below is
stated with the number measured against it.
"""

import dataclasses
import math
import os
import tracemalloc

import numpy as np
import pytest
from numpy.testing import assert_allclose

from backends.matched import stash_bound
from bounds import (
    AccuracyEnvelope,
    analytic_bound,
    assert_within,
    assert_work_within,
    departure,
    gamma,
    higham_gamma,
    ladder_bound,
    record_backend,
)
from convergence_records import record
from convergence_studies import BACKEND_EQUIVALENCE
from solprop.backends.numba_cpu import NumbaCpuBackend, openmp
from solprop.backends.numba_cpu import fft as numba_fft
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.backends.registry import select
from solprop.exceptions import BackendError
from solprop.nonlinear.ionisation import RunningIntegral
from solprop.stats import StatsSpec
from solprop.stepper import DOPRI5, FixedSchedule, PIController

EPS = np.finfo(float).eps


@pytest.fixture
def spm_field_problem(field_grid):
    """A field grid whose Kerr term does not radiate the harmonic: the third family.

    ``kerr_coefficients`` compiles ``thg=False`` on a field grid to the analytic form
    with ``analytic_signal`` set, which is the only configuration that makes the column
    work on the Hilbert transform of the real field.
    """
    import warnings  # noqa: PLC0415 - helium is outside its index model's range

    from solprop.exceptions import MaterialRangeWarning  # noqa: PLC0415
    from solprop.modes.capillary import MarcatiliMode  # noqa: PLC0415
    from solprop.nonlinear.responses import KerrResponse  # noqa: PLC0415
    from solprop.profiles import Constant  # noqa: PLC0415
    from solprop.pulses import GaussianPulse  # noqa: PLC0415
    from solprop.stepper.controller import PIController  # noqa: PLC0415
    from solprop.translate.problem import assemble  # noqa: PLC0415

    mode = MarcatiliMode(125e-6, "He", Constant(1.0), cladding=1.45)
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-6).build(field_grid)

    def build_problem():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MaterialRangeWarning)
            return assemble(
                field_grid,
                mode,
                [KerrResponse(chi3=2.2e-52, thg=False)],
                start,
                controller=PIController(rtol=1e-9, atol=1e-15),
                save_planes=[0.05],
                dz_max=0.01,
                dz_init=1e-4,
            )

    return build_problem


def run(backend_class, problem, grid, planes, *, modes=None, noise=None, **options):
    """Propagate a problem to every plane and return the fields and the summaries."""
    backend = backend_class(problem, grid=grid, modes=modes, **options)
    try:
        backend.set_initial(problem.initial.field0, noise)
        fields, summaries = [], []
        for plane in planes:
            summaries.append(backend.propagate_to(plane))
            fields.append(backend.observe().require_field().copy())
        return fields, summaries
    finally:
        backend.close()


def disagreement(ours, theirs):
    """Return the relative L2 difference of two states."""
    return float(np.linalg.norm(ours - theirs) / np.linalg.norm(theirs))


def rung(request, backend_name, name, position, measured, bound):
    """Record one rung of the equivalence ladder of `docs/dev/accuracy.md` section 7.

    The bound travels with the measurement, because what the figure is about is the
    distance between them: every one is derived from a summation length first and
    measured second, and a bound quietly loosened to fit a measurement is the failure
    the record exists to catch.
    """
    record_name = record_backend(backend_name)
    record(
        BACKEND_EQUIVALENCE,
        request=request,
        backend=record_name,
        rows=[
            {
                "series": f"{record_name} against numpy_ref",
                "axis": name,
                "deviation": measured,
                "bound": float(bound),
                "order": float(position),
            }
        ],
    )


# --- the ladder ------------------------------------------------------------------


def test_the_linear_flow_is_the_reference_s(
    equivalence_backend_class,
    equivalence_backend_name,
    linear_problem,
    narrow_grid,
    zero_kerr,
    request,
):
    """Rung P0: with no nonlinear term the step is the exact linear flow.

    Not bitwise, though the arithmetic is only two complex products per step: numpy
    contracts them into fused multiply-adds and numba does not. Bound 4 eps per step
    over 8 steps, so 32 eps; measured 1.1e-16, which is under one.
    """
    problem = linear_problem(narrow_grid, save_planes=[0.05], responses=[zero_kerr])
    ours, _ = run(equivalence_backend_class, problem, narrow_grid, [0.05])
    theirs, _ = run(NumpyRefBackend, problem, narrow_grid, [0.05])
    measured = disagreement(ours[0], theirs[0])
    assert measured < 32 * EPS
    rung(
        request,
        equivalence_backend_name,
        "one linear application",
        0,
        measured,
        32 * EPS,
    )


def test_one_step_of_the_soliton_agrees(
    equivalence_backend_class,
    equivalence_backend_name,
    soliton_problem,
    soliton_grid,
    request,
):
    """Rung P1, one step: one nonlinear evaluation through the whole pipeline."""
    problem = soliton_problem(
        soliton_grid,
        3,
        save_planes=[1e-3],
        schedule=FixedSchedule.uniform(0.0, 1e-3, 1e-3),
    )
    ours, _ = run(equivalence_backend_class, problem, soliton_grid, [1e-3])
    theirs, _ = run(NumpyRefBackend, problem, soliton_grid, [1e-3])
    # One step is seven stages of a few complex products and two transforms each;
    # 16 eps by Higham on log2(8192), measured 2.4e-16.
    measured = disagreement(ours[0], theirs[0])
    bound = ladder_bound(equivalence_backend_name, 16 * EPS, soliton_grid.n_to, 7)
    assert measured < bound
    rung(request, equivalence_backend_name, "one step", 1, measured, bound)


def test_eight_steps_of_the_soliton_agree(
    equivalence_backend_class,
    equivalence_backend_name,
    soliton_problem,
    soliton_grid,
    request,
):
    """Rung P1, eight steps: the accumulation over a schedule, not one evaluation."""
    problem = soliton_problem(
        soliton_grid,
        3,
        save_planes=[8e-3],
        schedule=FixedSchedule.uniform(0.0, 8e-3, 1e-3),
    )
    ours, summaries = run(equivalence_backend_class, problem, soliton_grid, [8e-3])
    theirs, reference = run(NumpyRefBackend, problem, soliton_grid, [8e-3])
    assert summaries[0].n_steps == reference[0].n_steps == 8
    measured = disagreement(ours[0], theirs[0])
    bound = ladder_bound(
        equivalence_backend_name, 8 * 16 * EPS, soliton_grid.n_to, 8 * 7
    )
    assert measured < bound
    rung(request, equivalence_backend_name, "eight steps", 2, measured, bound)


def test_a_full_adaptive_run_meets_the_reference_s_accuracy(
    equivalence_backend_class,
    equivalence_backend_name,
    soliton_problem,
    soliton_grid,
    request,
):
    """A whole run, adaptively: within the oracle's accuracy, at the oracle's work.

    Two adaptive integrations need not take the same steps: a rounding difference can
    tip an accept/reject decision, after which they are different integrations of one
    problem (SEP 25 section 4.4). Each is therefore held to an accuracy envelope fixed
    before the compiled run: its departure from a run at ``rtol = 1e-13`` may be at
    most four times the oracle's own at the same settings
    (:class:`bounds.AccuracyEnvelope`), and its step count is a work band of a factor
    1.25 about the oracle's. The cross-backend departure is still recorded against the
    ladder's 1600 eps, the rung of ``docs/dev/accuracy.md`` section 7, as a diagnostic.
    """
    planes = [0.02, 0.05]
    problem = soliton_problem(soliton_grid, 3, save_planes=planes)
    tight = soliton_problem(
        soliton_grid,
        3,
        save_planes=planes,
        controller=PIController(rtol=1e-13, atol=1e-20),
    )
    converged, _ = run(NumpyRefBackend, tight, soliton_grid, planes)
    theirs, reference = run(NumpyRefBackend, problem, soliton_grid, planes)
    envelope = AccuracyEnvelope(
        reference="numpy_ref at rtol 1e-13, N = 3 soliton, both planes, relative L2",
        error=max(
            disagreement(one, best) for one, best in zip(theirs, converged, strict=True)
        ),
        factor=4.0,
    )
    assert envelope.error > 0
    ours, summaries = run(equivalence_backend_class, problem, soliton_grid, planes)
    for plane, (ourf, best) in enumerate(zip(ours, converged, strict=True)):
        assert disagreement(ourf, best) <= envelope.bound, plane
    for one, other in zip(summaries, reference, strict=True):
        assert_work_within(one.n_steps, other.n_steps, factor=1.25, what="steps")
    bound = ladder_bound(
        equivalence_backend_name, 100 * 16 * EPS, soliton_grid.n_to, 100 * 7
    )
    worst = max(
        disagreement(ourf, theirf) for ourf, theirf in zip(ours, theirs, strict=True)
    )
    rung(
        request,
        equivalence_backend_name,
        "a full adaptive run",
        3,
        worst,
        bound,
    )


def test_a_capillary_run_with_an_absorber_and_a_kink_agrees(
    equivalence_backend_class,
    equivalence_backend_name,
    capillary_problem,
    field_grid,
    request,
    substitute,
):
    """A field grid, the real cubic, a pressure gradient, a hard kink, an absorber.

    Everything the soliton case does not exercise: the ``field_real`` column, a
    z-varying operator with a breakpoint the march has to land on, and the in-stage
    temporal absorber, whose term the stage carries.

    **On a replayed schedule**, which is what makes this a comparison of the two
    backends' arithmetic rather than of two adaptive trajectories. With the absorber in
    the derivative it is also in the error norm, so the round-off between the two
    backends now reaches the controller: run adaptively they take 111 and 94 steps
    against 111 and 91, and the second plane is then a different trajectory rather than
    a different rounding of the same one. Holding the schedule removes the fork and the
    two agree at the bound they always did.
    """
    problem = capillary_problem()
    fixed = substitute(
        problem,
        stepper=dataclasses.replace(
            problem.stepper,
            # Below the collar's own stability limit of 1.1 mm (ADR-025).
            schedule=FixedSchedule.uniform(0.0, 0.2, 0.0005, save_planes=[0.1, 0.2]),
        ),
    )
    ours, summaries = run(equivalence_backend_class, fixed, field_grid, [0.1, 0.2])
    theirs, reference = run(NumpyRefBackend, fixed, field_grid, [0.1, 0.2])
    assert [one.n_steps for one in summaries] == [one.n_steps for one in reference]
    bound = ladder_bound(
        equivalence_backend_name, 200 * 16 * EPS, field_grid.n_to, 200 * 7
    )
    worst = 0.0
    for ourf, theirf in zip(ours, theirs, strict=True):
        measured = disagreement(ourf, theirf)
        assert measured < bound
        worst = max(worst, measured)
    rung(
        request,
        equivalence_backend_name,
        "a capillary with an absorber and a kink",
        4,
        worst,
        bound,
    )


def test_the_analytic_field_family_agrees(
    equivalence_backend_class,
    equivalence_backend_name,
    spm_field_problem,
    field_grid,
    request,
):
    """The third family: a field grid whose Kerr term works on the analytic signal.

    This is the one where the two do not compute the same transform at all. The
    reference forms the analytic signal with a whole complex pair inside the Kerr pass,
    once per node chunk; this backend gets it from the inverse plan it was already
    running, by doubling the positive bins. Same operator, different route, so the bound
    is the transform's rather than the arithmetic's.
    """
    problem = spm_field_problem()
    ours, _ = run(equivalence_backend_class, problem, field_grid, [0.05])
    theirs, _ = run(NumpyRefBackend, problem, field_grid, [0.05])
    measured = disagreement(ours[0], theirs[0])
    bound = analytic_bound(equivalence_backend_name, field_grid.n_to)
    assert measured < bound
    rung(
        request,
        equivalence_backend_name,
        "the analytic field family",
        5,
        measured,
        bound,
    )


# --- what is this backend's alone -------------------------------------------------


def test_two_runs_agree(equivalence_backend_class, soliton_problem, soliton_grid):
    """Repeatability: two runs of the eight-step rung on its fixed schedule.

    The same discrete computation twice, so the rung's own budget holds (``8 x 16
    eps``, :func:`test_eight_steps_of_the_soliton_agree`) and nothing more is promised:
    a threaded reduction may order its sums differently from one run to the next.
    """
    problem = soliton_problem(
        soliton_grid,
        3,
        save_planes=[8e-3],
        schedule=FixedSchedule.uniform(0.0, 8e-3, 1e-3),
    )
    first, _ = run(equivalence_backend_class, problem, soliton_grid, [8e-3])
    second, _ = run(equivalence_backend_class, problem, soliton_grid, [8e-3])
    assert_within(second[0], first[0], 8 * 16 * EPS, what="repeat run")


def _replayed_capillary(capillary_problem, substitute, *, names=None):
    """Return the capillary rung's problem on its fixed 0.5 mm schedule.

    Below the collar's own stability limit of 1.1 mm (ADR-025). A run to the first
    plane, 0.1 m, takes 200 Runge-Kutta steps, so two thread counts take the same
    discrete method and differ by rounding alone.
    """
    problem = capillary_problem()
    stepper = dataclasses.replace(
        problem.stepper,
        schedule=FixedSchedule.uniform(0.0, 0.2, 0.0005, save_planes=[0.1, 0.2]),
    )
    fixed = substitute(problem, stepper=stepper)
    if names is not None:
        fixed = dataclasses.replace(
            fixed, stats_spec=StatsSpec(names=names, periodic=names, period=1)
        )
    return fixed


CAPILLARY_BOUND = 200 * 16 * EPS
"""The capillary rung's budget for the 200 steps to 0.1 m, 16 eps a step."""


def test_one_thread_and_all_of_them_agree(capillary_problem, substitute, field_grid):
    """What the two-stage reductions are for: the same steps at any thread count.

    Every reduction writes one partial per column and the driver sums them on the
    host, so a thread count changes at most the order of a sum: the capillary rung's
    rounding budget of 16 eps a step (:data:`CAPILLARY_BOUND`).
    """
    problem = _replayed_capillary(capillary_problem, substitute)
    serial, one = run(NumbaCpuBackend, problem, field_grid, [0.1], threads=1)
    parallel, many = run(NumbaCpuBackend, problem, field_grid, [0.1], threads=4)
    assert one[0].n_steps == many[0].n_steps == 200
    assert_within(parallel[0], serial[0], CAPILLARY_BOUND, what="four threads")


def test_one_thread_and_all_of_them_agree_on_the_statistics_rows(
    capillary_problem, substitute, field_grid
):
    """The channel's numbers are reductions too, so they carry the same promise.

    The rows go through :meth:`stats`, whose field statistics reduce per column and
    are summed on the host. On the fixed schedule the rows stand at the same
    positions and steps, exactly; their values are squared fields and their sums,
    twice the field's budget and a reduction over the samples, with a factor of four
    for the half-maximum interpolation of ``fwhm_t``.
    """
    names = ("energy", "peak_power", "fwhm_t")
    problem = _replayed_capillary(capillary_problem, substitute, names=names)

    def rows_at(threads):
        backend = NumbaCpuBackend(problem, grid=field_grid, threads=threads)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(0.1)
            return backend.observe().step_stats
        finally:
            backend.close()

    serial, parallel = rows_at(1), rows_at(4)
    assert serial.n_rows == parallel.n_rows > 0
    # A prescribed schedule puts its rows at the same positions: structural.
    assert np.array_equal(serial.z, parallel.z)
    assert np.array_equal(serial.step, parallel.step)
    bound = 4 * (2 * CAPILLARY_BOUND + gamma(field_grid.n_to))
    for name in names:
        assert_allclose(
            parallel.values[name], serial.values[name], rtol=bound, err_msg=name
        )


def _plasma_channel(problem, names=("electron_density", "peak_ionisation_rate")):
    """Return `problem` asking for `names` between planes as well as at them."""
    return dataclasses.replace(
        problem,
        stats_spec=StatsSpec(names=names, periodic=names, period=1),
    )


@pytest.mark.parametrize(
    ("plasma", "spatial", "parallel", "components", "selected"),
    [
        (False, False, False, 1, False),
        (True, True, False, 1, False),
        (True, False, False, 1, True),
        (True, False, False, 2, True),
        (True, False, True, 1, True),
        (True, False, True, 2, False),
    ],
)
@pytest.mark.parametrize("channels", [1, 2])
def test_fraction_reuse_selects_the_measured_column_domain(
    plasma, spatial, parallel, components, selected, channels
):
    """The policy uses physical response components, including xy modal rows."""
    from solprop.backends.numba_cpu import backend, kernels  # noqa: PLC0415

    spec = kernels.ColumnSpec(
        family="field_real",
        n_pol=components,
        raman_thg=False,
        harmonic=False,
        raman=False,
        plasma=plasma,
        parallel=parallel,
        stash=False,
        n_channels=channels,
    )
    assert backend._reuse_plasma_fraction(spec, spatial) is (
        selected if channels == 1 else False
    )


def _seeded_bounds(problem, grid, outcome, structural=()):
    """Bound each recorded quantity of the ionising run by its seeded response.

    The method of :func:`test_every_statistic_matches_the_reference`: this run is
    strongly ionising and ill-conditioned, so two computations that differ by rounding
    at every stage are bounded per quantity by its measured response to a one-ulp
    launch perturbation (the largest of three random-sign perturbations, normwise)
    times the count of seeds -- two steps of seven stages, each at most a transform's
    ``7 log2(n_to)`` ulp -- plus ``gamma(n_to)`` for the quantity's own evaluation.
    `outcome` maps a launch to the recorded arrays; `structural` names those that are
    exact and need no bound.
    """
    launch = np.asarray(problem.initial.field0)
    base = outcome(launch)
    response = dict.fromkeys(
        (name for name in base if name not in structural and np.any(base[name])), 0.0
    )
    generator = np.random.default_rng(7)
    for _ in range(3):
        signs = generator.choice([-1.0, 1.0], size=launch.shape)
        moved = outcome(launch * (1 + EPS / 2 * signs))
        for name in response:
            response[name] = max(response[name], departure(moved[name], base[name]))
    seeds = 2 * DOPRI5.n_stages * 7 * math.log2(grid.n_to)
    return {name: seeds * value + gamma(grid.n_to) for name, value in response.items()}


def _reuse_capture(backend, launch, spec, stash):
    """Run the ionising capillary from `launch` and copy everything the step wrote."""
    backend.set_initial(launch, None)
    backend.propagate_to(2e-3)
    observation = backend.observe()
    history = observation.step_history
    assert history is not None
    captured = {
        "field": observation.field.copy(),
        "z": history.z.copy(),
        "dz": history.dz.copy(),
        "norm": history.error_norm.copy(),
        "accepted": history.accepted.copy(),
        "recycled": backend._recycled.copy(),
        **{name: value.copy() for name, value in backend.stats(spec).items()},
    }
    if stash:
        captured["stash"] = backend._stash.copy()
        rows = observation.step_stats
        assert rows is not None and rows.n_rows > 0
        captured.update(
            {f"step_{name}": values.copy() for name, values in rows.values.items()}
        )
    return captured


@pytest.mark.parametrize("regime,threads", [("small", 1), ("large", 4)])
@pytest.mark.parametrize("stash", [False, True])
@pytest.mark.parametrize("propagator", ["recompute", "tabulate"])
def test_fraction_reuse_preserves_propagation_statistics_and_owned_buffers(
    ionising_capillary, field_grid, regime, threads, stash, propagator
):
    """The selected column agrees with the original, including the accepted stash.

    It reuses the fraction the original column recomputes, so the two are one
    computation by two routes on one fixed schedule: a rounding comparison.
    """
    from solprop.backends.numba_cpu import fft, kernels  # noqa: PLC0415

    problem, modes = ionising_capillary
    if stash:
        problem = _plasma_channel(problem)
    # The reuse under test is the fused trapezoid column's; the band column reuses its
    # fraction too, and is held to the reference by its own tests.
    problem = dataclasses.replace(
        problem,
        stepper=dataclasses.replace(problem.stepper, propagator_mode=propagator),
        responses=dataclasses.replace(
            problem.responses, plasma_integral=RunningIntegral.trapezoid()
        ),
    )
    spec = StatsSpec(names=("energy", "electron_density", "peak_ionisation_rate"))
    options = {
        "modes": modes,
        "regime": regime,
        "threads": threads,
        "fftw_threads": 1,
        "blas_threads": 1,
        "plan_flag": "estimate",
    }

    def capture(backend, launch):
        return _reuse_capture(backend, launch, spec, stash)

    def capture_with(launch):
        backend = NumbaCpuBackend(problem, grid=field_grid, **options)
        try:
            return capture(backend, launch)
        finally:
            backend.close()

    captures, ledgers = [], []
    for original in (True, False):
        backend = NumbaCpuBackend(problem, grid=field_grid, **options)
        try:
            assert backend.metadata()["plasma_fraction_reuse"] is True
            # The backend caps the request at the CPUs this process may run (a
            # three-core runner gives three).
            assert backend.metadata()["numba_threads"] == min(
                threads, fft._available_threads()
            )
            assert backend._spec.parallel is (regime == "large")
            if original:
                backend._column = kernels.make_response_column(backend._spec)
            captures.append(capture(backend, problem.initial.field0))
            # FFTW planning heap deltas depend on allocator reuse. Compare the
            # owned arrays exactly, as before SEP 27 added its two stream plans.
            ledgers.append(
                {
                    name: size
                    for name, size in backend.memory_ledger().items()
                    if not name.endswith("fft_plan_heap_growth")
                }
            )
        finally:
            backend.close()
    assert ledgers[0] == ledgers[1]
    assert captures[0].keys() == captures[1].keys()
    # The two columns are one computation by two routes, on one fixed schedule: the
    # schedule and the acceptances are structural and exact; every number is held to
    # the rounding bound of this ill-conditioned run (:func:`_seeded_bounds`).
    structural = ("z", "dz", "accepted")
    bounds = _seeded_bounds(problem, field_grid, capture_with, structural)
    for name, before in captures[0].items():
        after = captures[1][name]
        assert np.all(np.isfinite(before)) and np.all(np.isfinite(after)), name
        assert before.dtype == after.dtype and before.shape == after.shape, name
        if name in structural:
            np.testing.assert_array_equal(after, before, err_msg=name)
        elif np.any(before):
            assert_within(after, before, bounds[name], what=name)
        else:
            # An identically zero record (no stash row yet, say) stays zero.
            np.testing.assert_array_equal(after, before, err_msg=name)


def test_the_stashed_plasma_scan_is_the_synthesised_one(ionising_capillary, field_grid):
    """The channel's cheap path must be the same numbers, to the scan's rounding.

    Between planes the plasma statistics are read off what the accepted step's last
    stage left behind, because re-synthesising every node into a buffer of megabytes on
    every step costs a seventh of the run. The two routes run the same compiled bodies
    over the same nodes, from one state synthesised twice -- at the stage's own
    position and at the driver's -- so they differ by one synthesis's rounding carried
    through the rate (:func:`matched.stash_bound`).
    """
    from solprop.nonlinear.optmat_rates import OptmatRate  # noqa: PLC0415
    from solprop.physdata.gases import Gas  # noqa: PLC0415

    problem, modes = ionising_capillary
    problem = _plasma_channel(problem)
    backend = NumbaCpuBackend(problem, grid=field_grid, modes=modes)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(2e-3)
        state = backend._require_state()
        plasma = problem.responses.plasma_coefficients
        assert backend._stash_current
        stashed = [np.copy(one) for one in backend._ionised_nodes(state[0], plasma)]
        backend._stash_current = False
        fresh = backend._ionised_nodes(state[0], plasma)
    finally:
        backend.close()
    rate = OptmatRate(Gas.Ar, 800e-9, model="ppt_ilkov")
    bound = stash_bound(rate, float(np.max(fresh[1])), (1e9, 2e11), field_grid.n_to)
    for name, mine, theirs in zip(("fraction", "rate"), stashed, fresh, strict=True):
        assert_within(mine, theirs, bound, what=name)


def test_a_plane_scans_each_member_once(ionising_capillary, field_grid, monkeypatch):
    """Every plasma statistic of a plane reads one scan, not one scan per statistic.

    A default capillary spec asks for eight plasma statistics, and each used to scan
    every node again. Now the species statistics, the totals and the integration
    diagnostics share the member's scan: one under the band rule, whose statistics
    column is the diagnostics' own, and two under the trapezoid (its own column and the
    diagnostics' band column). The values are the per-statistic ones exactly, because
    the shared scan is the same compiled column on the same nodes (a structural
    identity, not a rounding comparison).
    """
    from solprop.stats import PLASMA_INTEGRATION_STATISTICS  # noqa: PLC0415

    problem, modes = ionising_capillary
    names = (
        "electron_density",
        "peak_ionisation_rate",
        "species_ionisation",
        "species_electron_density",
        "species_peak_ionisation_rate",
        "species_density",
        "species_ionisation_energy",
        *PLASMA_INTEGRATION_STATISTICS,
    )
    backend = NumbaCpuBackend(problem, grid=field_grid, modes=modes)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(2e-3)
        backend._stash_current = False
        state = backend._require_state()
        scans = []
        scan = type(backend)._plasma_stats

        def counted(self, *args, **kwargs):
            scans.append(1)
            return scan(self, *args, **kwargs)

        monkeypatch.setattr(type(backend), "_plasma_stats", counted)
        shared = backend.stats(StatsSpec(names=names))
        assert len(scans) == (1 if backend._spec.band else 2)
        density, rate = backend._ionisation(state[0])
        species = backend._species_statistics(state[0])
    finally:
        backend.close()
    np.testing.assert_array_equal(shared["electron_density"][0], density)
    np.testing.assert_array_equal(shared["peak_ionisation_rate"][0], rate)
    for name in names[2:7]:
        np.testing.assert_array_equal(shared[name][0], species[name], err_msg=name)


def test_a_current_stash_leaves_the_diagnostics_scan_unshared(
    ionising_capillary, field_grid, monkeypatch
):
    """With the stash current the species statistics read it, not the diagnostics' scan.

    The integration diagnostics still scan their band column, but leave nothing for
    the species statistics, which read the stash the accepted step left behind: one
    scan in all, and the shared values are the stash's exactly.
    """
    from solprop.stats import PLASMA_INTEGRATION_STATISTICS  # noqa: PLC0415

    problem, modes = ionising_capillary
    names = ("electron_density", "species_ionisation", *PLASMA_INTEGRATION_STATISTICS)
    backend = NumbaCpuBackend(_plasma_channel(problem), grid=field_grid, modes=modes)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(2e-3)
        state = backend._require_state()
        assert backend._spec.band and backend._stash_is_current(state[0])
        scans = []
        scan = type(backend)._plasma_stats

        def counted(self, *args, **kwargs):
            scans.append(1)
            return scan(self, *args, **kwargs)

        monkeypatch.setattr(type(backend), "_plasma_stats", counted)
        shared = backend.stats(StatsSpec(names=names))
        assert len(scans) == 1
        species = backend._species_statistics(state[0])
        assert len(scans) == 1
    finally:
        backend.close()
    np.testing.assert_array_equal(
        shared["electron_density"][0], species["electron_density"]
    )
    np.testing.assert_array_equal(
        shared["species_ionisation"][0], species["species_ionisation"]
    )


def test_a_run_without_the_channel_does_not_compile_the_stash(
    ionising_capillary, field_grid
):
    """A closure constant, so a run that did not ask pays nothing and stores nothing."""
    problem, modes = ionising_capillary
    plain = NumbaCpuBackend(problem, grid=field_grid, modes=modes)
    watched = NumbaCpuBackend(_plasma_channel(problem), grid=field_grid, modes=modes)
    try:
        assert not plain._spec.stash
        assert watched._spec.stash
        assert plain._stash.shape == (1, 2)
        assert watched._stash.shape == (
            problem.transform.n_nodes,
            1 + 2 * problem.flags.n_channels,
        )
        # Two specialisations, so neither pays the other's branch.
        assert plain._spec != watched._spec
    finally:
        plain.close()
        watched.close()


def test_the_stash_is_not_read_where_it_would_not_describe_the_state(
    ionising_capillary, field_grid
):
    """A fresh load breaks the identity, and so does a field that is not the state."""
    problem, modes = ionising_capillary
    problem = _plasma_channel(problem)
    backend = NumbaCpuBackend(problem, grid=field_grid, modes=modes)
    try:
        backend.set_initial(problem.initial.field0, None)
        # Before a step, no stage has run: the entrance plane takes the fresh scan.
        assert not backend._stash_current
        backend.propagate_to(2e-3)
        state = backend._require_state()
        assert backend._stash_is_current(state[0])
        # Another plane's field is not this state, whatever the flag says.
        assert not backend._stash_is_current(np.array(state[0]))
        backend.set_initial(problem.initial.field0, None)
        assert not backend._stash_current
    finally:
        backend.close()


def test_the_two_propagator_modes_agree(soliton_problem, soliton_grid, substitute):
    """Tabulating and recomputing give the same numbers, to the eight-step budget.

    Both are asked for by name: this backend's ``"auto"`` resolves to ``recompute``,
    so the problem's own default would exercise one of the two twice. They are one
    method by two routes on one fixed schedule, so the eight-step rung's budget
    holds (``8 x 16 eps``).
    """
    problem = soliton_problem(
        soliton_grid,
        3,
        save_planes=[8e-3],
        schedule=FixedSchedule.uniform(0.0, 8e-3, 1e-3),
    )
    both = {
        mode: substitute(
            problem,
            stepper=dataclasses.replace(problem.stepper, propagator_mode=mode),
        )
        for mode in ("tabulate", "recompute")
    }
    tabulated, _ = run(NumbaCpuBackend, both["tabulate"], soliton_grid, [8e-3])
    fresh, _ = run(NumbaCpuBackend, both["recompute"], soliton_grid, [8e-3])
    assert_within(tabulated[0], fresh[0], 8 * 16 * EPS, what="propagator modes")


def test_a_resume_continues_where_the_run_stopped(
    equivalence_backend_class,
    equivalence_backend_name,
    soliton_problem,
    soliton_grid,
):
    """``set_initial(z=)`` on a replayed schedule, to round-off against a whole run."""
    from solprop.stepper import FixedSchedule  # noqa: PLC0415 - the replay needs one

    schedule = FixedSchedule.uniform(0.0, 0.02, 1e-3)
    problem = soliton_problem(
        soliton_grid, 3, save_planes=[0.01, 0.02], schedule=schedule
    )
    whole, _ = run(equivalence_backend_class, problem, soliton_grid, [0.01, 0.02])

    backend = equivalence_backend_class(problem, grid=soliton_grid)
    try:
        backend.set_initial(whole[0], None, z=0.01)
        backend.propagate_to(0.02)
        resumed = backend.observe().require_field().copy()
    finally:
        backend.close()
    # A load clears the recycled stage, which both backends keep across a plane, so
    # the resume evaluates the first derivative afresh after the plane. Thirty-two
    # epsilon covers that one additional FFT round-off.
    assert equivalence_backend_name in ("numba_cpu", "native")
    assert disagreement(resumed, whole[1]) < 32 * EPS


@pytest.mark.parametrize("backend_class", [NumpyRefBackend, NumbaCpuBackend])
def test_the_observation_hook_is_the_identity_for_existing_backends(
    backend_class, soliton_problem, narrow_grid
):
    """The new device-state hook cannot copy or perturb either existing backend."""
    problem = soliton_problem(narrow_grid, 1, save_planes=[0.01])
    backend = backend_class(problem, grid=narrow_grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        state = backend._require_state()
        assert backend._observed_field(state) is state
        assert np.array_equal(backend.observe().require_field(), state)
    finally:
        backend.close()


def test_the_ledger_names_every_buffer_and_stays_inside_the_budget(
    capillary_problem, field_grid
):
    """Every buffer, including the plan work areas, and no more than the budget allows.

    ``architecture.md`` section 8 sizes the solver registers at 9 to 14 field-sized
    units and the node buffers by the chunk rather than by the node count. The point of
    naming the plan buffers is that FFTW's are real memory that the reference, which
    allocates per pass, never has to declare.
    """
    problem = capillary_problem()
    backend = NumbaCpuBackend(problem, grid=field_grid)
    try:
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    # Released with everything else, and still named: a closed backend owes the budget
    # a row of zero, not a missing row.
    assert backend.memory_ledger()["statistics_power_weight"] == 0
    reference = NumpyRefBackend(problem, grid=field_grid).memory_ledger()
    for name in (
        "state",
        "stages",
        "accumulator_solution",
        "accumulator_error",
        "propagators",
        "fine_spectrum",
        "modal_time",
        "projected",
        "node_field",
        "node_polarisation",
        "statistics_power_weight",
    ):
        assert name in ledger, name
    assert all(value >= 0 for value in ledger.values())
    # One float64 weight per column and none per batch member: the members of one
    # problem share the fibre, so they share `Re n_eff` (kernel-contract.md §5).
    assert ledger["statistics_power_weight"] == int(np.prod(problem.shape[1:])) * 8
    assert "node_raman_drive" in ledger or not problem.flags.has_raman

    # The solver registers are the reference's, plus the one buffer a preallocating
    # backend needs that a per-pass one does not: the state a trial step writes before
    # it is known whether the step will be kept.
    registers = (
        "state",
        "stages",
        "accumulator_solution",
        "accumulator_error",
    )
    for name in registers:
        assert ledger[name] == reference[name], name
    # The propagators are the one row where the two differ, because the two resolve
    # the contract's `"auto"` differently: the reference keeps the six distinct
    # exponents of a step and this backend recomputes into one scratch pair, which is
    # a sixth of the memory for the same numbers.
    distinct = np.unique(np.append(problem.stepper.tableau.nodes, 1.0)).size
    assert ledger["propagators"] * distinct == reference["propagators"]
    assert ledger["advanced"] == ledger["state"]
    # The node buffers are sized by the chunk and not by the node count, which is the
    # whole point of the chunk loop; this problem is mode-averaged, so the chunk is one
    # node and the buffers are one column.
    assert ledger["node_field"] == reference["node_field"]
    # And the plan work areas, which the reference never has to declare because it
    # allocates a fresh array per pass.
    assert ledger["projected_spectrum"] > 0


def test_the_metadata_records_what_the_run_actually_did(soliton_problem, soliton_grid):
    """The pools were measured to be worth a factor of nine, so a run records them."""
    problem = soliton_problem(soliton_grid, 1, save_planes=[0.01])
    backend = NumbaCpuBackend(
        problem, grid=soliton_grid, threads=2, plan_flag="estimate"
    )
    try:
        metadata = backend.metadata()
    finally:
        backend.close()
    assert metadata["backend"] == "numba_cpu"
    assert metadata["column_family"] == "envelope"
    assert metadata["kernel_regime_elements"] == numba_fft.REGIME_ELEMENTS
    assert metadata["plan_flag"] == "estimate"
    assert metadata["numba_threads"] == min(2, os.cpu_count() or 1)
    # The problem asked for `"auto"`; what a run records is what was resolved.
    assert problem.stepper.propagator_mode == "auto"
    assert metadata["propagator_mode"] == "recompute"
    assert isinstance(metadata["wisdom_digest"], str)
    assert metadata["fftw_wait_policy"] == openmp.WAIT_POLICY
    assert metadata["blas_mechanism"] in {"uncontrolled", "openblas", "mkl", "blis"}


def test_a_closed_or_unloaded_backend_refuses(soliton_problem, soliton_grid):
    """The seam's errors, which a run's own error handling relies on."""
    problem = soliton_problem(soliton_grid, 1, save_planes=[0.01])
    backend = NumbaCpuBackend(problem, grid=soliton_grid)
    with pytest.raises(BackendError, match="no initial state"):
        backend.propagate_to(0.01)
    backend.set_initial(problem.initial.field0, None)
    backend.close()
    backend.close()
    with pytest.raises(BackendError, match="closed"):
        backend.observe()


def test_the_registry_builds_it_by_name(soliton_problem, soliton_grid):
    """``select`` is how a run reaches a backend, and it reaches this one."""
    problem = soliton_problem(soliton_grid, 1, save_planes=[0.01])
    backend = select("numba_cpu", problem, grid=soliton_grid)
    try:
        assert isinstance(backend, NumbaCpuBackend)
        assert backend.name == "numba_cpu"
        assert backend.max_batch == 1
    finally:
        backend.close()


def test_a_batched_problem_is_refused(batched_problem):
    """One problem at a time, said by name rather than by a shape error later."""
    with pytest.raises(BackendError, match="max_batch"):
        NumbaCpuBackend(batched_problem)


# --- the statistics -------------------------------------------------------------


@pytest.fixture
def ionising_capillary(field_grid):
    """A gradient capillary of argon, ionising, for the plasma statistics."""
    import warnings  # noqa: PLC0415 - argon is outside its index model's range

    from solprop.exceptions import MaterialRangeWarning  # noqa: PLC0415
    from solprop.modes.capillary import MarcatiliMode  # noqa: PLC0415
    from solprop.nonlinear.optmat_rates import OptmatRate  # noqa: PLC0415
    from solprop.nonlinear.responses import (  # noqa: PLC0415
        KerrResponse,
        PlasmaResponse,
    )
    from solprop.physdata.gases import Gas  # noqa: PLC0415
    from solprop.profiles import PressureGradient  # noqa: PLC0415
    from solprop.pulses import GaussianPulse  # noqa: PLC0415
    from solprop.stepper import FixedSchedule  # noqa: PLC0415
    from solprop.stepper.controller import PIController  # noqa: PLC0415
    from solprop.translate.ionisation import build_rate_table  # noqa: PLC0415
    from solprop.translate.problem import assemble  # noqa: PLC0415

    mode = MarcatiliMode(
        125e-6, "Ar", PressureGradient.two_point(0.2, 2.0, 0.2), cladding=1.45
    )
    table = build_rate_table(OptmatRate(Gas.Ar, 800e-9, model="ppt_ilkov"), (1e9, 2e11))
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=200e-6).build(
        field_grid
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem = assemble(
            field_grid,
            mode,
            [
                KerrResponse(chi3=2.2e-52, thg=True),
                PlasmaResponse(Gas.Ar, table, preionisation=1e-3),
            ],
            start,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[2e-3],
            schedule=FixedSchedule.uniform(0.0, 2e-3, 1e-3),
            dz_max=0.01,
            dz_init=1e-4,
        )
    return problem, [mode]


def test_every_statistic_matches_the_reference(ionising_capillary, field_grid):
    """The four the backend owns and the ones it delegates, at the same plane.

    The four are computed in the compiled passes rather than through the reference's
    per-plane re-synthesis and multipass scan, which was measured at two thirds of a
    whole CtC step on an x86-64 server. What must not change is the number, so the whole
    spec is taken on both backends after the same propagation.

    **The bound, stated first.** This run is strongly ionising and ill-conditioned: a
    launch moved by one ulp in random directions ends up to about 5e5 ulp away in the
    field after its two steps (measured on an Apple-silicon laptop and on an x86-64
    server alike). Two backends differ by rounding at every stage -- transforms, the
    rate's `exp`/`log`, the reductions -- and each such seed is carried the same way, so
    each statistic is bounded by its own measured response to a one-ulp launch
    perturbation times the count of seeds: two steps of seven stages, each at most a
    transform's `7 log2(n_to)` ulp; to which each backend's own evaluation of the
    statistic adds at most `gamma(n_to)` (the density's running sum, which a launch ulp
    does not move at all). The response is the largest of three random-sign
    perturbations, a launch seed having the whole run to grow. A fixed 1e-10 held here
    by luck: on an x86-64 server, where numpy's own SIMD `exp`/`log` and FFTW's planner
    part the two backends more than one libm does on an Apple-silicon laptop, the peak
    rate missed it at 1.08e-10, with this bound about 3e-8.
    """
    problem, modes = ionising_capillary
    names = (
        "energy",
        "peak_power",
        "electron_density",
        "peak_ionisation_rate",
        "peak_intensity",
    )
    spec = StatsSpec(names=(*names, "quadrature_error"))

    def statistics(backend_class, launch):
        backend = backend_class(problem, grid=field_grid, modes=modes)
        try:
            backend.set_initial(launch, None)
            backend.propagate_to(2e-3)
            return backend.stats(spec)
        finally:
            backend.close()

    launch = np.asarray(problem.initial.field0)
    ours = statistics(NumbaCpuBackend, launch)
    theirs = statistics(NumpyRefBackend, launch)
    assert set(ours) == set(theirs)
    unit = np.finfo(float).eps / 2
    generator = np.random.default_rng(7)
    response = dict.fromkeys(names, 0.0)
    for _ in range(3):
        signs = generator.choice([-1.0, 1.0], size=launch.shape)
        moved = statistics(NumpyRefBackend, launch * (1 + unit * signs))
        for name in names:
            change = np.max(np.abs(moved[name] / theirs[name] - 1))
            response[name] = max(response[name], float(change))
    seeds = 2 * DOPRI5.n_stages * 7 * math.log2(field_grid.n_to)
    # Each backend also evaluates the statistic itself: the density is a running sum
    # of positive terms over the fine time axis, within `gamma(n_to)` for either order.
    scan = 2 * field_grid.n_to * unit / (1 - 2 * field_grid.n_to * unit)
    # Three random-sign perturbations sample the response; they do not bound it, and a
    # statistic the launch's last bit does not move has no sampled term at all. On an
    # x86-64 cluster node (GCC 16 toolchain, FFTW 3.3.11) the two backends' electron
    # density parted by 5.3e-13 against a sampled bound of 2.2e-13 on an unchanged tree:
    # their own libm and SIMD differences, which no launch perturbation samples. The
    # sampled term carries a factor of four, under a floor of four times that value.
    sampling = 4.0
    floor = 4 * 5.4e-13
    for name in names:
        # Energy and power are delegated to solprop.stats and the rest computed in the
        # compiled passes; either way the only difference is the propagated state.
        assert np.all(np.asarray(ours[name]) > 0.0)
        assert ours[name] == pytest.approx(
            theirs[name], rel=max(sampling * seeds * response[name] + scan, floor)
        )
    # A mode-averaged run's rule has no embedded pair, so the difference of the two
    # projections is identically zero; both backends report that as nothing rather
    # than as convergence.
    assert math.isnan(float(ours["quadrature_error"][0]))
    assert math.isnan(float(theirs["quadrature_error"][0]))


def test_the_statistics_refuse_what_the_run_does_not_carry(
    soliton_problem, soliton_grid
):
    """A plasma statistic without a plasma, and an axis statistic without the modes."""
    from solprop.stats import StatsSpec  # noqa: PLC0415 - as above

    problem = soliton_problem(soliton_grid, 1, save_planes=[0.01])
    backend = NumbaCpuBackend(problem, grid=soliton_grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        with pytest.raises(BackendError, match="compiled plasma response"):
            backend.stats(StatsSpec(names=("electron_density",)))
        with pytest.raises(BackendError, match="needs the mode set"):
            backend.stats(StatsSpec(names=("peak_intensity",)))
    finally:
        backend.close()
    without_grid = NumbaCpuBackend(problem)
    try:
        without_grid.set_initial(problem.initial.field0, None)
        with pytest.raises(BackendError, match="'numba_cpu'"):
            without_grid.stats(StatsSpec(names=("energy",)))
    finally:
        without_grid.close()


def test_the_plasma_statistics_work_on_the_analytic_family(
    spm_field_problem, field_grid
):
    """The statistics path takes the same analytic route the stage loop does.

    ``_nodes_at`` re-synthesises every node at once, which is the one place outside the
    stage loop that has to know which family the column is; a field grid working on the
    analytic signal needs its positive bins doubled there too.
    """
    from solprop.stats import StatsSpec  # noqa: PLC0415 - one test builds a spec

    problem = spm_field_problem()
    backend = NumbaCpuBackend(problem, grid=field_grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(0.01)
        nodes = backend._nodes_at(backend.observe().require_field()[0])
        assert np.iscomplexobj(nodes)
        # The real part is the field the plasma would read, and it is not nothing.
        assert float(np.max(np.abs(np.real(nodes)))) > 0.0
        assert backend.stats(StatsSpec(names=("energy",)))["energy"].shape[0] == 1
    finally:
        backend.close()


def test_a_thread_count_from_the_environment_is_read_and_checked(
    soliton_problem, soliton_grid, monkeypatch
):
    """The registry constructs with the seam's arguments alone, so a run sets a pool
    through the environment; a value that is not a count says so rather than being
    silently ignored."""
    problem = soliton_problem(soliton_grid, 1, save_planes=[0.01])
    monkeypatch.setenv("SOLPROP_NUMBA_THREADS", "1")
    backend = NumbaCpuBackend(problem, grid=soliton_grid)
    try:
        assert backend.metadata()["numba_threads"] == 1
    finally:
        backend.close()
    monkeypatch.setenv("SOLPROP_NUMBA_THREADS", "  ")
    NumbaCpuBackend(problem, grid=soliton_grid).close()
    monkeypatch.setenv("SOLPROP_NUMBA_THREADS", "as many as you like")
    with pytest.raises(BackendError, match="SOLPROP_NUMBA_THREADS"):
        NumbaCpuBackend(problem, grid=soliton_grid)


def test_a_harmonic_term_without_its_carrier_is_refused(thg_problem):
    """The envelope's harmonic sits an offset from its carrier and needs the factor.

    ``validate`` refuses the pairing before a backend sees it, which is why the helper
    is called directly: the guard exists for a hand-built problem, and the reference
    tests its own the same way.
    """
    import dataclasses  # noqa: PLC0415 - only this test empties a group

    from solprop.backends.numba_cpu import backend as module  # noqa: PLC0415

    problem = thg_problem(thg=True)
    without = dataclasses.replace(
        problem, grids=dataclasses.replace(problem.grids, carrier_2w0=None)
    )
    with pytest.raises(BackendError, match="carrier_2w0 is absent"):
        module._harmonic_carrier(without)
    assert module._harmonic_carrier(thg_problem(thg=False)).shape == (
        problem.grids.n_to,
    )


def test_the_in_stage_absorber_term_is_planned_and_allocates_nothing(
    capillary_problem, field_grid
):
    """The temporal collar's term through the backend's own plan, against the reference.

    Read off the same fine samples the last stage left in the plan's buffer: the
    samples are one product, ``(-0.5 rate) x``, on either side, so what differs is
    chiefly FFTW's summation order against ``scipy.fft``'s.
    """
    problem = capillary_problem()
    backend = NumbaCpuBackend(problem, grid=field_grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend._trial_step(0.0, 1e-4)
        measured = backend._absorber_term().copy()
        samples = backend._plans.modal_time.copy()
        tracemalloc.start()
        backend._absorber_term()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        backend.close()
    rate = problem.responses.temporal_absorber_fine
    assert rate is not None
    n_columns, n_to = samples.shape
    expected = passes.absorber_term(
        samples.reshape(n_columns, 1, n_to),
        rate,
        field_grid.crop_index,
        field_grid.active_mask,
        kind="field",
    ).reshape(measured.shape)
    # Two transforms of identical samples: each bin is within about log2(n_to)
    # roundings of the absolute sum of its samples, twice over.
    absolute = n_to * float(np.max(np.abs(0.5 * rate * samples)))
    bound = 2 * 4 * int(np.log2(n_to)) * EPS * absolute
    assert bound > 0
    np.testing.assert_allclose(measured, expected, rtol=0, atol=bound)
    # One column of the fine spectrum is the smallest buffer it could allocate.
    assert peak < n_to * 16


# ------------------------------------------------- the recycled first-same-as-last


def test_the_recycled_stage_is_the_one_it_replaces(capillary_problem, field_grid):
    """The recycled-stage identity, on a case with both collars and a varying operator.

    ``k1^{n+1} = exp(dPhi^n) k7^n``: the tableau's last row is its solution weights, so
    stage seven's *input* is the accept pass's own result and its derivative is stage
    one of the next step, rebased for the re-anchored interaction picture. Measured
    here against the stage the backend would otherwise have evaluated.

    The identity measures 9.9e-17 on the GNLSE soliton, whose operator has no loss
    and no collar. This case has both, and it agrees at the same order -- but only
    because the in-stage absorber term is masked to the active band. Without that
    mask the stage writes outside the band the equation is solved on, the accept pass's
    hard mask removes it, and the identity fails at 7.4e-7.
    """
    problem = capillary_problem()
    backend = NumbaCpuBackend(problem, grid=field_grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(0.02)
        assert backend._has_recycled
        recycled = backend._recycled.copy()
        # Evaluate the same stage from scratch, at the same state and step.
        backend._has_recycled = False
        backend._trial_step(backend._z, float(backend._history[-1][1]))
        fresh = backend._stages[0].copy()
        gap = float(np.linalg.norm(recycled - fresh) / np.linalg.norm(fresh))
        # The derivative's sensitivity to an elementwise relative perturbation of the
        # state, of the size and kind rounding makes: a measured conditioning estimate,
        # not an operation count, taken before the bound is set and independent of the
        # gap it bounds, so the bound can still fail.
        state = backend._require_state()
        rng = np.random.default_rng(7)
        delta = 1e-11
        state *= 1.0 + delta * np.exp(2j * np.pi * rng.random(state.shape))
        backend._has_recycled = False
        backend._trial_step(backend._z, float(backend._history[-1][1]))
        moved = backend._stages[0]
        sensitivity = float(np.linalg.norm(moved - fresh) / np.linalg.norm(fresh))
        sensitivity /= delta
    finally:
        backend.close()
    # Rounding, not the identity, is all that separates the two (SEP 25 section 4.3):
    # the stage-seven input and the accepted state are the same sum, formed by two
    # kernels whose contracted roundings differ, each within gamma_20 of the state
    # (seven weighted stages, the propagator's complex product); that departure is
    # carried by the derivative's sensitivity, plus 32 eps for the exp(+dPhi) that
    # undoes the stage's exp(-dPhi).
    bound = 2 * higham_gamma(20) * sensitivity + 32 * np.finfo(float).eps
    assert gap < bound, (gap, sensitivity)


def test_the_saving_does_not_change_the_answer(
    capillary_problem, field_grid, substitute
):
    """Six evaluations in seven, and the same numbers to the reduction's own round-off.

    9.2 to 9.5 % of a step -- four modes and one, on a replayed schedule so
    that the step count cannot move; the two answers over 400 fixed steps
    agree to 2.9e-16 and 5.8e-16 of the peak.
    """
    problem = capillary_problem()
    # A replayed schedule, so that what is compared is the step and not the trajectory:
    # the recycled stage differs from the one it replaces by a few ulps, and an
    # adaptive controller turns a few ulps into a different number of steps.
    fixed = substitute(
        problem,
        stepper=dataclasses.replace(
            problem.stepper,
            schedule=FixedSchedule.uniform(0.0, 0.2, 0.0005, save_planes=[0.1, 0.2]),
        ),
    )

    class WithoutTheSaving(NumbaCpuBackend):
        """The same backend with the recycled stage refused, for the comparison."""

        @property
        def _fsal(self) -> bool:
            return False

    def run(*, recycle):
        chosen = NumbaCpuBackend if recycle else WithoutTheSaving
        backend = chosen(fixed, grid=field_grid)
        try:
            backend.set_initial(fixed.initial.field0, None)
            summary = backend.propagate_to(0.1)
            return backend.observe().require_field().copy(), summary.n_steps
        finally:
            backend.close()

    with_it, taken = run(recycle=True)
    without, alone = run(recycle=False)
    assert taken == alone
    peak = float(np.max(np.abs(without)))
    assert float(np.max(np.abs(with_it - without))) / peak < 1e-14


def test_the_split_step_absorber_gives_the_recycled_stage_up(
    capillary_problem, field_grid
):
    """It changes the state after the last stage, which is what the identity forbids.

    The other absorber scheme multiplies the accepted state by ``exp(-alpha(t) h / 2)``
    after every stage has been evaluated, so stage seven's derivative belongs to a
    state the run no longer holds. That is why the in-stage term is what makes the
    saving available at all.
    """
    inside = NumbaCpuBackend(capillary_problem(), grid=field_grid)
    split = NumbaCpuBackend(capillary_problem(scheme="split_step"), grid=field_grid)
    try:
        assert inside._fsal
        assert not split._fsal
        split.set_initial(split._problem.initial.field0, None)
        split.propagate_to(0.02)
        assert not split._has_recycled
        # The statistics channel's cheap plasma path rests on the same identity, so
        # this scheme gives that up too.
        assert not split._stash_current
    finally:
        inside.close()
        split.close()


def test_a_plane_boundary_keeps_the_recycled_stage(capillary_problem, field_grid):
    """The state a plane observes is the one the run adopted, so the stage holds.

    Kept across planes since SEP 14 phase 3 (ADR-039), saving one evaluation per plane;
    a resume reloads the state with no stage behind it and agrees to round-off.
    """
    backend = NumbaCpuBackend(capillary_problem(), grid=field_grid)
    try:
        backend.set_initial(backend._problem.initial.field0, None)
        backend.propagate_to(0.02)
        assert backend._has_recycled
        backend.propagate_to(0.04)
        # The hook fires at the top of the call, so what is held now is this plane's.
        assert backend._has_recycled
        backend._begin_plane()
        assert backend._has_recycled
    finally:
        backend.close()
