# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The compiled backend against the reference: the validation ladder.

The ladder of ``docs/dev/backends.md`` section 3.7 -- one linear application, one step,
eight steps, a full run, a capillary with an absorber and a kink, the analytic field
family -- and repeatability and a resume on the same schedules. Each rung's measurement
is recorded with its bound in the ``backend_equivalence`` study
(``docs/dev/accuracy.md`` section 7). Rows of the retired ``numba_cpu`` backend, written
when these tests also ran on it, stay in that record as evidence of what it measured.

**The bounds are derived, not fitted.** The reference and a compiled backend perform the
same arithmetic in the same order wherever they can, and where they cannot the
difference has a measured cause: a contracted fused multiply-add on one side and not the
other (1.42 eps per product), a modulus formed as ``np.abs(z) ** 2`` against summed
squares (1.4 eps), and two FFT libraries differing by ``log2(n) eps`` over a transform.
A step is a few of each, so a step agrees to a few eps and a run of ``n`` steps to a few
``n`` eps at worst -- and, because the two trajectories are the same to that accuracy,
the controller takes the same steps and the errors do not compound the way an
independent integration's would. Every bound below is stated with the number measured
against it.
"""

import dataclasses

import numpy as np
import pytest

from bounds import (
    AccuracyEnvelope,
    analytic_bound,
    assert_within,
    assert_work_within,
    ladder_bound,
    record_backend,
)
from convergence_records import record
from convergence_studies import BACKEND_EQUIVALENCE
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.stepper import FixedSchedule, PIController

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


def run(  # noqa: PLR0913 - the run and its context
    backend_class,
    problem,
    run_request,
    grid,
    planes,
    *,
    modes=None,
    noise=None,
    **options,
):
    """Propagate a problem to every plane and return the fields and the summaries."""
    backend = backend_class(problem, run_request, grid=grid, modes=modes, **options)
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

    Not bitwise, though the arithmetic is only two complex products per step: the two
    sides may contract them into fused multiply-adds differently. Bound 4 eps per step
    over 8 steps, so 32 eps; measured 1.1e-16 on the retired ``numba_cpu``, which is
    under one, and 3.1e-22 on native.
    """
    problem, run_request = linear_problem(
        narrow_grid, save_planes=[0.05], responses=[zero_kerr]
    )
    ours, _ = run(equivalence_backend_class, problem, run_request, narrow_grid, [0.05])
    theirs, _ = run(NumpyRefBackend, problem, run_request, narrow_grid, [0.05])
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
    problem, run_request = soliton_problem(
        soliton_grid,
        3,
        save_planes=[1e-3],
        schedule=FixedSchedule.uniform(0.0, 1e-3, 1e-3),
    )
    ours, _ = run(equivalence_backend_class, problem, run_request, soliton_grid, [1e-3])
    theirs, _ = run(NumpyRefBackend, problem, run_request, soliton_grid, [1e-3])
    # One step is seven stages of a few complex products and two transforms each;
    # 16 eps by Higham on log2(8192), measured 2.4e-16 on the retired numba_cpu.
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
    problem, run_request = soliton_problem(
        soliton_grid,
        3,
        save_planes=[8e-3],
        schedule=FixedSchedule.uniform(0.0, 8e-3, 1e-3),
    )
    ours, summaries = run(
        equivalence_backend_class, problem, run_request, soliton_grid, [8e-3]
    )
    theirs, reference = run(NumpyRefBackend, problem, run_request, soliton_grid, [8e-3])
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
    problem, run_request = soliton_problem(soliton_grid, 3, save_planes=planes)
    tight, tight_request = soliton_problem(
        soliton_grid,
        3,
        save_planes=planes,
        controller=PIController(rtol=1e-13, atol=1e-20),
    )
    converged, _ = run(NumpyRefBackend, tight, tight_request, soliton_grid, planes)
    theirs, reference = run(NumpyRefBackend, problem, run_request, soliton_grid, planes)
    envelope = AccuracyEnvelope(
        reference="numpy_ref at rtol 1e-13, N = 3 soliton, both planes, relative L2",
        error=max(
            disagreement(one, best) for one, best in zip(theirs, converged, strict=True)
        ),
        factor=4.0,
    )
    assert envelope.error > 0
    ours, summaries = run(
        equivalence_backend_class, problem, run_request, soliton_grid, planes
    )
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
    problem, run_request = capillary_problem()
    fixed = problem
    run_request = run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper,
            # Below the collar's own stability limit of 1.1 mm (ADR-025).
            schedule=FixedSchedule.uniform(0.0, 0.2, 0.0005, save_planes=[0.1, 0.2]),
        )
    )
    ours, summaries = run(
        equivalence_backend_class, fixed, run_request, field_grid, [0.1, 0.2]
    )
    theirs, reference = run(NumpyRefBackend, fixed, run_request, field_grid, [0.1, 0.2])
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
    once per node chunk; a compiled backend gets it from the inverse transform it was
    already running, by doubling the positive bins. Same operator, different route, so
    the bound is the transform's rather than the arithmetic's.
    """
    problem, run_request = spm_field_problem()
    ours, _ = run(equivalence_backend_class, problem, run_request, field_grid, [0.05])
    theirs, _ = run(NumpyRefBackend, problem, run_request, field_grid, [0.05])
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


# --- repeatability and a resume --------------------------------------------------


def test_two_runs_agree(equivalence_backend_class, soliton_problem, soliton_grid):
    """Repeatability: two runs of the eight-step rung on its fixed schedule.

    The same discrete computation twice, so the rung's own budget holds (``8 x 16
    eps``, :func:`test_eight_steps_of_the_soliton_agree`) and nothing more is promised:
    a threaded reduction may order its sums differently from one run to the next.
    """
    problem, run_request = soliton_problem(
        soliton_grid,
        3,
        save_planes=[8e-3],
        schedule=FixedSchedule.uniform(0.0, 8e-3, 1e-3),
    )
    first, _ = run(
        equivalence_backend_class, problem, run_request, soliton_grid, [8e-3]
    )
    second, _ = run(
        equivalence_backend_class, problem, run_request, soliton_grid, [8e-3]
    )
    assert_within(second[0], first[0], 8 * 16 * EPS, what="repeat run")


def test_a_resume_continues_where_the_run_stopped(
    equivalence_backend_class,
    soliton_problem,
    soliton_grid,
):
    """``set_initial(z=)`` on a replayed schedule, to round-off against a whole run."""
    schedule = FixedSchedule.uniform(0.0, 0.02, 1e-3)
    problem, run_request = soliton_problem(
        soliton_grid, 3, save_planes=[0.01, 0.02], schedule=schedule
    )
    whole, _ = run(
        equivalence_backend_class, problem, run_request, soliton_grid, [0.01, 0.02]
    )

    backend = equivalence_backend_class(problem, run_request, grid=soliton_grid)
    try:
        backend.set_initial(whole[0], None, z=0.01)
        backend.propagate_to(0.02)
        resumed = backend.observe().require_field().copy()
    finally:
        backend.close()
    # A load clears a recycled stage, which a backend may keep across a plane, so the
    # resume evaluates the first derivative afresh after the plane. Thirty-two epsilon
    # covers that one additional FFT round-off.
    assert disagreement(resumed, whole[1]) < 32 * EPS


def test_the_observation_hook_is_the_identity_on_the_reference(
    soliton_problem, narrow_grid
):
    """The device-state hook cannot copy or perturb the reference's state."""
    problem, run_request = soliton_problem(narrow_grid, 1, save_planes=[0.01])
    backend = NumpyRefBackend(problem, run_request, grid=narrow_grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        state = backend._require_state()
        assert backend._observed_field(state) is state
        assert np.array_equal(backend.observe().require_field(), state[0])
    finally:
        backend.close()
