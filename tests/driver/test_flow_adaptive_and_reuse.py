# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A flow under the adaptive controller, and a flow backend reused after a failure.

Two cases the fixed-schedule flow tests do not reach:

* **A production adaptive run with rejections beside a flow**: every other flow test
  replays a fixed schedule. Here a varying coherent wall -- the taper of
  ``tests/flow_parity.py`` under nitrogen with ``prop_capillary``'s default plasma
  and Raman -- runs on the PI controller from a first step as long as the whole
  guide, so that the controller rejects, and agrees with the same run at a hundred
  times tighter tolerance within the controller's own error bound; the compiled
  backends run the same case adaptively and agree with the reference within the
  same kind of bound (their step sequences need not be the reference's: the embedded
  error estimate is itself rounding at these small amplitudes, so the controller's
  steps may differ in their last digits; the test records which holds).
* **A forced failure followed by reuse** on a flow: a backend refuses a step whose
  measured conditioning exceeds the declared certificate, part way through a fixed
  schedule, and the next backend built in the same process for the same problem runs it
  as one built before the failure did: the same march, within its rounding budget.

Nothing here propagates further than a fifth of a millimetre on the walls, or a
millimetre on the moving guide.
"""

import dataclasses
import math
from dataclasses import replace

import numpy as np
import pytest

from driver.test_moving_guide import GRIN_PULSE, grin_taper, kinked_radius
from driver.test_wall_beside_gas_responses import (
    entry_point,
    lipschitz_scale,
    simulation,
)
from flow_parity import LENGTH, QUIET, STEPS, build, departure, run
from march_bounds import FIXED_ORDER, assert_matched
from modal_provider_cases import grin_grid
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import available
from solprop.driver import Simulation
from solprop.exceptions import BackendError, ConvergenceError, SolpropWarning
from solprop.output.memory import MemoryOutput
from solprop.stepper import FixedSchedule
from solprop.stepper.controller import PIController
from solprop.translate.coupled import constant_schur_flow
from state_records import record_bound

pytestmark = [
    pytest.mark.filterwarnings(f"ignore::{category.__module__}.{category.__name__}")
    for category in QUIET
] + [
    pytest.mark.filterwarnings("ignore:the temporal absorber attenuates"),
]
"""The qualification cases' deliberate warnings (``flow_parity.QUIET``)."""

RTOL = 1e-8
"""The adaptive run's relative tolerance."""

TIGHT = 1e-10
"""The comparison run's, a hundred times tighter."""


def backend_class(name):
    """Return one CPU backend's class, the Kokkos one from its optional wheel."""
    if name == "native":
        # Optional wheel: absent from the base environment.
        from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            NativeBackend,
        )

        return NativeBackend
    return {"numpy_ref": NumpyRefBackend, "numba_cpu": NumbaCpuBackend}[name]


def installed(names, *marks):
    """Return backend parameters, skipped where the backend is not installed."""
    return [
        pytest.param(
            name,
            marks=[
                *marks,
                *(
                    []
                    if name in available()
                    else [pytest.mark.skip(reason=f"{name} is not installed")]
                ),
            ],
        )
        for name in names
    ]


# ------------------------------------------------------------ the adaptive run


def adaptive(backend, rtol):
    """Return the taper under nitrogen on the PI controller, from a whole-guide step.

    ``atol = 0``, so the controller's scale is ``rtol max(||y||, ||y_new||)``
    exactly (``stepper.controller.weak_norm_scale``). The first proposal is the
    guide's whole fifth of a millimetre, which the error estimate cannot accept.
    """
    # The first step exceeds the frozen split's advisory step, which says so: that
    # is the point of starting there, and the controller's error estimate carries
    # the explicit residual it is about.
    with pytest.warns(SolpropWarning, match="advisory step"):
        return entry_point(
            "taper",
            backend=backend,
            schedule=None,
            dz_init=LENGTH,
            dz_max=LENGTH,
            rtol=rtol,
            atol=0.0,
        )


def history(output: MemoryOutput) -> tuple[np.ndarray, np.ndarray]:
    """Return a run's attempted step lengths and their acceptances."""
    steps = output.planes()[-1].step_history
    return np.asarray(steps.h), np.asarray(steps.accepted)


def controller_bound(runs) -> float:
    """Return the bound on two adaptive runs' final states, relative to the first.

    **Stated before any measurement.** An accepted step's embedded estimate is below
    ``rtol max(||y||, ||y_new||)``, and the propagated higher-order solution's own
    local error is below its estimate (local extrapolation, asymptotically). Each
    local error is carried to the end by the rest of the run: through the frozen
    flow of each of the sections still ahead by at most the declared certificate
    ``C`` (declared over ``dz_max``, the longest length a section's action is
    taken over), through the explicit residual by ``exp(K L)`` with ``K`` its
    declared norm, and through the source by ``exp(sum_r k_r Phi_r)``, the growth
    scale of the default responses (``lipschitz_scale``). So each run is within
    ``G sum_k rtol ||y||_max`` of the exact solution, ``G = C^sections exp(K L +
    sum_r k_r Phi_r)``, and two runs within the sum of theirs, relative to the final
    state with ``||y||_max`` the larger of the entrance and exit norms (every
    response and the wall here remove energy, and the Kerr and Raman halves keep it).
    """
    built = simulation("taper")
    flow = built.translate().linear.coupled_flow
    assert flow is not None
    growth = float(np.max(flow.inverse_condition_bound)) ** len(
        flow.section_methods
    ) * math.exp(
        float(np.max(flow.residual_norm_bound)) * LENGTH + lipschitz_scale("taper")
    )
    total = 0.0
    for output, rtol in runs:
        planes = output.planes()
        largest = max(float(np.linalg.norm(plane.field)) for plane in planes)
        _, accepted = history(output)
        total += int(np.sum(accepted)) * rtol * largest
    final = float(np.linalg.norm(runs[0][0].planes()[-1].field))
    return growth * total / final


@pytest.fixture(scope="module")
def reference():
    """The adaptive reference run and the tight one, once for the module."""
    return adaptive("numpy_ref", RTOL), adaptive("numpy_ref", TIGHT)


@pytest.mark.medium
def test_an_adaptive_run_beside_a_flow_rejects_and_agrees_with_a_tighter_one(
    reference, record_property
):
    """At least one rejection, and the tighter run within the controller's bound."""
    loose, tight = reference
    _, accepted = history(loose)
    rejected = int(np.sum(~accepted))
    record_property("accepted", int(np.sum(accepted)))
    record_property("rejected", rejected)
    assert rejected > 0
    bound = controller_bound([(loose, RTOL), (tight, TIGHT)])
    measured = departure(loose.planes()[-1].field, tight.planes()[-1].field)
    record_property("bound", bound)
    record_property("measured", measured)
    assert bound > 0.0
    assert measured <= bound


@pytest.mark.parametrize(
    "backend", installed(["numba_cpu", "native"], pytest.mark.slow)
)
def test_the_compiled_backends_reject_and_agree_within_the_controller_bound(
    reference, backend, record_property
):
    """The same adaptive run on a compiled backend: rejections, and the bound.

    The bound is :func:`controller_bound` over the two runs: each is within its own
    tolerance of the exact solution. Whether the step sequences are the reference's
    bit for bit is recorded rather than asserted: at these small amplitudes the
    embedded error estimate is rounding, so the backends' step sizes may differ in
    their last digits.
    """
    expected, _ = reference
    got = adaptive(backend, RTOL)
    steps, accepted = history(got)
    reference_steps, reference_accepted = history(expected)
    assert int(np.sum(~accepted)) > 0
    bound = controller_bound([(expected, RTOL), (got, RTOL)])
    measured = departure(got.planes()[-1].field, expected.planes()[-1].field)
    same = steps.shape == reference_steps.shape and bool(
        np.array_equal(accepted, reference_accepted)
    )
    record_property("bound", bound)
    record_property("measured", measured)
    record_property("same_acceptances", same)
    record_property(
        "steps_bitwise", same and bool(np.array_equal(steps, reference_steps))
    )
    if same:
        record_property(
            "largest_step_departure",
            float(np.max(abs(steps / reference_steps - 1.0))),
        )
    assert measured <= bound


# ------------------------------------------------------ failure, then reuse


def mid_run_refusal(problem):
    """Return the scalar wall's problem refused part way through its schedule.

    Five steps of 0.1 um, then five that share the rest, with the flow's
    conditioning row declared over the short step only: the row is
    ``exp(dz (mu_max - mu_min))``, so the first long step, four hundred times the
    declared length, measures more than it declares and is refused by name, after
    five accepted ones.
    """
    flow = problem.linear.coupled_flow
    assert flow is not None and flow.generator is not None
    short = LENGTH / 2000
    long = (LENGTH - 5 * short) / 5
    steps = np.array([short] * 5 + [long] * 5)
    assert math.isclose(float(np.sum(steps)), LENGTH)
    declared = constant_schur_flow(
        problem.grids.omega,
        problem.linear.phi_breakpoints[[0, -1]],
        flow.generator[0, 0],
        dz_max=short,
        condition_limit=1e6,
        gauge_id=flow.gauge_id,
        certificate_id=flow.certificate_id,
    )
    # The certificate guards the Runge-Kutta march's inverse flow action; the
    # default exponential formulation acts forward only (ADR-044) and does not
    # measure it.
    return replace(
        problem,
        linear=replace(problem.linear, coupled_flow=declared),
        stepper=replace(
            problem.stepper, schedule=FixedSchedule(steps), integrator=None
        ),
    )


@pytest.mark.medium
@pytest.mark.parametrize("backend", installed(["numpy_ref", "numba_cpu", "native"]))
def test_a_flow_refused_mid_run_leaves_nothing_that_breaks_the_next_backend(backend):
    """A refusal after five accepted steps, then the same problem run again.

    Bound: the march's rounding budget (``tests/state_records.py``). The run after the
    failure is built fresh in the same process and must be the one built before it:
    the same fixed-order march, which anything the failure left behind would move by
    far more than rounding; the failed backend closes cleanly, and the refusal names
    the backend.
    """
    cls = backend_class(backend)
    simulation_ = build("scalar", integrator=FIXED_ORDER)
    problem, grid = simulation_.translate(), simulation_.grid
    before = run(cls, problem, grid)
    refused = mid_run_refusal(problem)
    failing = cls(refused, grid=grid)
    try:
        failing.set_initial(refused.initial.field0, None)
        with pytest.raises(BackendError, match="measured a coupled flow conditioning"):
            failing.propagate_to(LENGTH)
        # The five short steps were taken before the refusal: it happened mid-run.
        assert 0.0 < failing.observe().z < LENGTH
    finally:
        failing.close()
    after = run(cls, problem, grid)
    assert_matched(after, before, record_bound(simulation_, before))


@pytest.fixture(scope="module")
def moving():
    """The kinked graded-index taper's ten-step problem and grid, no collar."""
    length = 1e-3
    built = Simulation.from_waveguide(
        (),
        length,
        provider=grin_taper(kinked_radius(), kerr=True),
        grid=grin_grid(),
        save_planes=[length],
        dz_max=length / STEPS,
        dz_init=length / STEPS,
        schedule=FixedSchedule(np.full(STEPS, length / STEPS)),
        integrator=FIXED_ORDER,
        backend="numpy_ref",
        **{"noise": None, **GRIN_PULSE},
    )
    built.settings = dataclasses.replace(built.settings, absorber=None)
    return built, length


@pytest.mark.slow
@pytest.mark.parametrize("backend", installed(["numpy_ref", "numba_cpu", "native"]))
def test_a_moving_guide_that_fails_to_converge_leaves_the_next_backend_intact(
    moving, backend
):
    """The general route's moving guide: a controller failure, then the same run.

    The guide is lossless, so its frozen blocks are normal and no certificate can be
    exceeded; the failure forced here is the controller's instead. At a relative
    tolerance of 1e-30 every trial is rejected (its scaled error is about 1e14), and a
    first step that may not shrink (``dz_min = dz_init``) is a :class:`ConvergenceError`
    by name (``stepper.controller.limit_step``). The tolerance was 1e-300, whose scaled
    error overflows the norm's dot product on an OpenBLAS host and warns there. Bound:
    the march's rounding budget -- the backend built after the failure runs the
    fixed-order march the one built before it did.
    """
    built, length = moving
    problem, grid = built.translate(), built.grid
    cls = backend_class(backend)

    def final(one):
        driver = cls(one, grid=grid)
        try:
            driver.set_initial(one.initial.field0, None)
            driver.propagate_to(length)
            return driver.observe().require_field().copy()
        finally:
            driver.close()

    before = final(problem)
    stuck = replace(
        problem,
        stepper=replace(
            problem.stepper,
            schedule=None,
            controller=PIController(rtol=1e-30, atol=0.0),
            dz_min=problem.stepper.dz_init,
        ),
    )
    with pytest.raises(ConvergenceError):
        final(stuck)
    assert_matched(final(problem), before, record_bound(built, before))
