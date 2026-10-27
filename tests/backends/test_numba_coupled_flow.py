# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""``numba_cpu`` executes the coupled flow: parity, admission, FSAL, ledger, cost.

The compiled backend runs ``linear.coupled_flow`` in the combinations the reference
admits but a CF4 section: the constant-Schur section (the scalar wall, the TE/TM wall,
the split block on the general route) and the varying ``"frozen_residual"`` flow with
its explicit residual. The bound every parity test asserts is derived in
``tests/flow_parity.py`` before any measurement.
"""

import tracemalloc
import warnings
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from flow_parity import (
    CASES,
    GRID,
    LENGTH,
    QUIET,
    STEPS,
    UNIT,
    WEAK_COLLAR,
    apply_bound,
    build,
    departure,
    eigenbasis_formation_bound,
    flow_scales,
    formation_bound,
    reference_formation_bound,
    run,
    ten_step_bound,
)
from solprop.backends.driver import refuse_general_leaves
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numba_cpu.batch import NumbaCpuBatchBackend
from solprop.backends.numba_cpu.coupled_flow import (
    FLOW_LEDGER_ROWS,
    flow_accept,
    form_flow_actions,
)
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.backend import (
    _check_flow_conditioning,
    _coupled_flow_maps,
    _flow_action,
    _refuse_flow_conditioning,
)
from solprop.driver import prop_capillary
from solprop.exceptions import BackendError, DriverError
from solprop.modes import CoherentWall
from solprop.output.memory import MemoryOutput
from solprop.stepper import FixedSchedule
from solprop.translate.coupled import (
    constant_schur_flow,
    eigenbasis_limit,
    orthonormal_metric,
    schur_eigenbasis,
)

pytestmark = [
    pytest.mark.filterwarnings(f"ignore::{category.__module__}.{category.__name__}")
    for category in QUIET
] + [pytest.mark.filterwarnings("ignore:the temporal absorber attenuates")]
"""The warnings the qualification cases raise on purpose (``flow_parity.QUIET``)."""


@pytest.fixture(scope="module")
def cases():
    """Build each case's simulation and problem once for the module, on demand."""
    built = {}

    def get(name):
        if name not in built:
            simulation = build(name)
            built[name] = (simulation, simulation.translate())
        return built[name]

    return get


def spoiled(problem, **changes):
    """Return the problem with its coupled flow's fields replaced."""
    flow = problem.linear.coupled_flow
    return replace(
        problem, linear=replace(problem.linear, coupled_flow=replace(flow, **changes))
    )


# ------------------------------------------------------------------------ parity


@pytest.mark.parametrize("name", CASES)
def test_the_flow_s_factors_are_the_reference_s(cases, name):
    """The Schur factors, frozen blocks and certificate are the reference's arrays.

    Both backends take them from the reference's own host function on the same
    inputs, in one process: a structural contract (no compiled arithmetic forms them),
    so exact. What the compiled backend forms from them is bounded below.
    """
    simulation, problem = cases(name)
    backend = NumbaCpuBackend(problem, grid=simulation.grid)
    try:
        flow = backend._flow
        assert flow is not None
        reference = _coupled_flow_maps(problem)
        assert reference is not None
        for field in ("vectors", "triangular", "frozen", "condition_bound"):
            assert np.array_equal(getattr(flow.held, field), getattr(reference, field))
    finally:
        backend.close()


@pytest.mark.parametrize("route", ["eigenbasis", "scaling_and_squaring"])
@pytest.mark.parametrize("name", CASES)
def test_the_flow_s_actions_are_the_reference_s_within_the_formation_bound(
    cases, name, route
):
    """Every stage's compiled action pair is the reference's within ``eps_form``.

    Bound, stated first (``tests/flow_parity.py``): the compiled action and
    the reference's are each within their route's formation bound of the exact
    ``Z exp(l T) Z^H``, so within the sum of the two, relative to the action; the
    whole step's measured conditioning is the reference's singular-value product
    within ``32 n u``, half the certificate's own allowance. Every bin of these cases
    takes the eigenbasis; the second route forces every bin to the compiled scaling
    and squaring, which a nearly degenerate nonnormal block would take.
    """
    simulation, problem = cases(name)
    backend = NumbaCpuBackend(problem, grid=simulation.grid)
    try:
        flow = backend._flow
        assert flow is not None
        assert np.all(flow.route)
        if route == "scaling_and_squaring":
            flow.route[:] = False
        reference = _coupled_flow_maps(problem)
        assert reference is not None
        dz = float(np.max(problem.stepper.schedule.steps))
        bound = formation_bound(problem, dz)
        assert 0.0 < bound < 1e-12
        worst = 0.0
        for section in range(len(flow.held.vectors)):
            flow.prepare(section, dz)
            for slot, fraction in enumerate(flow.fractions):
                if fraction == 0.0:
                    continue
                pair = _flow_action(reference, section, float(fraction) * dz)
                for got, expected in zip(
                    (flow.forward[slot], flow.inverse[slot]), pair, strict=True
                ):
                    error = np.linalg.norm(got - expected, ord=2, axis=(-2, -1))
                    scale = np.linalg.norm(expected, ord=2, axis=(-2, -1))
                    worst = max(worst, float(np.max(error / scale)))
            forward, inverse = _flow_action(reference, section, dz)
            measured = np.linalg.norm(forward, ord=2, axis=(-2, -1)) * np.linalg.norm(
                inverse, ord=2, axis=(-2, -1)
            )
            assert np.max(abs(flow.measured / measured - 1.0)) <= (
                32 * flow.held.n_modes * UNIT
            )
        assert worst <= bound
    finally:
        backend.close()


def test_a_nearly_degenerate_nonnormal_bin_is_formed_by_scaling_and_squaring(cases):
    """The eigenbasis route is refused where its basis is ill-conditioned.

    A bin whose two eigenvalues are made to coincide to within 1e-9 of their spread
    with an order-one coupling has ``kappa`` about 1e9 and is routed to the compiled
    scaling and squaring; one made exactly degenerate has no eigenbasis at all and is
    routed there too, without a warning. Both are then the reference's within the
    squaring route's bound.
    """
    simulation, problem = cases("scalar")
    backend = NumbaCpuBackend(problem, grid=simulation.grid)
    try:
        flow = backend._flow
        assert flow is not None
        triangular = flow.held.triangular[0].copy()
        spread = abs(triangular[:2, 0, 0] - triangular[:2, 1, 1])
        triangular[0, 1, 1] = triangular[0, 0, 0] + 1e-9 * spread[0]
        triangular[1, 1, 1] = triangular[1, 0, 0]
        triangular[:2, 0, 1] = spread
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, _, _, route = schur_eigenbasis(flow.held.vectors[0], triangular)
        assert not route[0] and not route[1] and np.all(route[2:])
        held = replace(flow.held, triangular=triangular[None])
        left, right, eigenvalues, _ = schur_eigenbasis(flow.held.vectors[0], triangular)
        dz = LENGTH / STEPS
        form_flow_actions(
            flow.fractions,
            dz,
            left,
            right,
            eigenvalues,
            route,
            flow.held.vectors[0],
            triangular,
            flow.forward,
            flow.inverse,
            flow._exponents,
            flow._work,
        )
        pair = _flow_action(held, 0, dz)
        norm = float(np.max(np.linalg.norm(dz * triangular[:2], axis=(-2, -1))))
        bound = 2.0 * reference_formation_bound(flow.held.n_modes, norm, 0)
        for got, expected in zip(
            (flow.forward[-1], flow.inverse[-1]), pair, strict=True
        ):
            error = np.linalg.norm(got[:2] - expected[:2], ord=2, axis=(-2, -1))
            scale = np.linalg.norm(expected[:2], ord=2, axis=(-2, -1))
            assert np.all(error / scale <= bound)
    finally:
        backend.close()


@pytest.mark.parametrize("n_modes", [1, 2, 3, 4, 8, 16, 64])
def test_the_eigenbasis_limit_is_where_the_two_formation_bounds_meet(n_modes):
    """``eigenbasis_limit`` is the ``kappa`` equating the two routes' bounds at zero.

    At a vanishing step the eigenbasis route's bound is ``kappa (n + 11) u`` and the
    reference's scaling and squaring's ``16 u + 2 sqrt(2) n (n + 2) u``; the gate
    sits where they meet, so a gated bin is never formed less accurately, in bound.
    """
    limit = eigenbasis_limit(n_modes)
    eigen = eigenbasis_formation_bound(n_modes, 0.0, limit, 0.0, 0.0)
    assert eigen == pytest.approx(reference_formation_bound(n_modes, 0.0, 0), rel=1e-14)
    assert limit >= 2.0


def test_an_empty_measurement_is_not_refused(cases):
    """A flow with no active bin measures nothing and passes the certificate."""
    _, problem = cases("scalar")
    held = _coupled_flow_maps(problem)
    assert held is not None
    _refuse_flow_conditioning(held, 0, np.empty(0), LENGTH / STEPS)
    # And the reference's own check, before it measures anything.
    channels = problem.shape[0]
    nothing = np.empty((0, channels, channels), dtype=complex)
    _check_flow_conditioning(held, 0, (nothing, nothing), LENGTH / STEPS)


def on_march(problem, march):
    """Return the problem on the default integrator or on the Runge-Kutta march."""
    if march == "default":
        return problem
    return replace(problem, stepper=replace(problem.stepper, integrator=None))


MARCHES = pytest.mark.parametrize("march", ["default", "runge_kutta"])
"""Both integrators: each backend steps a flow through its own stage pipeline."""


@pytest.mark.medium
@MARCHES
@pytest.mark.parametrize("name", CASES)
def test_ten_steps_are_the_reference_within_the_bound(
    cases, name, march, record_property
):
    """Ten fixed steps on ``numba_cpu`` against ``numpy_ref``, within ``B_10``.

    The bound (``tests/flow_parity.py``), stated before the measurement:
    ``C^S exp(Phi) (||a_0||/||a_L||) S eps_step`` with ``C`` the declared
    conditioning, ``Phi`` the launch's nonlinear scale and ``eps_step`` the diagonal
    half's 32 u, the source's share and the flow's own products. The run is also
    checked to exercise the flow at all: without the leaf it departs by more than a
    thousand bounds.
    """
    simulation, problem = cases(name)
    problem = on_march(problem, march)
    grid = simulation.grid
    reference = run(NumpyRefBackend, problem, grid)
    compiled = run(NumbaCpuBackend, problem, grid)
    bound = ten_step_bound(flow_scales(problem, grid, reference))
    measured = departure(compiled, reference)
    record_property("bound", bound)
    record_property("measured", measured)
    assert bound > 0.0
    assert measured <= bound
    bare = replace(problem, linear=replace(problem.linear, coupled_flow=None))
    assert departure(run(NumbaCpuBackend, bare, grid), reference) > 1e3 * bound


@pytest.mark.medium
@MARCHES
@pytest.mark.parametrize("name", ["scalar", "taper", "gradient"])
def test_the_kerr_half_is_exercised_beside_the_flow(name, march):
    """The capillary cases' nonlinear half moves the answer by far more than the bound.

    The same case with its responses removed is a linear run -- which on
    ``numba_cpu`` takes the no-source branch, with the residual alone on a varying
    guide -- and it departs from the Kerr run by more than a thousand bounds; the
    linear run itself is the reference's within ``B_10`` at ``Phi = 0``, with the
    in-stage collar moved to the split-step scheme so that the stage has no absorber
    term either. The entry point refuses a capillary run with no nonlinearity, so the
    responses are removed from the built simulation rather than declared away.
    """
    simulation = build(name)
    problem = on_march(simulation.translate(), march)
    grid = simulation.grid
    nonlinear = run(NumpyRefBackend, problem, grid)
    # A fresh one: a simulation holds the problem it translated.
    linear = build(name)
    object.__setattr__(linear, "responses", ())
    linear_problem = linear.translate()
    assert len(linear_problem.responses.response_order) == 1
    linear_problem = on_march(
        replace(
            linear_problem,
            responses=replace(linear_problem.responses, absorber_scheme="split_step"),
        ),
        march,
    )
    compiled = run(NumbaCpuBackend, linear_problem, grid)
    reference = run(NumpyRefBackend, linear_problem, grid)
    scales = replace(flow_scales(linear_problem, grid, reference), phi=0.0)
    bound = ten_step_bound(scales)
    assert departure(compiled, reference) <= bound
    assert departure(compiled, nonlinear) > 1e3 * bound


def test_one_application_of_an_action_is_the_reference_s_within_eps_apply(cases):
    """The accept pass's ``U a`` against ``einsum``, on a real wall's action.

    Bound, stated first: ``eps_apply = 2 sqrt(2) (n + 2) sqrt(n) C u`` relative to
    the result, with the declared conditioning ``C``.
    """
    simulation, problem = cases("taper")
    backend = NumbaCpuBackend(problem, grid=simulation.grid)
    try:
        flow = backend._flow
        assert flow is not None
        dz = LENGTH / STEPS
        flow.prepare(2, dz)
        state = np.ascontiguousarray(problem.initial.field0[0, :, 0, :])
        zero = np.zeros_like(state)
        advanced = np.zeros_like(state)
        flow_accept(
            state,
            np.ones_like(state),
            dz,
            zero,
            zero,
            flow.forward[-1],
            flow.slots,
            np.ones(state.shape[1], dtype=bool),
            advanced,
            np.zeros_like(state),
            np.zeros((3, state.shape[0])),
        )
        active = flow.held.active
        expected = state.copy()
        expected[:, active] = np.einsum(
            "wij,jw->iw", flow.forward[-1], state[:, active]
        )
        bound = apply_bound(flow.held.n_modes, float(np.max(flow.held.condition_bound)))
        assert departure(advanced, expected) <= bound
    finally:
        backend.close()


# --------------------------------------------------------------------- admission


@MARCHES
def test_every_flow_the_reference_refuses_numba_refuses_in_the_same_words(cases, march):
    """The reference's own refusals run under this backend's name: one list.

    Each spoiler is refused by both, and the two messages differ in the backend's
    name alone, so the admitted sets cannot drift apart. Under the default
    integrator a section method it does not step is refused first, by the
    integrator; the Runge-Kutta march reaches the leaf's own list.
    """
    simulation, problem = cases("scalar")
    problem = on_march(problem, march)
    grid = simulation.grid
    flow = problem.linear.coupled_flow
    assert flow is not None and flow.generator is not None
    constant = problem.linear.phi_constant
    assert constant is not None
    varied = constant.copy()
    varied[0, 1] *= 1.0000001
    metric = orthonormal_metric(
        problem.grids.omega,
        problem.linear.phi_breakpoints[[0, -1]],
        defect=0.0,
        gauge_id="lab",
        certificate_id="sampled-gram",
    )
    spoilers = {
        "method": spoiled(problem, section_methods=("diagonal",)),
        "connection": spoiled(
            problem, connection=np.zeros((1, 1, 3, 3, grid.n_omega), dtype=complex)
        ),
        "residual": spoiled(problem, residual_norm_bound=np.ones((1, grid.n_omega))),
        "degree": spoiled(
            problem, generator=np.concatenate([flow.generator] * 2, axis=1)
        ),
        "factored": spoiled(
            problem,
            generator=None,
            fixed_terms=np.zeros((1, 3, 3, grid.n_omega), dtype=complex),
            term_weights=np.zeros((1, 1, 1), dtype=complex),
            wall_vectors=None,
            wall_weights=None,
        ),
        "phi": replace(problem, linear=replace(problem.linear, phi_constant=varied)),
        "metric": replace(problem, linear=replace(problem.linear, modal_metric=metric)),
        "noise": replace(
            problem,
            initial=replace(
                problem.initial,
                noise0=np.zeros_like(problem.initial.field0),
                # Input noise is admitted beside a coupled flow (SEP 37); the
                # modified model is still refused, by both backends alike.
                noise_model="modified",
            ),
        ),
    }
    for name, problem_spoiled in spoilers.items():
        with pytest.raises(BackendError) as theirs:
            NumpyRefBackend(problem_spoiled, grid=grid)
        with pytest.raises(BackendError) as ours:
            NumbaCpuBackend(problem_spoiled, grid=grid)
        assert "numpy_ref" in str(theirs.value), name
        # The one difference by design: a varying flow's methods, where this
        # backend names 'frozen_residual' alone because it refuses 'cf4' first.
        expected = (
            str(theirs.value)
            .replace("numpy_ref", "numba_cpu")
            .replace("'frozen_residual' or 'cf4'", "'frozen_residual'")
        )
        assert str(ours.value) == expected, name


def test_a_backend_that_does_not_admit_the_leaf_is_refused_by_its_name(cases):
    """The shared leaf door, which both in-tree backends now pass for a flow.

    Every in-tree backend names all four general-modal leaves, so the door
    ``refuse_general_leaves`` keeps for a separately built adapter is exercised here
    directly: a backend admitting none of them is refused by the flow's dotted name.
    """
    _, problem = cases("scalar")
    with pytest.raises(BackendError, match=r"adapter does not execute linear\.coupled"):
        refuse_general_leaves(problem, "adapter")
    refuse_general_leaves(problem, "adapter", admits=("linear.coupled_flow",))


def test_a_varying_carrier_beside_a_constant_section_is_refused_in_the_same_words():
    """The reference's constant-representation door, under this backend's name."""
    simulation = build("scalar", pressure=(1.0, 0.9), wall=None)
    problem = simulation.translate()
    spoiled_problem = replace(
        problem,
        linear=replace(
            problem.linear,
            coupled_flow=constant_schur_flow(
                problem.grids.omega,
                problem.linear.phi_breakpoints[[0, -1]],
                np.zeros((3, 3, simulation.grid.n_omega), dtype=complex),
                dz_max=problem.stepper.dz_max,
                condition_limit=1e6,
                gauge_id="capillary HE_1m",
                certificate_id="constant-schur",
            ),
        ),
    )
    with pytest.raises(BackendError, match="numba_cpu executes a constant-Schur"):
        NumbaCpuBackend(spoiled_problem, grid=simulation.grid)


def test_a_cf4_section_is_refused_by_name_at_the_backend_and_the_entry_point():
    """CF4 stays the reference's: named at both doors, never run as frozen."""
    cf4 = CoherentWall(method="cf4")
    simulation = build("taper", wall=cf4)
    problem = simulation.translate()
    assert problem.linear.coupled_flow is not None
    assert "cf4" in problem.linear.coupled_flow.section_methods
    with pytest.raises(
        BackendError, match=r"'cf4' section of linear\.coupled_flow.*on numpy_ref"
    ):
        NumbaCpuBackend(problem, grid=simulation.grid)
    with pytest.raises(DriverError, match=r"method='cf4'.*not yet implemented"):
        build("taper", wall=cf4, backend="numba_cpu")


def test_a_flow_beside_a_free_space_problem_is_refused_by_name(cases):
    """No translation builds it; a hand-built one is named rather than half run."""
    simulation, problem = cases("scalar")
    linear = replace(problem.linear)
    object.__setattr__(linear, "separable", object())
    with pytest.raises(BackendError, match="Cartesian free-space"):
        NumbaCpuBackend(replace(problem, linear=linear), grid=simulation.grid)


def test_a_step_longer_than_the_certificate_is_refused_under_numba_s_name(cases):
    """The declared conditioning row is measured per step here too, and named.

    The certificate guards the Runge-Kutta march's inverse flow action; the default
    exponential formulation acts forward only (ADR-044) and does not measure it.
    """
    simulation, problem = cases("scalar")
    problem = replace(problem, stepper=replace(problem.stepper, integrator=None))
    flow = problem.linear.coupled_flow
    assert flow is not None and flow.generator is not None
    understated = replace(
        problem,
        linear=replace(
            problem.linear,
            coupled_flow=constant_schur_flow(
                problem.grids.omega,
                problem.linear.phi_breakpoints[[0, -1]],
                flow.generator[0, 0],
                dz_max=problem.stepper.dz_max / 100.0,
                condition_limit=1e6,
                gauge_id=flow.gauge_id,
                certificate_id=flow.certificate_id,
            ),
        ),
    )
    with pytest.raises(BackendError, match="numba_cpu measured a coupled flow"):
        run(NumbaCpuBackend, understated, simulation.grid)


def test_an_action_needing_too_many_squarings_is_refused_under_numba_s_name(cases):
    """The squaring limit is the reference's, and so is its message."""
    simulation, problem = cases("scalar")
    backend = NumbaCpuBackend(problem, grid=simulation.grid)
    try:
        flow = backend._flow
        assert flow is not None
        with pytest.raises(BackendError, match="numba_cpu cannot take a coupled"):
            flow.prepare(0, 1e30)
        # A refused length leaves nothing held behind a key: the next prepare forms.
        flow.prepare(0, LENGTH / STEPS)
        assert flow.section == 0
    finally:
        backend.close()


def test_a_lockstep_batch_refuses_a_member_carrying_a_flow(cases):
    """Refused exactly as the reference's batch refuses it: by the leaf's name."""
    simulation, problem = cases("scalar")
    member = NumbaCpuBackend(problem, grid=simulation.grid)
    try:
        with pytest.raises(BackendError, match="lockstep batch does not execute"):
            NumbaCpuBatchBackend([member])
    finally:
        member.close()


# ---------------------------------------------------------- FSAL, ledger, memory


def test_fsal_is_not_taken_beside_a_flow_so_every_stage_is_evaluated(
    cases, monkeypatch
):
    """The reference evaluates all seven stages of a flow step, and so does this.

    FSAL would reuse the last stage through the whole step's forward action and only
    within one section; parity is kept here and the saving is E5's to measure. FSAL
    and the seven stages are the Runge-Kutta march's.
    """
    simulation, problem = cases("scalar")
    problem = replace(problem, stepper=replace(problem.stepper, integrator=None))
    backend = NumbaCpuBackend(problem, grid=simulation.grid)
    calls = []
    original = backend._stage

    def counted(*arguments):
        calls.append(arguments[1])
        return original(*arguments)

    monkeypatch.setattr(backend, "_stage", counted)
    try:
        assert not backend._fsal
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(LENGTH)
        assert not backend._has_recycled
    finally:
        backend.close()
    assert len(calls) == STEPS * problem.stepper.tableau.n_stages


def test_every_flow_ledger_row_is_zero_on_a_route_without_one():
    """The rows are named on every route and hold nothing where no flow is carried."""
    simulation = build("scalar", wall=None)
    backend = NumbaCpuBackend(simulation.translate(), grid=simulation.grid)
    try:
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    assert all(ledger[row] == 0 for row in FLOW_LEDGER_ROWS)


@pytest.mark.parametrize("name", ["scalar", "taper"])
def test_the_ledger_counts_the_flow_s_buffers_and_the_reference_s_rows(cases, name):
    """The three shared rows are the reference's numbers; the rest are owned buffers."""
    simulation, problem = cases(name)
    ours = NumbaCpuBackend(problem, grid=simulation.grid)
    theirs = NumpyRefBackend(problem, grid=simulation.grid)
    try:
        ledger, reference = ours.memory_ledger(), theirs.memory_ledger()
        flow = ours._flow
        assert flow is not None
        for row in FLOW_LEDGER_ROWS[:3]:
            assert ledger[row] == reference[row], row
        owned = {
            "coupled_flow_eigenbasis": flow.left.nbytes
            + flow.right.nbytes
            + flow.eigenvalues.nbytes
            + flow.route.nbytes
            + flow.row_sum.nbytes,
            "coupled_flow_formation": flow.measured.nbytes
            + 16 * (2 * flow.held.n_modes + 4 * flow.held.n_modes**2),
            "coupled_flow_residual_action": flow.block,
            "coupled_flow_lab": flow.lab,
            "coupled_flow_drive": flow.drive,
            "coupled_flow_error": flow.error,
            "coupled_flow_slots": flow.slots.nbytes + flow.residual_slots.nbytes,
        }
        for row, buffer in owned.items():
            size = buffer if isinstance(buffer, int) else buffer.nbytes
            assert ledger[row] == size, row
        assert (ledger["coupled_flow_residual_action"] > 0) == (name == "taper")
    finally:
        ours.close()
        theirs.close()


def test_a_trial_at_a_held_length_allocates_nothing_the_diagonal_route_does_not(
    cases,
):
    """The flow's passes add no allocation to a warm trial.

    Measured with ``tracemalloc`` on a warm trial at the length the actions were
    formed for, against the same trial of the same problem without the leaf: the
    diagonal route's own host work (the propagator's exponent, the returned copy of
    the state) is what both allocate, and the flow may add small objects and nothing
    field-sized.
    """
    simulation, problem = cases("taper")
    grid = simulation.grid
    dz = LENGTH / STEPS
    bare = replace(problem, linear=replace(problem.linear, coupled_flow=None))
    peaks = []
    for one in (problem, bare):
        backend = NumbaCpuBackend(one, grid=grid)
        try:
            backend.set_initial(one.initial.field0, None)
            backend._trial_step(0.0, dz)
            tracemalloc.start()
            backend._trial_step(0.0, dz)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        finally:
            backend.close()
        peaks.append(peak)
    # Below one field-sized buffer: what the flow adds is Python's small objects and
    # the stage bundles' own `flow_weights` rows (one float per term per stage, made
    # by the shared driver for either backend), measured 4.2 KiB here at most.
    _, n_modes, n_pol, n_omega = problem.shape
    assert peaks[0] - peaks[1] < n_modes * n_pol * n_omega * 16


# -------------------------------------------------------------------- end to end


@pytest.mark.medium
def test_prop_capillary_runs_a_wall_on_numba_as_the_reference_does():
    """``prop_capillary(..., wall=CoherentWall(), backend="numba_cpu")`` end to end.

    The user's door, with the record, the planes and the statistics: the last
    plane's field is the reference run's within the ten-step bound of the scalar
    case, its energy within the same relative bound, and both records carry the
    same flow entry.
    """
    outputs = {}
    for backend in ("numpy_ref", "numba_cpu"):
        with warnings.catch_warnings():
            for category in QUIET:
                warnings.simplefilter("ignore", category)
            outputs[backend] = cast(
                "MemoryOutput",
                prop_capillary(
                    50e-6,
                    LENGTH,
                    "He",
                    1.0,
                    wall=CoherentWall(),
                    lambda0=800e-9,
                    grid=GRID,
                    absorber=WEAK_COLLAR,
                    fwhm=30e-15,
                    energy=1e-4,
                    modes=3,
                    n_save=2,
                    backend=backend,
                    dz_max=LENGTH / STEPS,
                    dz_init=LENGTH / STEPS,
                    schedule=FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
                    plasma=False,
                    raman=False,
                    noise=None,
                ),
            )
    simulation = build("scalar")
    problem = simulation.translate()
    reference_plane, compiled_plane = (
        outputs[one].planes()[-1] for one in ("numpy_ref", "numba_cpu")
    )
    bound = ten_step_bound(flow_scales(problem, simulation.grid, reference_plane.field))
    assert departure(compiled_plane.field, reference_plane.field) <= bound
    energies = [
        float(np.sum(plane.stats["energy"]))
        for plane in (reference_plane, compiled_plane)
    ]
    assert energies[1] == pytest.approx(energies[0], rel=bound, abs=0.0)
    records = [outputs[one].metadata for one in ("numpy_ref", "numba_cpu")]
    assert all(record is not None and record.modal is not None for record in records)
    assert records[0].modal["flow"] == records[1].modal["flow"]  # type: ignore[union-attr]


@pytest.mark.medium
def test_a_split_block_runs_through_from_waveguide_on_numba():
    """Slice D3b's rotated pair through the entry point, now on ``numba_cpu`` too."""
    runs = {}
    for backend in ("numpy_ref", "numba_cpu"):
        simulation = build("split", backend=backend)
        output = simulation.run(progress="none")
        assert isinstance(output, MemoryOutput)
        runs[backend] = output.planes()[-1].field
    simulation = build("split")
    problem = simulation.translate()
    bound = ten_step_bound(flow_scales(problem, simulation.grid, runs["numpy_ref"]))
    assert departure(runs["numba_cpu"], runs["numpy_ref"]) <= bound
