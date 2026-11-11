# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The coupled flow's doors and factors on ``native``, against the reference (SEP 43).

These carry ``numba_cpu``'s ``test_numba_coupled_flow.py`` where its obligation was not
Numba's: what is refused, in which words, what the two backends share exactly, and that
the Kerr half is really on beside a flow.
"""

import importlib.util
from dataclasses import replace

import numpy as np
import pytest

from bounds import UNIT_ROUNDOFF, higham_gamma
from flow_parity import (
    QUIET,
    build,
    departure,
    flow_scales,
    run,
    ten_step_bound,
)
from solprop.backends.driver import refuse_general_leaves
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.backend import _coupled_flow_maps
from solprop.exceptions import BackendError
from solprop.translate.coupled import constant_schur_flow, orthonormal_metric

pytestmark = [
    pytest.mark.filterwarnings(f"ignore::{category.__module__}.{category.__name__}")
    for category in QUIET
] + [
    pytest.mark.filterwarnings("ignore:the temporal absorber attenuates"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
]

needs_native = pytest.mark.skipif(
    importlib.util.find_spec("solprop_native") is None,
    reason="solprop_native is not installed",
)


def native_class():
    """Return the optional wheel's backend class."""
    # Optional wheel: absent from the base environment.
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        NativeBackend,
    )

    return NativeBackend


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


def on_march(run_request, march):
    """Return the request on the default integrator or on the Runge-Kutta march."""
    if march == "default":
        return run_request
    return run_request.with_stepper(replace(run_request.stepper, integrator=None))


MARCHES = pytest.mark.parametrize("march", ["default", "runge_kutta"])


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


@pytest.mark.parametrize("backend", ["numpy_ref", "native"])
def test_a_varying_carrier_beside_a_constant_section_is_refused_in_the_same_words(
    backend,
):
    """The reference's constant-representation door, under each backend's name."""
    if backend == "native":
        pytest.importorskip("solprop_native")
    implementation = NumpyRefBackend if backend == "numpy_ref" else native_class()
    simulation = build("scalar", pressure=(1.0, 0.9), wall=None)
    problem = simulation.translate()
    run_request = simulation._run_request()
    spoiled_problem = replace(
        problem,
        linear=replace(
            problem.linear,
            coupled_flow=constant_schur_flow(
                problem.grids.omega,
                problem.linear.phi_breakpoints[[0, -1]],
                np.zeros((3, 3, simulation.grid.n_omega), dtype=complex),
                dz_max=run_request.stepper.dz_max,
                condition_limit=1e6,
                gauge_id="capillary HE_1m",
                certificate_id="constant-schur",
            ),
        ),
    )
    with pytest.raises(BackendError, match=f"{backend} executes a constant-Schur"):
        implementation(spoiled_problem, run_request, grid=simulation.grid)


@needs_native
@pytest.mark.parametrize("name", ["scalar", "te_tm", "split"])
def test_a_constant_flow_s_factors_are_the_reference_s(cases, name):
    """The Schur factors, frozen blocks and certificate are the reference's arrays.

    Both backends form them on the host from one stored block: a structural contract (no
    compiled arithmetic forms them, and a section of one degree is that block unchanged,
    ``frozen_generator``), so exact, as the retired test held every case.
    """
    simulation, problem = cases(name)
    backend = native_class()(problem, simulation._run_request(), grid=simulation.grid)
    try:
        flow = backend._flow
        assert flow is not None
        reference = _coupled_flow_maps(problem)
        assert reference is not None
        for field in ("vectors", "triangular", "frozen", "condition_bound"):
            assert np.array_equal(getattr(flow, field), getattr(reference, field)), (
                field
            )
    finally:
        backend.close()


@needs_native
@pytest.mark.parametrize("name", ["taper", "gradient"])
def test_a_varying_flow_s_factors_are_the_reference_s_to_their_rounding(cases, name):
    """A varying flow's frozen block and its factors, weaker than the retired identity.

    The retired test held these exactly too, but the frozen block's ``tensordot`` goes
    through the host BLAS, whose rounding depends on its width, and a live native
    backend fits that width: the block formed under one and the reference's formed
    under another differ in the last bits. Schur vectors are not continuous in their
    matrix where eigenvalues nearly coincide, so the factors can differ by far more
    than that (0.71 of the largest element on the taper, 2026-11-03). So: each frozen
    element within ``2 gamma(n_deg)`` of the magnitudes' sum, the certificate exactly
    (it is read, not formed), and native's factors reassembling the block to the Schur
    decomposition's backward error, ``10 n^2 u`` of its norm.
    """
    simulation, problem = cases(name)
    backend = native_class()(problem, simulation._run_request(), grid=simulation.grid)
    try:
        flow = backend._flow
        assert flow is not None
        reference = _coupled_flow_maps(problem)
        assert reference is not None
        tables = problem.linear.coupled_flow
        active = flow.active
        majorant = np.sum(np.abs(tables.generator), axis=1)[:, :, :, active]
        majorant = np.moveaxis(majorant, -1, 1)
        rounding = 2 * higham_gamma(tables.n_deg) * majorant
        assert np.all(np.abs(flow.frozen - reference.frozen) <= rounding)
        assert np.array_equal(flow.condition_bound, reference.condition_bound)
        n = tables.n_modes
        rebuilt = (
            flow.vectors @ flow.triangular @ np.conj(np.swapaxes(flow.vectors, -1, -2))
        )
        limit = 10 * n**2 * UNIT_ROUNDOFF * np.linalg.norm(flow.frozen, axis=(-2, -1))
        assert np.all(np.linalg.norm(rebuilt - flow.frozen, axis=(-2, -1)) <= limit)
    finally:
        backend.close()


@needs_native
@MARCHES
def test_every_flow_the_reference_refuses_native_refuses_in_the_same_words(
    cases, march
):
    """The reference's own refusals run under native's name: one list.

    Each spoiler is refused by both, and the two messages differ in the backend's
    name alone, so the admitted sets cannot drift apart. Under the default
    integrator a section method it does not step is refused first, by the
    integrator; the Runge-Kutta march reaches the leaf's own list.
    """
    simulation, problem = cases("scalar")
    run_request = on_march(simulation._run_request(), march)
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
            NumpyRefBackend(problem_spoiled, run_request, grid=grid)
        with pytest.raises(BackendError) as ours:
            native_class()(problem_spoiled, run_request, grid=grid)
        assert "numpy_ref" in str(theirs.value), name
        # The one difference by design: a varying flow's methods, where native names
        # 'frozen_residual' alone because it refuses 'cf4' first.
        expected = (
            str(theirs.value)
            .replace("numpy_ref", "native")
            .replace("'frozen_residual' or 'cf4'", "'frozen_residual'")
        )
        assert str(ours.value) == expected, name


@needs_native
@pytest.mark.slow
@MARCHES
@pytest.mark.parametrize("name", ["scalar", "taper", "gradient"])
def test_the_kerr_half_is_exercised_beside_the_flow(name, march):
    """The capillary cases' nonlinear half moves the answer by far more than the bound.

    The same case with its responses removed is a linear run -- which takes native's
    no-source branch, with the residual alone on a varying guide -- and it departs from
    the Kerr run by more than a thousand bounds; the linear run itself is the
    reference's within ``B_10`` at ``Phi = 0``, with the in-stage collar moved to the
    split-step scheme so that the stage has no absorber term either. The entry point
    refuses a capillary run with no nonlinearity, so the responses are removed from the
    built simulation rather than declared away.
    """
    simulation = build(name)
    problem = simulation.translate()
    run_request = on_march(simulation._run_request(), march)
    grid = simulation.grid
    nonlinear = run(NumpyRefBackend, problem, run_request, grid)
    # A fresh one: a simulation holds the problem it translated.
    linear = build(name)
    object.__setattr__(linear, "responses", ())
    linear_problem = linear.translate()
    assert len(linear_problem.responses.response_order) == 1
    linear_problem = replace(
        linear_problem,
        responses=replace(linear_problem.responses, absorber_scheme="split_step"),
    )
    linear_request = on_march(linear._run_request(), march)
    compiled = run(native_class(), linear_problem, linear_request, grid)
    reference = run(NumpyRefBackend, linear_problem, linear_request, grid)
    scales = replace(
        flow_scales(linear_problem, linear_request, grid, reference), phi=0.0
    )
    bound = ten_step_bound(scales)
    assert departure(compiled, reference) <= bound
    assert departure(compiled, nonlinear) > 1e3 * bound
