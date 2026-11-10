# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native's coupled flow on the retired kernels' pathological inputs (SEP 43).

The inventory's items 26 to 30. The Numba kernels were called one at a time on random
operands; native's flow runs through its state, so these reach the same inputs through
the held Schur factors and the problem's mask, and compare with ``numpy_ref``:

- a bin of ``kappa`` about 1e9 and an exactly degenerate bin, routed off the eigenbasis
  and formed natively by scaling and squaring, within twice the squaring route's bound;
- an empty measurement through the conditioning guards, and a flow with no active bin;
- the spectral norm's Jacobi with a zero pivot and with no rotation at all;
- the error norm's three partials over a mask with a hard zero on the block, the old
  norm's being the state's before the action;
- why the finish's non-contiguous crop has no witness: translation refuses the only
  grid that would have one.

**Weaker than the original.** The accept pass is read through the partials of one real
trial step rather than called on random operands. The spectral norm is read as the
product of two norms, as ``form_flow_actions`` returns it, so each norm's own ``8 n u``
is reached only through their product's ``16 n u``. The finish's reversed crop is out
of reach, so it is not ported.
"""

from dataclasses import replace

import numpy as np
import pytest

import flow_parity
from flow_parity import QUIET, STEPS, UNIT, build, reference_formation_bound
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref import passes as reference_passes
from solprop.backends.numpy_ref.backend import (
    _check_flow_conditioning,
    _coupled_flow_maps,
    _flow_action,
)
from solprop.boundaries import absorber
from solprop.exceptions import BackendError, TranslationError
from solprop.grid import EnvelopeGrid
from solprop.translate.coupled import schur_eigenbasis

pytestmark = [
    pytest.mark.filterwarnings(f"ignore::{category.__module__}.{category.__name__}")
    for category in QUIET
] + [
    pytest.mark.filterwarnings("ignore:the temporal absorber attenuates"),
    # Windows kept tight for speed: the window's edge isn't what this module tests.
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
    # A bin off the eigenbasis route takes the host march, which is the route these
    # tests form actions for and never step.
    pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning"),
]

solprop_native = pytest.importorskip("solprop_native")
NativeBackend = solprop_native.NativeBackend
native_backend = pytest.importorskip("solprop_native.backend")


def scalar_case():
    """Return the scalar wall's problem, request and grid: three modes, ten steps."""
    simulation = build("scalar")
    return simulation.translate(), simulation._run_request(), simulation.grid


def with_factors(held, triangular):
    """Return `held` with section 0's Schur factor replaced and its route recomputed."""
    stack = held.triangular.copy()
    stack[0] = triangular
    left, right, eigenvalues, route = schur_eigenbasis(held.vectors[0], triangular)
    return replace(
        held,
        triangular=stack,
        left=np.concatenate((left[None], held.left[1:])),
        right=np.concatenate((right[None], held.right[1:])),
        eigenvalues=np.concatenate((eigenvalues[None], held.eigenvalues[1:])),
        route=np.concatenate((route[None], held.route[1:])),
        row_sum=np.asarray(
            [float(np.max(np.sum(abs(one), axis=-1), initial=0.0)) for one in stack]
        ),
    )


def formed(monkeypatch, problem, run_request, grid, held, dz):
    """Return native's whole-step actions and measurement from the factors `held`."""
    monkeypatch.setattr(native_backend, "_host_flow", lambda _: held)
    backend = NativeBackend(problem, run_request, grid=grid)
    try:
        native = backend._native_state
        n_modes = held.vectors.shape[-1]
        tableau = run_request.stepper.tableau
        fractions = np.array([*(float(node) for node in tableau.nodes), 1.0])
        measured = np.asarray(native.form_flow_actions(0, dz, fractions)[0])
        shape = (fractions.size, -1, n_modes, n_modes)
        forward = native.flow_actions_to_host(inverse=False).reshape(shape)[-1]
        inverse = native.flow_actions_to_host(inverse=True).reshape(shape)[-1]
        return forward, inverse, measured
    finally:
        backend.close()


def test_a_nearly_degenerate_nonnormal_bin_is_formed_by_scaling_and_squaring(
    monkeypatch,
):
    """Eigenvalues within 1e-9 of their spread, and exactly coincident, on native.

    The original's construction and bound: an order-one coupling makes ``kappa`` about
    1e9, so that bin leaves the eigenbasis, and an exact coincidence has no eigenbasis,
    without a warning (warnings are errors here). Both are then formed by native's
    scaling and squaring, within twice the reference's squaring bound of its action.
    """
    problem, run_request, grid = scalar_case()
    held = native_backend._host_flow(problem)
    triangular = held.triangular[0].copy()
    spread = abs(triangular[:2, 0, 0] - triangular[:2, 1, 1])
    triangular[0, 1, 1] = triangular[0, 0, 0] + 1e-9 * spread[0]
    triangular[1, 1, 1] = triangular[1, 0, 0]
    triangular[:2, 0, 1] = spread
    crafted = with_factors(held, triangular)
    route = crafted.route[0]
    assert not route[0]
    assert not route[1]
    assert np.all(route[2:])
    dz = flow_parity.LENGTH / STEPS
    forward, inverse, _ = formed(monkeypatch, problem, run_request, grid, crafted, dz)
    reference = _coupled_flow_maps(problem)
    assert reference is not None
    expected = _flow_action(replace(reference, triangular=crafted.triangular), 0, dz)
    norm = float(np.max(np.linalg.norm(dz * triangular[:2], axis=(-2, -1))))
    bound = 2.0 * reference_formation_bound(reference.n_modes, norm, 0)
    for got, want in zip((forward, inverse), expected, strict=True):
        error = np.linalg.norm(got[:2] - want[:2], ord=2, axis=(-2, -1))
        scale = np.linalg.norm(want[:2], ord=2, axis=(-2, -1))
        assert np.all(error / scale <= bound)


def test_an_empty_measurement_is_not_refused():
    """A flow with no active bin measures nothing and passes the certificate."""
    problem, _, _ = scalar_case()
    dz = flow_parity.LENGTH / STEPS
    held = native_backend._host_flow(problem)
    native_backend._check_flow_conditioning(held, 0, np.empty(0), dz)
    reference = _coupled_flow_maps(problem)
    assert reference is not None
    channels = problem.shape[0]
    nothing = np.empty((0, channels, channels), dtype=complex)
    _check_flow_conditioning(reference, 0, (nothing, nothing), dz)
    # The guard is live: one bin over its declared row is refused, on either.
    over = np.array(held.condition_bound[0]) * 2.0
    with pytest.raises(BackendError, match="above"):
        native_backend._check_flow_conditioning(held, 0, over, dz)


def test_the_spectral_norm_bounds_the_singular_value_to_its_rounding(monkeypatch):
    """The Jacobi of the Gram matrix with a zero pivot and with no rotation at all.

    The original's blocks: a coupling of channels 0 and 2 alone leaves the Gram matrix a
    zero pivot, which takes the skipped rotation, and a diagonal block no rotation. The
    rest are random and non-normal. The factor is taken as the block's exponent, and the
    bound compares native's measurement with the singular values of the very matrices
    native formed, so its operands are the same: ``16 n u`` on the product.
    """
    problem, run_request, grid = scalar_case()
    held = native_backend._host_flow(problem)
    dz = flow_parity.LENGTH / STEPS
    triangular = held.triangular[0].copy()
    n_modes = triangular.shape[-1]
    assert n_modes == 3
    rng = np.random.default_rng(8)
    for bin_ in range(6):
        block = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        triangular[bin_] = np.triu(block) / dz
    coupled = np.zeros((3, 3), dtype=complex)
    coupled[0, 2] = (0.3 + 0.1j) / dz
    triangular[6] = coupled
    triangular[7] = np.diag(np.log([1.0, 2.0, 0.5])).astype(complex) / dz
    vectors = held.vectors.copy()
    vectors[0, :8] = np.eye(3)
    crafted = with_factors(replace(held, vectors=vectors), triangular)
    # Sections after the first keep their own: only section 0 is formed here.
    forward, inverse, measured = formed(
        monkeypatch, problem, run_request, grid, crafted, dz
    )
    # The zero pivot's Gram matrix, and the diagonal's, as the docstring says.
    gram = np.conj(forward[6].T) @ forward[6]
    assert gram[0, 1] == 0.0
    assert gram[1, 2] == 0.0
    assert np.count_nonzero(forward[7] - np.diag(np.diag(forward[7]))) == 0
    expected = np.linalg.norm(forward, ord=2, axis=(-2, -1)) * np.linalg.norm(
        inverse, ord=2, axis=(-2, -1)
    )
    assert np.all(abs(measured / expected - 1.0) <= 16 * n_modes * UNIT)


def test_the_partials_read_a_mask_with_a_hard_zero_and_the_old_norm_before_the_action():
    """The error norm's three partials beside a flow, a populated block bin masked off.

    The old norm is the launch's own, ``sum |a|^2`` over the mask, not the action's
    ``U a``: the original's last assertion. Both operands are the launch, so the
    original's relative ``BOUND`` holds, one sum's. The new norm and the error norm are
    formed from the state each side reached, so they are within twice the ten-step bound
    of ``flow_parity`` on one step, the squares of quantities within it.
    """
    problem, run_request, grid = scalar_case()
    dz = flow_parity.LENGTH / STEPS
    field = np.asarray(problem.initial.field0)
    held = native_backend._host_flow(problem)
    active = np.flatnonzero(held.active)
    power = np.sum(np.abs(field) ** 2, axis=(0, 1))
    # The active bin holding most of the launch, masked off with a hard zero.
    masked = active[int(np.argmax(power[active]))]
    mask = np.array(problem.grids.norm_mask)
    assert mask[masked]
    mask[masked] = False
    problem = replace(problem, grids=replace(problem.grids, norm_mask=mask))

    backend = NativeBackend(problem, run_request, grid=grid)
    try:
        backend.set_initial(field, None)
        partials = np.asarray(backend._prepare_trial(0.0, dz))
    finally:
        backend.close()

    captured = {}
    original = reference_passes.error_norm

    def capture(error, state, advanced, reference_mask, controller):
        captured["partials"] = [
            np.sum(np.abs(one[..., reference_mask]) ** 2)
            for one in (state, advanced, error)
        ]
        return original(error, state, advanced, reference_mask, controller)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(reference_passes, "error_norm", capture)
        reference = NumpyRefBackend(problem, run_request, grid=grid)
        try:
            reference.set_initial(field, None)
            reference._trial_step(0.0, dz)
            expected_state = reference._require_state().copy()
        finally:
            reference.close()

    old, new, error = (float(np.sum(row)) for row in partials[:3])
    n_modes = field.shape[0]
    sum_bound = 2.0 * np.sqrt(2.0) * (n_modes + 8) * np.sqrt(n_modes) * UNIT
    masked_old = float(np.sum(np.abs(expected_state[..., mask]) ** 2))
    assert old == pytest.approx(masked_old, rel=sum_bound)
    assert old == pytest.approx(captured["partials"][0], rel=sum_bound)
    # Vacuous unless the mask moves it by far more than the bound.
    whole = float(np.sum(np.abs(expected_state) ** 2))
    assert abs(whole - old) > 1e3 * sum_bound * whole
    expected = flow_parity.run(NumpyRefBackend, problem, run_request, grid)
    scales = flow_parity.flow_scales(problem, run_request, grid, expected)
    bound = 2.0 * flow_parity.ten_step_bound(scales)
    assert new == pytest.approx(captured["partials"][1], rel=bound)
    assert error == pytest.approx(captured["partials"][2], rel=bound)


def test_a_flow_on_a_wrapped_crop_is_refused_at_translation():
    """Why the finish's reversed crop has no native witness: no problem can have one.

    The original fed ``make_flow_finish`` a reversed crop on random operands. A field
    grid's crop is the identity, and the only other crop, an oversampled envelope's
    wrapped one, is refused beside a coupled flow by translation, on every backend. If
    this ever stops being refused, the crop needs a parity test here.
    """
    grid = EnvelopeGrid.from_limits(
        flow_parity.LENGTH, 800e-9, (400e-9, 1.6e-6), 300e-15, thg=True
    )
    assert np.any(np.diff(np.asarray(grid.crop_index)) != 1)
    simulation = build(
        "scalar",
        grid=grid,
        absorber=absorber(grid, boundary_length=1.0e3),
        envelope=True,
    )
    with pytest.raises(TranslationError, match="increasing"):
        simulation.translate()
