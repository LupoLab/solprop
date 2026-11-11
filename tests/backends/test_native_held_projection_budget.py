# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native's embedded-rule current projection, accumulated over partial chunks.

SEP 27 section 4.6. The two coarse projections of the held current, ``g J`` and
``g' J``, accumulate over the node chunks: an exact chunk of three, the original's, on
this case's nine nodes, and four, which leaves a partial last chunk. The allowance is
the column's, applied to ``J`` before either collar multiplies it and then projected
with absolute weights, so a tiny trailing-edge residual is not judged against the large
oscillating current whose cancellation produced it.

**Weaker than the original.** Native exposes neither ``J`` nor the two coarse streams
in the time domain. What it exposes is their combination after the forward transform
and the reciprocals of ``omega``, ``projected_to_host(True)``, which the reference
forms with ``_held_streams``. So the elementwise allowances are carried to each bin,
``l1 -> linf``, with the transform's normwise rounding on the magnitudes it reads. The
check on ``J`` itself is not reachable: an error there shows only through the
projection, and only against the sum over a row's samples.
``test_current_assembly_and_ledger_on_each_route`` still holds the fine stream at the
stage.
"""

import dataclasses

import numpy as np
import pytest

from backends.test_band_plasma_parity import bound, case
from backends.test_radial_nonlinear import QuadraticRate
from bounds import UNIT_ROUNDOFF, fft_bound, gamma
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.backends.session import ExecutionOptions
from solprop.translate.ionisation import build_rate_table
from solprop.translate.plasma_integration import design_running_integral
from solprop.translate.responses import ResponseStage

native = pytest.importorskip("solprop_native")

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
]


def _trial(backend, run_request, field, dz):
    """Load `field` into native's state and take one trial from z = 0 on its staged
    rows, the state's own (SEP 47 phase 5 retired the backend's driver calls)."""
    state = backend._native_state
    flat = np.ascontiguousarray(field, dtype=np.complex128).reshape(-1)
    state.set_initial(flat, np.empty(0, dtype=np.complex128))
    state.clear_fsal()
    backend._staged_rows()
    nodes = [float(node) for node in run_request.stepper.tableau.nodes]
    state.rk_trial(0.0, dz, nodes, prepare_only=False)


def current_only(problem, grid, rule):
    """Return multimode plasma on its own, both streams populated.

    The small He fixture has exactly zero tabulated ionisation; the analytic rate
    populates both projected streams, so accumulation cannot pass by comparing two
    empty arrays. The Kerr stage goes, as it would add its own coarse spectrum to the
    one spectrum native exposes.
    """
    rate = build_rate_table(QuadraticRate(), (1e3, 1e10))
    rate.log_rate.setflags(write=False)
    rate.coefficients.setflags(write=False)
    responses = problem.responses
    return dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            responses,
            plasma_integral=design_running_integral(grid, rule),
            ionisation_table=(rate,) * len(responses.plasma_channels),
            plasma_channels=tuple(
                dataclasses.replace(channel, rate=rate)
                for channel in responses.plasma_channels
            ),
            kerr_coefficients=None,
            response_order=tuple(
                stage
                for stage in responses.response_order
                if stage is not ResponseStage.KERR
            ),
        ),
    )


@pytest.mark.parametrize("chunk", [3, 4])
@pytest.mark.parametrize("rule", ["band", "trapezoid"])
def test_embedded_current_projection_accumulates_partial_chunks(
    rule, chunk, record_property
):
    grid, problem, run_request = case("multimode")
    assert problem.transform.projection_coarse is not None
    problem = current_only(problem, grid, rule)
    options = ExecutionOptions(node_chunk=chunk)
    relative = bound(problem, grid, "native", 0)
    assert relative > 0
    reference = NumpyRefBackend(problem, run_request, grid=grid, options=options)
    try:
        reference._hoist(0.0)
        state = problem.initial.field0
        bundle = reference._stage_scalars(0.0, 0.0)
        transform, grids = problem.transform, problem.grids
        nodes = passes.synthesise(
            passes.to_time(state, grids.crop_index, grids.n_to, kind="field"),
            transform.synthesis,
            bundle.taper_scale,
            transform.n_pol,
        )
        current = reference._column_sources(nodes, bundle, None).current
        assert current is not None
        # SEP 27 section 4.6: the existing column allowance on J before either collar
        # multiplies it, then projected with absolute weights.
        current_limit = relative * np.max(abs(current), axis=-1, keepdims=True)
        rounding = gamma(2 * transform.n_nodes * transform.n_pol + 4)
        collars = (grids.t_window_fine, grids.t_window_fine_derivative)
        limits = [
            passes.project(
                abs(collar) * current_limit + rounding * abs(collar * current),
                abs(transform.projection_coarse),
                bundle.taper_scale,
                transform.n_pol,
            )
            for collar in collars
        ]
        assert all(np.max(limit) > 0 for limit in limits)
        magnitudes = [
            passes.project(
                abs(collar * current),
                abs(transform.projection_coarse),
                bundle.taper_scale,
                transform.n_pol,
            )
            for collar in collars
        ]
        reference._nonlinear(state[None], bundle, np.ones_like(state[None]))
        assert reference._last_plasma is not None
        _, _, coarse_gate, coarse_edge = reference._last_plasma
        assert coarse_gate is not None and coarse_edge is not None
        assert np.max(abs(coarse_gate)) > 0 and np.max(abs(coarse_edge)) > 0
        expected = reference._held_streams(coarse_gate, coarse_edge)
    finally:
        reference.close()
    # The forward transform: l1 -> linf of each row's allowance, with its own normwise
    # rounding on what it reads; then each side's reciprocal, its square and the sum.
    inverse = np.zeros_like(grids.omega)
    np.divide(1.0, grids.omega, out=inverse, where=grids.active_mask)
    per_bin = [
        np.sum(limit, axis=-1)
        + fft_bound(grids.n_to) * np.sqrt(grids.n_to) * np.linalg.norm(mag, axis=-1)
        + operations * UNIT_ROUNDOFF * np.sum(mag, axis=-1)
        for limit, mag, operations in zip(limits, magnitudes, (8, 12), strict=True)
    ]
    allowed = per_bin[0][..., None] * abs(inverse) + per_bin[1][..., None] * inverse**2
    backend = native.NativeBackend(problem, run_request, grid=grid, options=options)
    try:
        _trial(backend, run_request, state, 0.0)
        raw = np.asarray(backend._native_state.projected_to_host(True))
    finally:
        backend.close()
    actual = raw.reshape(*expected.shape[:-1], grids.n_to // 2 + 1)[
        ..., grids.crop_index
    ]
    scale = float(np.max(abs(expected)))
    assert scale > 0
    record_property("measured", float(np.max(abs(actual - expected))) / scale)
    record_property("bound", float(np.max(allowed)) / scale)
    assert np.all(abs(actual - expected) <= allowed)
