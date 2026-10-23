# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The compiled coupled-flow kernels against the reference's own expressions.

Each kernel of :mod:`solprop.backends.numba_cpu.coupled_flow` is compared on random
operands with the arithmetic ``numpy_ref`` performs for the same pass -- ``einsum`` for
an action, ``tensordot`` for the residual's combination, and the expressions of the
reference's stage input, stage output, accumulation and accept passes around them,
written out on the two-dimensional layout the kernels take (the passes themselves are
annotated on the four-dimensional state). This module runs in the coverage lane's
interpreted pass (``poe cov-kernels``), so every kernel line is counted, and compiled in
the default lane.

**The bound, stated first**: one product of a flow action against an ``n``-vector is
within ``sqrt(2) (n + 2) u`` of exact relative to ``|U||x|`` on either side
(``tests/flow_parity.py``), so two implementations are within twice it, and a kernel
that also combines the stages or the residual's terms adds the length of that sum.
The operands here are unit-scale and well conditioned, so every comparison is taken
relative to the reference's result at ``2 sqrt(2) (n + 8) sqrt(n) u``, with the eight
covering the longest additional sum below (six previous stages, two operations of the
frame). Where the flow's kernels perform the diagonal route's arithmetic -- every bin
the block does not cover and every pass with no action or residual in it -- they are
compared with this backend's own **diagonal** kernels at the same bound, not at their
bits: two compiled kernels may contract differently (SEP 25). The pad, a copy and a
binary scaling, is exact.
"""

import math

import numpy as np
import pytest

from flow_parity import section_formation_bounds
from solprop.backends.numba_cpu import kernels
from solprop.backends.numba_cpu.coupled_flow import (
    _spectral_norm,
    flow_accept,
    flow_accumulate,
    flow_combine,
    flow_conditioning,
    flow_pad,
    form_flow_actions,
    make_flow_finish,
    residual_action,
)
from solprop.backends.numpy_ref.backend import _triangular_exponential
from solprop.translate.coupled import schur_blocks, schur_eigenbasis
from solprop.translate.stage import combine

N_MODES = 3
N_OMEGA = 12
N_STAGES = 7
UNIT = 2.0**-53
BOUND = 2.0 * math.sqrt(2.0) * (N_MODES + 8) * math.sqrt(N_MODES) * UNIT
"""The relative bound every compiled pass is held to against its reference form."""


def complex_normal(rng, *shape):
    """Return unit-scale complex normal samples."""
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.fixture
def operands():
    """Random operands of one small problem, half its bins on the block."""
    rng = np.random.default_rng(41)
    active = np.zeros(N_OMEGA, dtype=bool)
    active[::2] = True
    slots = np.full(N_OMEGA, -1, dtype=np.intp)
    slots[active] = np.arange(int(np.count_nonzero(active)), dtype=np.intp)
    n_active = int(np.count_nonzero(active))
    return {
        "rng": rng,
        "active": active,
        "slots": slots,
        "action": complex_normal(rng, n_active, N_MODES, N_MODES) / N_MODES,
        "state": complex_normal(rng, N_MODES, N_OMEGA),
        "stages": complex_normal(rng, N_STAGES, N_MODES, N_OMEGA),
        "propagator": np.exp(1j * rng.uniform(0, 2 * np.pi, (N_MODES, N_OMEGA))),
    }


def mixed(action, values, active):
    """Return the reference's ``_apply_flow`` on one state-shaped array."""
    carried = values.copy()
    carried[:, active] = np.einsum("wij,jw->iw", action, values[:, active])
    return carried


def stage_input(state, stage_sum, dz, propagator):
    """Return ``passes.stage_input``'s expression, with no noise."""
    return propagator * (state + dz * stage_sum)


def stage_output(polarisation, prefactor, window, inverse, absorber):
    """Return ``passes.stage_output``'s expression."""
    normalised = prefactor * window * polarisation
    if absorber is not None:
        normalised = normalised + absorber
    return inverse * normalised


def accumulate(total, stage, weight):
    """Return ``passes.accumulate``'s expression."""
    return total + weight * stage


def diagonal_combine(state, stages, weights, dz, propagator):
    """Return the diagonal route's compiled stage input, on the coarse layout."""
    lab = np.zeros_like(state)
    kernels.make_combine_and_pad(contiguous=True, noise=False)(
        state,
        stages,
        weights,
        weights.size,
        dz,
        propagator,
        state,
        np.arange(state.shape[1], dtype=np.intp),
        lab,
        1.0,
    )
    return lab


def relative(got, expected):
    """Return the relative L2 departure."""
    return float(np.linalg.norm(got - expected) / np.linalg.norm(expected))


@pytest.mark.parametrize("mix", [True, False])
def test_the_combination_is_the_reference_s_stage_input(operands, mix):
    """``exp(dPhi) (U a + dz U sum a_j k_j)``, the action applied to each part."""
    state, stages, action = operands["state"], operands["stages"], operands["action"]
    active, slots = operands["active"], operands["slots"]
    weights = np.array([0.2, 0.3, -0.4, 0.7, 0.1])
    dz = 1e-5
    lab = np.zeros_like(state)
    flow_combine(
        state,
        stages,
        weights,
        weights.size,
        dz,
        operands["propagator"],
        action,
        slots,
        mix,
        lab,
    )
    combination = np.zeros_like(state)
    for index, weight in enumerate(weights):
        combination = accumulate(combination, stages[index], float(weight))
    if mix:
        expected = stage_input(
            mixed(action, state, active),
            mixed(action, combination, active),
            dz,
            operands["propagator"],
        )
        assert relative(lab, expected) <= BOUND
    else:
        expected = stage_input(state, combination, dz, operands["propagator"])
        assert relative(lab, expected) <= BOUND
    # Off the block, and everywhere at the zero fraction, the arithmetic is the
    # diagonal route's compiled combination, to its rounding.
    diagonal = diagonal_combine(state, stages, weights, dz, operands["propagator"])
    covered = np.zeros(N_OMEGA, dtype=bool) if not mix else active
    assert relative(lab[:, ~covered], diagonal[:, ~covered]) <= BOUND


@pytest.mark.parametrize("contiguous", [True, False])
def test_the_pad_writes_every_bin_and_scales_exactly(operands, contiguous):
    """Every fine bin is written: the data where the crop puts it, zero elsewhere."""
    state = operands["state"]
    fine = np.full((N_MODES, 2 * N_OMEGA), np.nan + 0j)
    crop = (
        np.arange(N_OMEGA, dtype=np.intp)
        if contiguous
        else np.arange(2 * N_OMEGA - 1, 0, -2, dtype=np.intp)
    )
    flow_pad(state, crop, fine, 0.25)
    expected = np.zeros_like(fine)
    expected[:, crop] = state * 0.25
    # Structural: a copy, a binary scaling and zeros, none of which rounds.
    assert np.array_equal(fine, expected)


def test_the_residual_is_the_reference_s_combination_minus_the_frozen_block(
    operands,
):
    """``combine(w, T) - G_s``, a zero weight adding an exact zero."""
    rng, action = operands["rng"], operands["action"]
    terms = complex_normal(rng, 6, *action.shape)
    frozen = complex_normal(rng, *action.shape)
    weights = np.array([0.0, 0.0, 1.0, -0.5, 0.25, 0.0])
    out = np.zeros_like(frozen)
    residual_action(weights, terms, frozen, out)
    expected = combine(weights, terms) - frozen
    assert relative(out, expected) <= BOUND
    # A single term of weight one is the stored term itself on both sides.
    residual_action(np.array([1.0]), terms[:1], frozen, out)
    assert relative(out, combine(np.array([1.0]), terms[:1]) - frozen) <= BOUND


@pytest.mark.parametrize("absorber", [True, False])
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_the_finish_is_the_reference_s_stage_output_plus_the_residual(
    operands, absorber, residual, contiguous
):
    """``exp(-dPhi) normalised + exp(-dPhi) R lab``, in the reference's order."""
    rng, active, slots = operands["rng"], operands["active"], operands["slots"]
    block = complex_normal(rng, *operands["action"].shape)
    lab = complex_normal(rng, N_MODES, N_OMEGA)
    crop = (
        np.arange(N_OMEGA, dtype=np.intp)
        if contiguous
        else np.arange(2 * N_OMEGA - 1, 0, -2, dtype=np.intp)
    )
    spectrum = complex_normal(rng, N_MODES, 2 * N_OMEGA)
    prefactor = complex_normal(rng, N_MODES, N_OMEGA)
    window = rng.uniform(0.5, 1.0, N_OMEGA)
    inverse = np.conj(operands["propagator"])
    damping = complex_normal(rng, N_MODES, N_OMEGA)
    drive = np.zeros_like(lab)
    finish = make_flow_finish(
        contiguous=contiguous, absorber=absorber, residual=residual
    )
    finish(
        spectrum, crop, prefactor, window, inverse, damping, lab, block, slots, drive
    )
    expected = stage_output(
        spectrum[:, crop],
        prefactor,
        window,
        inverse,
        damping if absorber else None,
    )
    if residual:
        term = np.zeros_like(lab)
        term[:, active] = np.einsum("wij,jw->iw", block, lab[:, active])
        expected = expected + inverse * term
        assert relative(drive, expected) <= BOUND
    else:
        assert relative(drive, expected) <= BOUND
        # No residual: the diagonal route's own compiled finish, every bin.
        stage = np.zeros_like(drive)
        kernels.make_finish_stage(contiguous=contiguous, absorber=absorber)(
            spectrum,
            crop,
            prefactor,
            window,
            inverse,
            damping,
            0.0,
            0.0,
            stage,
            np.zeros_like(drive),
            np.zeros_like(drive),
        )
        assert relative(drive, stage) <= BOUND


@pytest.mark.parametrize("mix", [True, False])
def test_the_return_is_the_reference_s_inverse_action_and_accumulation(operands, mix):
    """``U^-1 drive`` and both accumulations, ``total + weight * stage``."""
    rng, action = operands["rng"], operands["action"]
    active, slots = operands["active"], operands["slots"]
    drive = complex_normal(rng, N_MODES, N_OMEGA)
    solution = complex_normal(rng, N_MODES, N_OMEGA)
    error = complex_normal(rng, N_MODES, N_OMEGA)
    stage = np.zeros_like(drive)
    got_solution, got_error = solution.copy(), error.copy()
    flow_accumulate(
        drive, action, slots, mix, 0.3, -0.1, stage, got_solution, got_error
    )
    expected = mixed(action, drive, active) if mix else drive
    if mix:
        assert relative(stage, expected) <= BOUND
    else:
        # Structural: with no action the drive is copied.
        assert np.array_equal(stage, expected)
    assert relative(got_solution, accumulate(solution, stage, 0.3)) <= BOUND
    assert relative(got_error, accumulate(error, stage, -0.1)) <= BOUND


def test_the_accept_pass_is_the_reference_s_through_the_whole_step_s_action(
    operands,
):
    """``exp(dPhi) (U a + dz U s)``, the norm's partials, and the mixed error kept."""
    rng, action = operands["rng"], operands["action"]
    active, slots = operands["active"], operands["slots"]
    state, propagator = operands["state"], operands["propagator"]
    solution = complex_normal(rng, N_MODES, N_OMEGA)
    error = complex_normal(rng, N_MODES, N_OMEGA)
    mask = np.ones(N_OMEGA, dtype=bool)
    mask[-3:] = False
    dz = 1e-5
    advanced = np.zeros_like(state)
    mixed_error = np.zeros_like(state)
    partials = np.zeros((3, N_MODES))
    flow_accept(
        state,
        propagator,
        dz,
        solution,
        error,
        action,
        slots,
        mask,
        advanced,
        mixed_error,
        partials,
    )
    # ``passes.accept_step`` is ``stage_input``'s expression over the whole step.
    expected = stage_input(
        mixed(action, state, active), mixed(action, solution, active), dz, propagator
    )
    assert relative(advanced, expected) <= BOUND
    assert relative(mixed_error, mixed(action, error, active)) <= BOUND
    scaled = propagator * (dz * mixed(action, error, active))
    reference = (
        np.sum(abs(state[:, mask]) ** 2, axis=1),
        np.sum(abs(expected[:, mask]) ** 2, axis=1),
        np.sum(abs(scaled[:, mask]) ** 2, axis=1),
    )
    for got, want in zip(partials, reference, strict=True):
        assert relative(got, want) <= BOUND
    # The old norm is of the state before the action, as the reference's.
    before = np.zeros(N_MODES)
    for column, row in enumerate(state):
        for value in row[mask]:
            before[column] += value.real * value.real + value.imag * value.imag
    assert relative(partials[0], before) <= BOUND


# ------------------------------------------------------------ matrix functions


@pytest.mark.parametrize("length", [2e-4, 3e-3])
def test_the_formed_actions_are_the_reference_s_on_either_route(length):
    """Both routes of ``form_flow_actions`` against ``Z exp(+-l T) Z^H``.

    Bound, stated first: the formation bound of ``tests/flow_parity.py``
    (``section_formation_bounds``) for each bin at each length, with the bin's own
    certificate ``exp(l (mu_max - mu_min))``, for the route the bin takes; forcing
    every bin to scaling and squaring bounds it by twice the reference's. Random
    blocks of this scale have ``||l T||`` between 0.3 and 5, so the reference squares
    zero to four times; the longer length exercises the squarings.
    """
    rng = np.random.default_rng(6)
    n_active = 5
    generator = complex_normal(rng, N_MODES, N_MODES, n_active) * 300.0
    vectors, triangular = schur_blocks(generator)
    left, right, eigenvalues, route = schur_eigenbasis(vectors, triangular)
    fractions = np.array([0.0, 0.5, 1.0])
    blocks = np.moveaxis(generator, -1, 0)
    hermitian = np.linalg.eigvalsh(
        0.5 * (blocks + np.conj(np.swapaxes(blocks, -1, -2)))
    )
    certificate = np.exp(length * (hermitian[:, -1] - hermitian[:, 0]))
    forward = np.zeros((3, n_active, N_MODES, N_MODES), dtype=complex)
    inverse = np.zeros_like(forward)
    exponents = np.zeros((2, N_MODES), dtype=complex)
    work = np.zeros((3, N_MODES, N_MODES), dtype=complex)
    adjoint = np.conj(np.swapaxes(vectors, -1, -2))
    for taken in (route, np.zeros_like(route)):
        form_flow_actions(
            fractions,
            length,
            left,
            right,
            eigenvalues,
            taken,
            vectors,
            triangular,
            forward,
            inverse,
            exponents,
            work,
        )
        assert not np.any(forward[0]) and not np.any(inverse[0])
        lengths = fractions[1:] * length
        bounds = section_formation_bounds(
            vectors, triangular, lengths, certificate, route=taken
        )
        for slot in (1, 2):
            exponent = fractions[slot] * length * triangular
            for got, sign in ((forward[slot], 1.0), (inverse[slot], -1.0)):
                expected = vectors @ _triangular_exponential(sign * exponent) @ adjoint
                error = np.linalg.norm(got - expected, ord=2, axis=(-2, -1))
                scale = np.linalg.norm(expected, ord=2, axis=(-2, -1))
                assert np.all(error / scale <= bounds[slot - 1])


def test_the_spectral_norm_bounds_the_singular_value_to_its_rounding():
    """``_spectral_norm`` against the largest singular value, from above in effect.

    Bound, stated first: the Gram matrix is within ``n gamma`` of ``U^H U`` relative
    to its norm and each rotation within a few ``u``, so the square root is within
    ``8 n u`` of ``sigma_max`` on these blocks; a block whose Gram matrix has a zero
    pivot (a coupling of channels 0 and 2 alone) takes the skipped rotation, and a
    diagonal one no rotation at all.
    """
    rng = np.random.default_rng(8)
    gram = np.zeros((N_MODES, N_MODES), dtype=complex)
    blocks = [complex_normal(rng, N_MODES, N_MODES) for _ in range(6)]
    coupled = np.eye(N_MODES, dtype=complex)
    coupled[0, 2] = 0.3 + 0.1j
    blocks += [coupled, np.diag([1.0, 2.0, 0.5]).astype(complex)]
    for block in blocks:
        expected = float(np.linalg.norm(block, ord=2))
        assert abs(_spectral_norm(block, gram) / expected - 1.0) <= 8 * N_MODES * UNIT
    forward = np.stack(blocks)
    inverse = np.stack([np.linalg.inv(block) for block in blocks])
    measured = np.zeros(len(blocks))
    flow_conditioning(forward, inverse, gram, measured)
    expected = np.linalg.norm(forward, ord=2, axis=(-2, -1)) * np.linalg.norm(
        inverse, ord=2, axis=(-2, -1)
    )
    assert np.all(abs(measured / expected - 1.0) <= 16 * N_MODES * UNIT)


def test_the_threaded_flow_passes_are_the_serial_ones():
    """The parallel variants of the three state passes, to a few roundings per bin.

    Each bin of the combination and the accumulation, and each column of the accept,
    is the serial arithmetic run by another thread; with contraction either may round
    a product once where the other rounds twice, so each output is held to ``8 eps`` of
    the largest magnitude, and each norm partial (a sum of ``n_omega`` squares) to
    ``2 gamma(n_omega + 2)`` (SEP 25 section 4.1).
    """
    from bounds import EPS, assert_within, higham_gamma  # noqa: PLC0415
    from solprop.backends.numba_cpu import coupled_flow  # noqa: PLC0415

    n_columns, n_omega = 4, coupled_flow.FLOW_ELEMENTS // 4
    serial = coupled_flow.flow_kernels(1, n_columns, n_omega)
    threaded = coupled_flow.flow_kernels(2, n_columns, n_omega)
    assert serial == (flow_combine, flow_accumulate, flow_accept)
    # Compiled only: under NUMBA_DISABLE_JIT both variants are the one plain function.
    if hasattr(flow_combine, "py_func"):
        assert threaded[0] is not flow_combine
    assert coupled_flow.flow_kernels(6, n_columns, n_omega // 2 - 1) == serial
    rng = np.random.default_rng(5)

    def draw(*shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    state, propagator = draw(n_columns, n_omega), draw(n_columns, n_omega)
    stages = draw(7, n_columns, n_omega)
    weights = rng.random(6)
    action = draw(n_omega, n_columns, n_columns)
    slots = np.where(np.arange(n_omega) % 5 == 0, -1, np.arange(n_omega))
    mask = np.arange(n_omega) % 9 != 0
    outputs = []
    for combine_, accumulate_, accept_ in (serial, threaded):
        lab = np.empty_like(state)
        combine_(state, stages, weights, 6, 1e-3, propagator, action, slots, True, lab)
        stage, solution, error = (
            np.empty_like(state),
            np.ones_like(state),
            np.ones_like(state),
        )
        accumulate_(state, action, slots, True, 0.3, 0.1, stage, solution, error)
        advanced, mixed = np.empty_like(state), np.empty_like(state)
        partials = np.empty((3, n_columns))
        accept_(
            state,
            propagator,
            1e-3,
            solution,
            error,
            action,
            slots,
            mask,
            advanced,
            mixed,
            partials,
        )
        outputs.append((lab, stage, solution, advanced, mixed, partials))
    names = ("lab", "stage", "solution", "advanced", "mixed error")
    for name, mine, theirs in zip(names, outputs[1][:5], outputs[0][:5], strict=True):
        assert_within(mine, theirs, 8 * EPS, what=name)
    assert_within(
        outputs[1][5], outputs[0][5], 2 * higham_gamma(n_omega + 2), what="partials"
    )
