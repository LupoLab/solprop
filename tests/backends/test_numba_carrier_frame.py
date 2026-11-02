# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""``numba_cpu``'s carrier frame: its compiled passes and its march (SEP 35)."""

import numpy as np
import pytest

from backends.test_carrier_frame import (
    SCALAR,
    capillary,
    manufactured,
    manufactured_budget,
    march,
    reference_rate,
)
from bounds import EPS, carrier_increment_budget
from solprop.backends.multistep import ExponentialKernels
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numba_cpu.multistep import (
    _per_row,  # pyright: ignore[reportPrivateUsage]
    _per_slot_row,  # pyright: ignore[reportPrivateUsage]
)
from solprop.exceptions import BackendError
from solprop.stepper.multistep import lagrange_coefficients


def test_the_compiled_passes_take_one_shift_per_row():
    with pytest.raises(BackendError, match="one carrier shift per batch member"):
        _per_row(np.zeros((1, 1, 1, 4), dtype=complex), (1, 2, 1, 4))
    with pytest.raises(BackendError, match="one carrier rotation per row"):
        _per_slot_row(np.ones((2, 1, 1, 1, 4), dtype=complex), (1, 2, 1, 4))


def one_trial_inputs(seed=35):
    """Return a random trial: exponent, shift, state, store, slots and coefficients."""
    rng = np.random.default_rng(seed)
    shape = (2, 2, 1, 64)
    exponent = rng.normal(size=shape) * 0.2 - 0.05 + 1j * rng.uniform(-30, 30, shape)
    # One shift per batch member, from its own reference element, under the cap.
    shift = (1j * np.array([0.8, -0.6])).reshape((2, 1, 1, 1))
    state = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    store = rng.normal(size=(7, *shape)) + 1j * rng.normal(size=(7, *shape))
    nodes = [0.0, -0.7, -1.5, -2.6]
    slots = np.array([3, 0, 5, 1], dtype=np.int64)
    return exponent, shift, state, store, nodes, slots


def test_the_compiled_passes_follow_the_reference_within_the_budget():
    # One trial's moments, flow, predictor and corrector on numba_cpu against the
    # reference, from the same inputs: within twice the increment's rounding budget
    # (bounds.carrier_increment_budget) plus the flow's few ulp of the state.
    from solprop.backends.multistep import (  # noqa: PLC0415 - the private helpers
        _carrier_factors,  # pyright: ignore[reportPrivateUsage]
    )
    from solprop.backends.numba_cpu.multistep import (  # noqa: PLC0415
        NumbaExponentialKernels,
    )

    exponent, shift, state, store, nodes, slots = one_trial_inputs()
    order = 3
    width = len(slots) + 1
    reference, compiled = ExponentialKernels(), NumbaExponentialKernels()
    expected = reference.moments(exponent, width, shift=shift)
    found = compiled.moments(exponent, width, shift=shift)
    np.testing.assert_allclose(found, expected, rtol=0, atol=512 * EPS)
    # The flow is the unshifted step's, formed beside the shifted moments.
    np.testing.assert_allclose(compiled.flow(exponent), np.exp(exponent), rtol=4 * EPS)
    factors = _carrier_factors(nodes, len(slots), shift)
    coefficients = lagrange_coefficients(nodes[:order])
    predicted = (
        reference.predict(
            state,
            np.exp(exponent),
            expected,
            store,
            slots[:order],
            coefficients,
            1.0,
            factors=factors[:order],
        ),
        compiled.predict(
            state,
            compiled.flow(exponent),
            found,
            store,
            slots[:order],
            coefficients,
            1.0,
            factors=factors[:order],
        ),
    )
    samples = np.stack([store[slot] for slot in slots[:order]])
    budget = 2 * carrier_increment_budget(
        coefficients, expected, samples, nodes[:order], shift, exponent - shift
    ) + 8 * EPS * np.abs(np.exp(exponent) * state)
    assert np.all(np.abs(predicted[1] - predicted[0]) <= budget)
    # Unrotated rows are a different sum: the factors are applied, not ignored.
    unrotated = compiled.predict(
        state, compiled.flow(exponent), found, store, slots[:order], coefficients, 1.0
    )
    assert np.max(np.abs(unrotated - predicted[1])) > 1e3 * np.max(budget)


def test_numba_cpu_steps_the_frame_exactly():
    # The manufactured carrier-following source through the compiled march: exact to
    # rounding, dense planes and the cap included, in the reference's calls. (The
    # native engine evaluates its source itself, so a host substitute cannot reach it;
    # its dense planes are compared with the reference's below.)
    simulation = capillary(SCALAR)
    source = manufactured(simulation.translate())
    expected = march(simulation, source)
    compiled: type = NumbaCpuBackend
    run = march(simulation, source, backend_class=compiled)
    assert max(run.errors) <= manufactured_budget(run)
    assert run.march.ledger.total == expected.march.ledger.total
    assert np.max(run.steps) * abs(reference_rate(run.problem)) <= 1 + 1e-9
