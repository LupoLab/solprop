# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native's phi-function moments against the independent augmented exponential.

``k! phi_{k+1}(x)`` is what every exponential Adams trial multiplies its history by.
``native/tests/native/test_adams_phi.cpp`` holds the block form to its own scalar form,
and nothing else holds either to an independent value, so this does: the reference's
augmented matrix exponential, at the same arguments and bound as
``test_exponential_weights.py``'s, among them a few ulp either side of the halving
thresholds, strong damping and resonance.

No binding returns the moments, so they are read through the pass that uses them,
one exponential trial per moment on the host's own lanes. The state is zero, so its
flow contributes nothing; the predictor is zero, so the source is the source of zero;
one history row holds exact ones and one corrector weight is one. The accepted field
at each bin is then exactly ``dz M_k(x)`` with ``dz = 1``, and the arguments ``x`` are
placed bin by bin in the operator's own table. This reaches the evaluator a production
step reaches, with its neighbours in the same block on other branches, and is not
weaker than calling the evaluator: no rounding is added between it and the reading.
"""

import math

import numpy as np
import pytest

from backends.test_exponential_weights import (
    ARGUMENTS,
    SQUARING_EDGES,
    augmented_phi,
    natural_scale,
)
from bounds import EPS
from solprop.backends.exponential import MAX_PHI_ORDER
from solprop.grid import EnvelopeGrid
from solprop.nonlinear.gnlse import problem_from
from solprop.pulses import SechPulse
from solprop.stepper import Multistep, PIController
from solprop.translate.problem import assemble

native = pytest.importorskip("solprop_native")

VALUES = np.array([*ARGUMENTS, *SQUARING_EDGES])
"""The arguments of ``test_exponential_weights.py``'s own kernel test."""

SLOTS = 8
"""History rows, the native limit; ``k = 0 .. 7`` are the moments' whole range."""


def _problem(grid):
    model = problem_from(
        [0.0, 0.0, -1e-26], omega_ref=grid.omega0, gamma=0.1, shock=False
    )
    field = SechPulse(wavelength=835e-9, width=100e-15, peak_power=1.0).build(grid)
    return assemble(
        grid,
        model.mode,
        [model.kerr],
        field,
        save_planes=[1e-3],
        density=1.0,
        shock=False,
        prefactor_index=1.0,
        controller=PIController(rtol=1e-8, atol=1e-16),
        dz_max=1e-3,
        dz_init=1e-4,
        integrator=Multistep(),
    )


def read_moments(module, grid, values):
    """Return ``M_k(x)`` of every value on active bins, ``(n_values, count)``.

    Parameters
    ----------
    module : module
        The ``solprop_native`` package.
    grid : solprop.grid.EnvelopeGrid
        A grid with at least one active bin per value.
    values : numpy.ndarray
        The arguments ``x``, complex.
    """
    problem, run_request = _problem(grid)
    active = np.flatnonzero(grid.active_mask)
    assert active.size >= values.size
    bins = active[: values.size]
    backend_module = module.backend
    backend = module.NativeBackend(problem, run_request, grid=grid)
    try:
        # The payload and the step's resident rows are the backend's own; the state
        # they open is this test's, so nothing is loaded into the backend's.
        payload = backend_module._native_problem(
            problem, run_request, backend._terms, None, None
        )
        arrays, material = backend._resident_arrays(0.0, 1e-3)
    finally:
        backend.close()
    terms = np.zeros_like(payload["phi_terms"])
    assert terms.shape[1] == 1
    terms[0, 0, bins] = values
    state = backend_module._core.State(payload | {"phi_terms": terms})
    try:
        (phi_delta, *rest) = arrays
        step = np.zeros_like(phi_delta)
        # The whole step's row, after the stages': one of the first term.
        step[-1, 0] = 1.0
        sample = np.zeros_like(phi_delta)
        n_omega = grid.n_omega
        zero = np.zeros(n_omega, dtype=complex)
        empty = np.empty(0, dtype=np.float64)
        width = SLOTS
        out = np.zeros((values.size, MAX_PHI_ORDER), dtype=complex)
        state.adams_allocate(SLOTS, False, False)
        store = np.zeros((SLOTS, 1, n_omega), dtype=complex)
        store[0] = 1.0
        for k in range(MAX_PHI_ORDER):
            state.set_initial(zero, np.empty(0, dtype=complex))
            state.adams_rows_from_host("store", store)
            corrector = np.zeros((width, width))
            corrector[1, k] = 1.0
            state.adams_exponential_trial(
                1.0,
                np.arange(width - 1, dtype=np.uint64),
                np.zeros(1),
                corrector.reshape(-1),
                np.zeros(2 * width),
                np.ones(n_omega),
                step,
                *rest,
                material,
                sample,
                *rest,
                material,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
            )
            state.adams_exponential_accept(
                1.0, width - 1, np.empty(0, dtype=np.uint64), False
            )
            out[:, k] = np.asarray(state.field_to_host()).reshape(-1)[bins]
    finally:
        state.close()
    return out


@pytest.fixture(scope="module")
def moments():
    """Return ``M_k(x)`` of every argument."""
    grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 1e-12)
    return read_moments(native, grid, VALUES)


@pytest.mark.parametrize(
    "index", range(VALUES.size), ids=[f"{one:.17g}" for one in VALUES]
)
def test_native_moments_agree_with_the_augmented_exponential(
    moments, index, record_property
):
    x = complex(VALUES[index])
    reference = augmented_phi(x, MAX_PHI_ORDER)
    margin = 0.0
    for k in range(MAX_PHI_ORDER):
        scale = math.factorial(k)
        error = abs(moments[index, k] - scale * reference[k + 1])
        # The bound of test_exponential_weights.py: each evaluator within tens of ulp
        # of the function's natural scale, the sum with a margin.
        limit = 256 * EPS * scale * natural_scale(x, k + 1)
        margin = max(margin, error / limit)
        assert error <= limit, k
    record_property("margin", margin)


def test_the_reading_is_not_of_a_constant(moments):
    """The reading follows its argument: ``phi_1(0)`` is 1, and at resonance it is 0."""
    at_zero = moments[0, :3]
    np.testing.assert_allclose(at_zero, [1.0, 0.5 * 1.0, 1.0 / 3.0], atol=8 * EPS)
    resonance = int(np.argmin(abs(VALUES - 2j * np.pi)))
    assert abs(moments[resonance, 0]) <= 8 * EPS / (2 * np.pi)
