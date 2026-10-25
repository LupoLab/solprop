# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Adams passes of ``numba_cpu`` with their rows in parallel (SEP 28).

:func:`solprop.backends.numba_cpu.multistep.make_passes` builds every diagonal pass of
both formulations in a serial and a parallel variant; the backend takes the parallel
one for a large problem. Each row's arithmetic is the same in both and each defect's
norm is summed from one partial per row in both, so the variants agree to the rounding
of one element's operations, and a march on either reaches the same field within the
rounding the routes' own parity allows. A separable route's exponent is formed by the
propagator kernel's element formula, which must be the host's volume to rounding.
"""

import numpy as np
import pytest

from bounds import EPS, assert_within
from capability_routes import ROUTES
from solprop.backends.numba_cpu import NumbaCpuBackend, fft
from solprop.backends.numba_cpu.multistep import (
    NumbaAdamsKernels,
    NumbaExponentialKernels,
    make_passes,
)
from solprop.stepper import PIController
from solprop.stepper.multistep import Multistep
from solprop.translate.linop import delta_phi

SHAPE = (1, 48, 2, 37)
"""A state of 96 rows: enough for each thread of the parallel variant to own several."""

ELEMENT_BOUND = 64 * EPS
"""One element's arithmetic, a few dozen operations, contracted differently at most."""


def _complex(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.fixture
def arrays():
    """Return a state, a five-row history and the trial's per-bin data."""
    rng = np.random.default_rng(28)
    return {
        "state": _complex(rng, SHAPE),
        "propagator": np.exp(-0.1 * _complex(rng, SHAPE)),
        "source": _complex(rng, SHAPE),
        "store": _complex(rng, (5, *SHAPE)),
        # Magnitudes across both sides of the phi functions' series/recurrence join.
        "exponent": 3.0 * _complex(rng, SHAPE),
        "mask": rng.random(SHAPE[-1]) > 0.2,
        "divisor": 1.0 + rng.random(SHAPE[-1]),
        "slots": np.array([3, 0, 4], dtype=np.int64),
    }


def test_the_parallel_variant_is_its_own_closure_and_memoised():
    serial, parallel = make_passes(parallel=False), make_passes(parallel=True)
    assert make_passes(parallel=True) is parallel
    assert serial is not parallel
    assert NumbaAdamsKernels()._passes is serial
    assert NumbaExponentialKernels(parallel=True)._passes is parallel
    coupled = NumbaExponentialKernels(parallel=True).coupled()
    assert isinstance(coupled, NumbaExponentialKernels)
    assert coupled._passes is parallel


def test_the_interaction_passes_agree_in_both_variants(arrays):
    weights = np.array([0.6, -0.3, 0.2])
    corrector = np.array([0.4, 0.3, -0.2, 0.1])
    rows = np.array([[0.5, -0.5, 0.25, 0.0], [1.0, -2.0, 1.0, 0.5]])
    results = []
    for parallel in (False, True):
        kernels = NumbaAdamsKernels(parallel=parallel)
        predicted = kernels.predict(
            arrays["state"],
            arrays["propagator"],
            arrays["store"],
            arrays["slots"],
            weights,
            1e-3,
        )
        corrected, norms = kernels.correct(
            arrays["state"],
            arrays["propagator"],
            arrays["source"],
            arrays["store"],
            arrays["slots"],
            corrector,
            rows,
            1e-3,
            arrays["mask"],
            arrays["divisor"],
        )
        store = arrays["store"].copy()
        kernels.transport(
            store, arrays["slots"][:2], arrays["propagator"], 1, arrays["source"]
        )
        results.append((predicted, corrected, norms, store))
    for name, serial, parallel in zip(
        ("predict", "correct", "norms", "transport"), *results, strict=True
    ):
        assert_within(parallel, serial, ELEMENT_BOUND, what=name)


@pytest.mark.parametrize("extra", [False, True])
def test_the_exponential_passes_agree_in_both_variants(arrays, extra):
    rng = np.random.default_rng(16)
    count = 3
    coefficients = rng.standard_normal((3, count))
    corrector = rng.standard_normal((4, count))
    rows = rng.standard_normal((2, 4 + count))
    compensation = (
        (_complex(rng, SHAPE), [_complex(rng, SHAPE), _complex(rng, SHAPE)])
        if extra
        else None
    )
    results = []
    for parallel in (False, True):
        kernels = NumbaExponentialKernels(parallel=parallel)
        moments = kernels.moments(arrays["exponent"], count).copy()
        flow = kernels.flow(arrays["exponent"]).copy()
        predicted = kernels.predict(
            arrays["state"],
            flow,
            moments,
            arrays["store"],
            arrays["slots"],
            coefficients,
            1e-3,
        )
        corrected, norms = kernels.correct(
            arrays["state"],
            flow,
            moments,
            arrays["source"],
            arrays["store"],
            arrays["slots"],
            corrector,
            rows,
            1e-3,
            arrays["mask"],
            arrays["divisor"],
            compensation,
        )
        carried = kernels.carry(
            arrays["exponent"],
            1e-3,
            arrays["store"],
            arrays["store"][::-1],
            arrays["slots"],
            np.array([0.0, 0.5, 1.0]),
        )
        results.append((moments, flow, predicted, corrected, norms, carried))
    names = ("moments", "flow", "predict", "correct", "norms", "carry")
    for name, serial, parallel in zip(names, *results, strict=True):
        assert_within(parallel, serial, ELEMENT_BOUND, what=name)


def _cartesian(formulation: str):
    simulation = ROUTES["cartesian"](None)
    settings = simulation.settings
    simulation.settings = type(settings)(
        **{
            **{name: getattr(settings, name) for name in settings.__dataclass_fields__},
            "controller": PIController(rtol=1e-8, atol=settings.controller.atol),
            "integrator": Multistep(formulation=formulation),  # pyright: ignore[reportArgumentType]
        }
    )
    return simulation, simulation.translate()


def test_a_separable_exponent_is_the_host_volume():
    simulation, problem = _cartesian("exponential")
    backend = NumbaCpuBackend(problem, grid=simulation.grid)
    try:
        bundle = backend._stage_scalars(1e-6, 4e-6)
        compiled = backend._delta_phi(bundle)
    finally:
        backend.close()
    # One element formula, a square root and a few products, in either order.
    assert_within(
        compiled, delta_phi(problem.linear, 1e-6, 4e-6), ELEMENT_BOUND, what="exponent"
    )


@pytest.mark.medium
@pytest.mark.parametrize("formulation", ["interaction", "exponential"])
def test_a_large_problem_marches_on_the_parallel_passes(
    thread_budget_lifted, formulation, monkeypatch
):
    simulation, problem = _cartesian(formulation)

    def march(threads: int):
        backend = NumbaCpuBackend(
            problem, grid=simulation.grid, threads=threads, regime="large"
        )
        try:
            march = backend._adams
            assert march is not None
            # The interaction formulation's march holds no exponential passes.
            held = (
                march._exponential if formulation == "exponential" else march._kernels
            )
            passes = held._passes  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
            backend.set_initial(problem.initial.field0, None)
            for plane in problem.stepper.save_planes:
                backend.propagate_to(float(plane))
            return passes, backend.observe().require_field().copy(), march.ledger.total
        finally:
            backend.close()

    serial_passes, serial, serial_calls = march(1)
    # Every problem counts as large here, so only the thread count decides.
    monkeypatch.setattr(fft, "REGIME_ELEMENTS", 1)
    parallel_passes, parallel, parallel_calls = march(2)
    assert serial_passes is make_passes(parallel=False)
    assert parallel_passes is make_passes(parallel=True)
    # The same decisions, and fields within the free-space routes' parity with the
    # reference (test_exponential_routes: measured to 5e-14 of the peak, bound 3e-9):
    # the variants differ by rounding alone, which the march carries forward.
    assert parallel_calls == serial_calls
    assert_within(parallel, serial, 3e-9, scale=np.max(np.abs(serial)), what="field")
