# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The mlx backend against its oracle, numpy_ref's single-precision path (SEP 33 7.2).

Class R throughout: both sides replay the same fixed schedule with the same method in
binary32, and the bounds are the oracle tests' own formulas. On the Metal GPU every
protected operation runs through its custom kernel; on MLX's CPU device, which
continuous integration uses when the runner exposes no Metal device, the graph code
runs and the kernels' tests are skipped. The local GPU lane is the whole module on the
Metal device.
"""

import dataclasses
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from typing import cast

import numpy as np
import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx

from backends.test_single_precision import capillary, gnlse
from bounds import EPS_32, UNIT_ROUNDOFF_32, assert_within, higham_gamma
from solprop.backends.memory import estimate_memory
from solprop.backends.mlx_gpu.arithmetic import (
    DoubleFloatTerms,
    gemm_probe,
    on_gpu,
    population,
    prefix_scan,
)
from solprop.backends.mlx_gpu.backend import MlxBackend
from solprop.backends.mlx_gpu.passes import DeviceRateTable
from solprop.backends.numpy_ref.single import NumpySingleBackend
from solprop.backends.protocol import Backend
from solprop.backends.registry import available, select, select_batch
from solprop.driver import Simulation
from solprop.exceptions import BackendError, ConvergenceError
from solprop.stepper import LowStorageRungeKutta
from solprop.translate.precision import SINGLE, prefix_sum
from vector_raman_cases import vector_problem

U32 = UNIT_ROUNDOFF_32

# MLX's CPU device where the runner has no Metal GPU, or where it is asked for; the
# kernels' tests then skip.
if not mx.metal.is_available() or os.environ.get("SOLPROP_MLX_DEVICE") == "cpu":
    mx.set_default_device(mx.cpu)

gpu_only = pytest.mark.skipif(
    not on_gpu(), reason="a custom Metal kernel runs on the Metal device only"
)


def on(backend: str, simulation: Simulation) -> Simulation:
    """Return the simulation on another backend, the same problem otherwise."""
    simulation.settings = dataclasses.replace(simulation.settings, backend=backend)
    return simulation


def loaded(simulation: Simulation, cleanup: ExitStack | None = None) -> Backend:
    """Return the simulation's own backend with its launch loaded."""
    problem = simulation.translate()
    backend = select(simulation.settings.backend, problem, grid=simulation.grid)
    if cleanup is not None:
        cleanup.callback(backend.close)
    backend.set_initial(problem.initial.field0, None)
    return backend


def finish(simulation: Simulation) -> np.ndarray:
    """Run to the last save plane on the simulation's backend; return the field."""
    backend = loaded(simulation)
    try:
        backend.propagate_to(float(simulation.settings.save_planes[-1]))
        field = backend.observe().field
        assert field is not None
        return field
    finally:
        backend.close()


def pair(build) -> tuple[np.ndarray, np.ndarray]:
    """Run a case on mlx and on the oracle; return both final fields."""
    device = on("mlx", build())
    oracle = build()
    assert device.settings.backend == "mlx" and oracle.settings.backend == "numpy_ref"
    return finish(device), finish(oracle)


def step_bound(steps: int, transforms: int, depth: int, scans: int = 0) -> float:
    """Return the class R bound of the oracle tests for `steps` matched steps."""
    per_stage = transforms * 16 * EPS_32 * depth + scans * 4 * higham_gamma(depth, U32)
    return steps * (2 * U32 + 7 * per_stage)


# Registration and refusals.


def test_mlx_is_a_single_precision_backend_only():
    assert "mlx" not in available()
    assert "mlx" in available("single")
    with pytest.raises(BackendError, match="binary32 only"):
        select("mlx", gnlse("double").translate())
    simulation = Simulation.from_gnlse(
        0.11, 0.05, [0.0, 0.0, -1e-26], lambda0=835e-9, trange=1e-12,
        wavelength_limits=(500e-9, 2e-6), width=50e-15, peak_power=1.0,
        pulseshape="sech", precision="single",
    )  # fmt: skip
    if mx.metal.is_available():
        assert simulation.settings.backend == "mlx"


def test_the_envelope_plasma_waits_for_phase_four():
    problem = capillary("single", envelope=True, plasma=True).translate()
    with pytest.raises(BackendError, match="envelope-grid plasma"):
        select("mlx", problem)


# Class R against the oracle.


@pytest.mark.parametrize("steps", [1, 10])
def test_a_soliton_with_raman_agrees_with_the_oracle(steps):
    device, oracle = pair(lambda: gnlse("single", steps=steps))
    assert_within(device, oracle, step_bound(steps, 2, 11))


def test_the_polarisation_taper_agrees_with_the_oracle():
    device, oracle = pair(lambda: gnlse("single", taper=True))
    assert_within(device, oracle, step_bound(10, 2, 11))


def test_low_storage_agrees_with_the_oracle():
    device, oracle = pair(lambda: gnlse("single", integrator=LowStorageRungeKutta()))
    assert_within(device, oracle, step_bound(10, 2, 11))


@pytest.mark.parametrize("gas", ["Ar", "N2"])
def test_a_plasma_capillary_agrees_with_the_oracle(gas):
    """Field plasma (argon), and molecular Raman beside it (nitrogen), 16 steps.

    The free current integrates an oscillating field to a small net value, a sum whose
    condition number the scans' bound carries.
    """
    limits = (200e-9, 3e-6) if gas == "N2" else (150e-9, 3e-6)
    device, oracle = pair(
        lambda: capillary("single", gas=gas, wavelength_limits=limits)
    )
    depth = 13
    assert_within(device, oracle, step_bound(16, 4, depth, scans=4))


def test_an_envelope_capillary_agrees_with_the_oracle():
    device, oracle = pair(lambda: capillary("single", envelope=True, plasma=False))
    assert_within(device, oracle, step_bound(16, 2, 11))


def test_a_multimode_capillary_agrees_with_the_oracle():
    """Four modes on their quadrature: the synthesis and projection products."""
    device, oracle = pair(
        lambda: capillary("single", modes=4, length=0.002, energy=20e-6)
    )
    assert_within(device, oracle, step_bound(16, 4, 13, scans=4))


def test_a_batch_member_is_its_solo_run():
    with ExitStack() as cleanup:
        members = tuple(
            loaded(on("mlx", gnlse("single", power=power)), cleanup)
            for power in (100.0, 150.0)
        )
        runner = select_batch(cast(Sequence[Backend], members))
        cleanup.callback(runner.close)
        runner.propagate_to(0.05)
        batch = [member.field for member in runner.observe_members()]
    for field, power in zip(batch, (100.0, 150.0), strict=True):
        assert np.array_equal(field, finish(on("mlx", gnlse("single", power=power))))


def test_a_resume_across_the_oracle_and_mlx_continues_the_run():
    """Same-precision resume on another backend: class T against the uninterrupted."""
    whole = finish(on("mlx", gnlse("single")))
    with ExitStack() as cleanup:
        first = loaded(gnlse("single"), cleanup)
        first.propagate_to(0.025)
        plane = first.observe().field
        second = loaded(on("mlx", gnlse("single")), cleanup)
        second.set_initial(plane, None, z=0.025)
        second.propagate_to(0.05)
        resumed = second.observe().field
    assert_within(resumed, whole, step_bound(10, 2, 11))


@pytest.mark.parametrize("family", ["envelope", "field"])
def test_tensor_raman_on_two_components_agrees_with_the_oracle(family):
    problem, grid, modes = vector_problem(family=family, steps=8)
    problem = dataclasses.replace(problem, precision=SINGLE)
    fields = []
    for backend_class in (MlxBackend, NumpySingleBackend):
        backend = backend_class(problem, grid=grid, modes=modes)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(float(problem.stepper.save_planes[-1]))
            fields.append(backend.observe().field)
        finally:
            backend.close()
    depth = math.ceil(math.log2(2 * grid.n_t)) + 4
    assert_within(fields[0], fields[1], step_bound(8, 4, depth))


# The protected operations.


@gpu_only
def test_the_propagator_kernel_does_not_err_with_the_phase():
    """The review's witness: a phase of 5e4 rad per term, damping up to 40 nepers."""
    rng = np.random.default_rng(0)
    terms = np.stack(
        [
            rng.standard_normal(4096) * 1e3 + 1j * rng.standard_normal(4096) * 5e4,
            rng.standard_normal(4096) - 0.01j * rng.random(4096),
        ]
    )
    weights = np.array([0.0137, 1.7])
    exact = np.exp(np.tensordot(weights, terms, axes=1))
    device = DoubleFloatTerms.upload(terms)
    # The kernel's own path: no binary64 copy is kept on the Metal device.
    assert device.host is None and len(device.parts) == 4
    forward = np.array(device.exponential(weights), dtype=np.complex128)
    inverse = np.array(device.exponential(weights, -1), dtype=np.complex128)
    # A few roundings, whatever the phase: the binary32 exponential, cosine and sine of
    # double-float-reduced arguments and their product, measured 5 u at worst; a
    # binary32 exponent would err by u |dPhi|, here up to 1e-2.
    assert np.max(np.abs(forward / exact - 1)) <= 8 * U32
    assert np.max(np.abs(inverse * exact - 1)) <= 8 * U32
    combined = np.array(device.combine(weights), dtype=np.complex128)
    total = np.tensordot(weights, terms, axes=1)
    assert np.max(np.abs(combined - total)) <= 2 * U32 * np.max(np.abs(total))


@gpu_only
def test_the_population_kernel_keeps_small_and_saturated_exposures():
    """MLX's own GPU expm1 loses up to 1e-4 at small arguments; the kernel does not."""
    exposure = np.concatenate([np.geomspace(1e-12, 80, 4000)]).astype(np.float32)
    wide = exposure.astype(np.float64)
    for initial in (0.0, 0.3):
        fraction, survival = population(mx.array(exposure), initial)
        expected = initial + (1 - initial) * -np.expm1(-wide)
        assert np.max(np.abs(np.array(fraction) / expected - 1)) <= 8 * U32
        assert (
            np.max(np.abs(np.array(survival) / ((1 - initial) * np.exp(-wide)) - 1))
            <= 8 * U32
        )


@gpu_only
def test_the_scan_kernel_is_the_oracles_order_exactly():
    rng = np.random.default_rng(5)
    for n in (1, 2, 4095, 4096, 65537):
        values = rng.random((3, n)).astype(np.float32)
        assert np.array_equal(
            np.array(prefix_scan(mx.array(values))), prefix_sum(values)
        )


@gpu_only
def test_the_device_rate_table_holds_the_slope_free_of_ln_e():
    table = capillary("single").translate().responses.plasma_channels[0].rate
    field = np.geomspace(table.field_min * 1.01, table.field_max * 0.99, 20001)
    field = field.astype(np.float32)
    exact = table(field.astype(np.float64))
    device = np.array(DeviceRateTable.upload(table)(mx.array(field)), np.float64)
    kept = exact > 1e-30 * exact.max()
    assert np.max(np.abs(device[kept] / exact[kept] - 1)) <= 8 * U32


@gpu_only
def test_the_gemm_probe_separates_float32_from_tf32():
    assert gemm_probe([(256, 256, 256), (64, 8, 4096)]) < 2.0**-17
    script = (
        "import mlx.core as mx, numpy as np\n"
        "a = mx.array(np.ones((8, 8), np.float32)); mx.eval(a @ a)\n"
        "import solprop.backends.mlx_gpu\n"
        "from solprop.backends.mlx_gpu.arithmetic import gemm_probe\n"
        "from solprop.exceptions import BackendError\n"
        "try:\n    gemm_probe([(256, 256, 256)])\nexcept BackendError as e:\n"
        "    print('refused', 'MLX_ENABLE_TF32=0' in str(e))\n"
        "else:\n    print('admitted')\n"
    )
    environment = dict(os.environ, MLX_ENABLE_TF32="1")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )
    assert result.stdout.strip() == "refused True"


# Guards, failures and memory.


@pytest.mark.parametrize("march", ["adaptive", "replayed", "low_storage"])
def test_a_non_finite_source_stops_every_march(march):
    simulation = {
        "adaptive": gnlse("single", steps=None),
        "replayed": gnlse("single"),
        "low_storage": gnlse("single", integrator=LowStorageRungeKutta()),
    }[march]
    with ExitStack() as cleanup:
        backend = cast(MlxBackend, loaded(on("mlx", simulation), cleanup))
        original = backend._device_sources

        def poisoned(field, bundle, raman):
            sources = original(field, bundle, raman)
            return type(sources)(sources.polarisation * math.nan, sources.current)

        backend._device_sources = poisoned
        with pytest.raises(ConvergenceError):
            backend.propagate_to(0.05)


def test_the_state_stays_complex64_on_the_device():
    with ExitStack() as cleanup:
        backend = cast(MlxBackend, loaded(on("mlx", gnlse("single")), cleanup))
        backend.propagate_to(0.01)
        state = backend._require_state()
        assert isinstance(state, mx.array)
        assert state.dtype == mx.complex64
        assert all(stage.dtype == mx.complex64 for stage in backend._last_stages)


@pytest.mark.parametrize("modes", [1, 4])
def test_the_memory_estimate_bounds_the_measured_peak(modes):
    simulation = on("mlx", capillary("single", modes=modes, length=0.002, energy=20e-6))
    problem = simulation.translate()
    estimate = estimate_memory(problem, "mlx")
    mx.reset_peak_memory()
    backend = select("mlx", problem, grid=simulation.grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(0.002)
        assert mx.get_peak_memory() <= estimate.total
    finally:
        backend.close()
