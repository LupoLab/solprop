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

from backends.test_single_precision import (
    capillary,
    gnlse,
    plasma_beam,
    spatial_kerr,
)
from bounds import EPS_32, UNIT_ROUNDOFF_32, assert_within, fft_bound, higham_gamma
from solprop.backends.memory import estimate_memory
from solprop.backends.mlx_gpu.arithmetic import (
    DoubleFloatTerms,
    fft_probe,
    gemm_probe,
    on_gpu,
    population,
    prefix_scan,
)
from solprop.backends.mlx_gpu.backend import MlxBackend
from solprop.backends.mlx_gpu.envelope import DeviceCycle, envelope_cycle
from solprop.backends.mlx_gpu.passes import DeviceRateTable
from solprop.backends.numpy_ref.envelope_plasma import (
    _cycle_quantities,
)
from solprop.backends.numpy_ref.single import NumpySingleBackend
from solprop.backends.protocol import Backend
from solprop.backends.registry import available, select, select_batch
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BackendError, ConvergenceError
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import (
    FixedSchedule,
    LowStorageRungeKutta,
    PIController,
    RungeKutta,
)
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


def envelope_plasma_case() -> Simulation:
    """Return a 1 mm envelope-plasma capillary on a 150 fs window, eight steps."""
    length = 0.001
    return capillary(
        "single",
        envelope=True,
        plasma=True,
        length=length,
        trange=150e-15,
        wavelength_limits=(300e-9, 2e-6),
        schedule=FixedSchedule.uniform(0.0, length, length / 8),
    )


@gpu_only
def test_an_envelope_plasma_capillary_agrees_with_the_oracle():
    """Class T: the device's cycle moments are a discretisation of the oracle's.

    On MLX's CPU device the envelope plasma is the oracle's own evaluation, so the
    comparison is run on the Metal device only.

    Each evaluation's moments may differ from the oracle's adaptive ones by the
    representation's budget, ``moment_rtol`` relative, on top of the binary32 rounding
    of the matched step; the plasma changes this field by order one.
    """
    device, oracle = pair(envelope_plasma_case)
    assert_within(device, oracle, step_bound(8, 4, 8, scans=4) + 8 * 2e-6)


def test_an_envelope_capillary_agrees_with_the_oracle():
    device, oracle = pair(lambda: capillary("single", envelope=True, plasma=False))
    assert_within(device, oracle, step_bound(16, 2, 11))


def test_a_multimode_capillary_agrees_with_the_oracle():
    """Four modes on their quadrature: the synthesis and projection products."""
    device, oracle = pair(
        lambda: capillary("single", modes=4, length=0.002, energy=20e-6)
    )
    assert_within(device, oracle, step_bound(16, 4, 13, scans=4))


@pytest.mark.parametrize("absorbed", [False, True])
@pytest.mark.parametrize("route", ["radial", "cartesian", "line"])
def test_the_spatial_routes_agree_with_the_oracle(route, absorbed):
    """A Kerr beam: the Hankel products over 32 nodes, or the 16-point transforms.

    Absorbed, the in-stage collar adds the radial absorber matrix or the Cartesian
    window's rate, each with its own pair of transforms.
    """
    device, oracle = pair(lambda: spatial_kerr(route, "single", absorbed=absorbed))
    transverse = {"radial": 32, "cartesian": 2 * 4, "line": 4}[route]
    transforms = 4 if absorbed else 2
    assert_within(device, oracle, step_bound(10, transforms, 11 + transverse))


@pytest.mark.parametrize(
    ("route", "gas"), [("radial", "Ar"), ("cartesian", "Ar"), ("radial", "N2")]
)
def test_a_plasma_beam_agrees_with_the_oracle(route, gas):
    """The held current on nodes (radial, projected once) or on columns (Cartesian).

    Nitrogen adds the molecular Raman response on the radial nodes.
    """
    device, oracle = pair(lambda: plasma_beam("single", route=route, gas=gas))
    transverse = 32 if route == "radial" else 2 * 4
    assert_within(device, oracle, step_bound(10, 4, 13 + transverse, scans=4))


def test_a_cartesian_beam_whose_energy_sum_overflows_binary32_runs_scaled():
    """A 10 GJ launch: its sum of squares exceeds binary32's range, its state does not.

    The state's power-of-two scale holds the device's sums, its norm among them, in
    range; the Kerr response at zero strength still puts the scaled field through
    every nonlinear pass.
    """

    def build() -> Simulation:
        return spatial_kerr("cartesian", "single", energy=1e10, chi3=0.0)

    launch = build().translate().initial.field0
    assert np.sum(np.abs(launch.astype(np.complex128)) ** 2) > np.finfo(np.float32).max
    device, oracle = pair(build)
    assert np.all(np.isfinite(device))
    assert_within(device, oracle, step_bound(10, 2, 11 + 2 * 4))


def test_a_gaussian_beam_diffracts_as_paraxial_optics_says():
    """A 100 um waist over one Rayleigh range on a 128 x 128 Cartesian grid.

    At the carrier bin the width grows by sqrt(2), the axial amplitude falls by
    sqrt(2) and the Gouy phase is ``arctan(1)`` (internal convention). The bound is
    four times the exact operator's first non-paraxial term, ``(k w0)^-2``; binary32's
    ten propagator roundings sit well inside it.
    """
    wavelength, waist = 800e-9, 100e-6
    k = 2 * math.pi / wavelength
    rayleigh = k * waist**2 / 2
    grid = EnvelopeGrid.from_limits(
        rayleigh, wavelength, (780e-9, 820e-9), 2e-12, thg=False
    )
    space = FreeGrid(1.5e-3, 128, 1.5e-3, 128, window_factor=0)
    settings = SimulationSettings(
        controller=PIController(rtol=1e-6, atol=0.0),
        save_planes=[rayleigh],
        dz_max=rayleigh,
        dz_init=rayleigh / 10,
        schedule=FixedSchedule.uniform(0.0, rayleigh, rayleigh / 10),
        backend="mlx",
        integrator=RungeKutta(),
        precision="single",
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=wavelength, fwhm=300e-15, energy=1e-3), waist
    )
    simulation = Simulation.from_free(
        grid, space, SpatialMedium(), [], pulse, settings=settings
    )
    index = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    assert grid.omega[index] == grid.omega0

    def beam(field: np.ndarray | None) -> np.ndarray:
        assert field is not None
        return np.fft.ifft2(field[0, :, 0, index].reshape(space.n_y, space.n_x))

    with ExitStack() as cleanup:
        backend = loaded(simulation, cleanup)
        entrance = beam(backend.observe().field)
        backend.propagate_to(rayleigh)
        exit_ = beam(backend.observe().field)
    x, y = np.meshgrid(space.x, space.y)

    def width(values: np.ndarray) -> float:
        intensity = np.abs(values) ** 2
        return float(np.sqrt(2 * np.sum(intensity * (x**2 + y**2)) / np.sum(intensity)))

    centre = (int(np.argmin(np.abs(space.y))), int(np.argmin(np.abs(space.x))))
    bound = 4 / (k * waist) ** 2
    assert abs(width(exit_) / (math.sqrt(2) * width(entrance)) - 1) < bound
    amplitude = abs(exit_[centre]) / abs(entrance[centre])
    assert abs(amplitude * math.sqrt(2) - 1) < bound
    gouy = float(np.angle(exit_[centre] / entrance[centre]))
    assert abs(gouy - math.atan(1)) < bound


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
        assert field is not None
        assert np.array_equal(field, finish(on("mlx", gnlse("single", power=power))))


def test_a_resume_across_the_oracle_and_mlx_continues_the_run():
    """Same-precision resume on another backend: class T against the uninterrupted."""
    whole = finish(on("mlx", gnlse("single")))
    with ExitStack() as cleanup:
        first = loaded(gnlse("single"), cleanup)
        first.propagate_to(0.025)
        plane = first.observe().field
        assert plane is not None
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


def test_the_propagator_drift_stays_linear_on_the_device():
    """A constant operator, no source: within ``M u`` after ``M`` steps (c <= 1)."""
    linear = {"gamma": 0.0, "raman": False, "length": 0.5}
    for steps in (10, 100, 1000):
        exact = finish(gnlse("double", steps=steps, **linear))
        actual = finish(on("mlx", gnlse("single", steps=steps, **linear)))
        assert_within(actual, exact, steps * U32, what=f"{steps} steps")


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
def ellipses(rng: np.random.Generator, n: int, low: float, high: float) -> np.ndarray:
    """Return two-component envelopes: linear, circular, elliptical, near-linear, zero.

    Amplitudes are log-uniform over ``[low, high]``.
    """
    amplitude = np.exp(rng.uniform(math.log(low), math.log(high), n))
    kind = rng.integers(0, 5, n)
    chi = np.select(
        [kind == 0, kind == 1, kind == 3],
        [0.0, math.pi / 4, 1e-4],
        rng.uniform(-math.pi / 4, math.pi / 4, n),
    )
    psi = rng.uniform(0, math.pi, n)
    x = np.cos(psi) * np.cos(chi) - 1j * np.sin(psi) * np.sin(chi)
    y = np.sin(psi) * np.cos(chi) + 1j * np.cos(psi) * np.sin(chi)
    field = np.stack([x, y]) * amplitude * np.exp(1j * rng.uniform(0, 2 * math.pi, n))
    field[:, kind == 4] = 0
    return field


@gpu_only
@pytest.mark.parametrize("n_pol", [1, 2])
def test_the_cycle_kernel_agrees_with_the_oracles_moments(n_pol):
    """Each sample's rate and loss current against the oracle's SVD and refinement.

    Both meet the representation's ``2e-6`` budget against the exact moments, so they
    differ by at most twice it plus the kernel's rounding of the rate's steep slope,
    relative to the representation's floor; zero rates (below the cutoff, zero fields)
    are zero on both.
    """
    channel = capillary("single", envelope=True, plasma=True).translate()
    envelope = channel.responses.envelope_plasma
    assert envelope is not None
    cycle = envelope.channels[0].cycle
    table = cycle.rate
    field = ellipses(
        np.random.default_rng(33), 4000, 0.5 * table.field_min, 2 * table.field_max
    )[:n_pol]
    exponent = 10
    scaled = (field * 2.0**-exponent).astype(np.complex64)
    rate, current, failed = envelope_cycle(
        mx.array(scaled[None]), DeviceCycle.upload(cycle), exponent
    )
    assert int(cast(int, failed.item())) == 0
    rate, current = np.array(rate)[0], np.array(current)[0]
    exact, loss = _cycle_quantities(
        scaled.astype(np.complex128) * 2.0**exponent, cycle, loss=True
    )
    assert np.array_equal(rate == 0, exact == 0)
    # Relative to the representation's own comparison floor: within 0.1 % of the
    # table's lower cutoff the cut angle, an arccosine near one, amplifies binary32's
    # rounding of the field, at rates below 1e-30 s^-1 that the floor already ignores.
    floor = np.maximum(exact, cycle.negligible_rate)
    bound = 2 * cycle.moment_rtol + 64 * UNIT_ROUNDOFF_32
    assert np.max(np.abs(rate - exact) / floor) < bound
    # The current per unit work is of order rate / E: at rates near the floor it is a
    # binary32 subnormal, which the GPU flushes (SEP 33 section 4.3, rule 7); its loss
    # term, times the work, is then below 1e-29 A/m^2.
    size = np.linalg.norm(loss, axis=0)
    live = size > float(np.finfo(np.float32).tiny) * 2**24
    error = np.linalg.norm(current - loss, axis=0)[live]
    assert np.max(error / size[live]) < bound


@gpu_only
def test_a_refinement_failure_stops_the_march():
    """A zero budget never converges: the step is refused, as the oracle refuses."""
    with ExitStack() as cleanup:
        backend = cast(
            MlxBackend,
            loaded(on("mlx", capillary("single", envelope=True, plasma=True)), cleanup),
        )
        backend._cycles = tuple(
            dataclasses.replace(cycle, moment_rtol=0.0) for cycle in backend._cycles
        )
        with pytest.raises(ConvergenceError, match="did not converge by order 512"):
            backend.propagate_to(0.001)


@gpu_only
def test_the_envelope_populations_are_the_oracles():
    """The species statistics' node scan on the device against the oracle's."""
    simulation = capillary("single", envelope=True, plasma=True)
    with ExitStack() as cleanup:
        device = cast(MlxBackend, loaded(on("mlx", simulation), cleanup))
        oracle = loaded(capillary("single", envelope=True, plasma=True), cleanup)
        device.propagate_to(0.002)
        field = device.observe().field
        assert field is not None
        nodes = oracle._nodes_at(field[0])  # pyright: ignore[reportAttributeAccessIssue]
        initials = (0.0,)
        fractions, peaks = device._envelope_populations(nodes, initials)
        expected = oracle._envelope_populations(nodes, initials)  # pyright: ignore[reportAttributeAccessIssue]
    assert np.max(fractions) > 1e-6
    bound = 2 * 2e-6 + 4096 * UNIT_ROUNDOFF_32
    assert_within(fractions, expected[0], bound)
    assert_within(peaks, expected[1], bound)


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


@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_every_transform_length_is_within_its_binary32_bound(kind):
    """The characterisation of SEP 33 6.3: powers of two and the smooth lengths."""
    for n in (1024, 4096, 1 << 14, 1 << 16, 3 * 1024, 5 * 7 * 512):
        assert fft_probe([(kind, n)]) <= fft_bound(n, EPS_32)


@gpu_only
def test_the_population_kernel_keeps_ieee_special_values():
    exposure = np.array([0.0, np.inf, np.nan], dtype=np.float32)
    fraction, survival = population(mx.array(exposure), 0.25)
    fraction, survival = np.array(fraction), np.array(survival)
    assert fraction[0] == np.float32(0.25) and survival[0] == np.float32(0.75)
    assert fraction[1] == 1.0 and survival[1] == 0.0
    assert np.isnan(fraction[2]) and np.isnan(survival[2])


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
    """With TF32 enabled the probe refuses, where the GPU has a TF32 path at all.

    Apple's newer GPUs compute float32 products in TF32 when MLX allows it; an older
    one (a hosted runner's) computes them in float32 regardless, and has nothing to
    refuse.
    """
    assert gemm_probe([(256, 256, 256), (64, 8, 4096)]) < 2.0**-17
    script = (
        "import mlx.core as mx, numpy as np\n"
        "rng = np.random.default_rng(0)\n"
        "x = rng.standard_normal((256, 256)).astype(np.float32)\n"
        "y = np.array(mx.array(x) @ mx.array(x), dtype=np.float64)\n"
        "w = x.astype(np.float64) @ x.astype(np.float64)\n"
        "if np.linalg.norm(y - w) / np.linalg.norm(w) < 2.0**-17:\n"
        "    print('no tf32'); raise SystemExit\n"
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
    if result.stdout.strip() == "no tf32":
        pytest.skip("this GPU computes float32 products even with TF32 enabled")
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


@pytest.mark.parametrize("route", ["radial", "cartesian"])
def test_the_memory_estimate_bounds_the_measured_peak_in_free_space(route):
    """The transverse transforms' transients, with both collars in each stage."""
    simulation = on("mlx", spatial_kerr(route, "single", absorbed=True))
    problem = simulation.translate()
    estimate = estimate_memory(problem, "mlx")
    mx.reset_peak_memory()
    backend = select("mlx", problem, grid=simulation.grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(float(simulation.settings.save_planes[-1]))
        assert mx.get_peak_memory() <= estimate.total
    finally:
        backend.close()


@gpu_only
def test_the_memory_estimate_bounds_the_measured_peak_with_envelope_plasma():
    """The cycle kernel's outputs, the histories and the drive and loss per stage."""
    simulation = on("mlx", envelope_plasma_case())
    problem = simulation.translate()
    estimate = estimate_memory(problem, "mlx")
    mx.reset_peak_memory()
    backend = select("mlx", problem, grid=simulation.grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(0.001)
        assert mx.get_peak_memory() <= estimate.total
    finally:
        backend.close()
