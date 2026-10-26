# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The single-precision path of numpy_ref, SEP 33's oracle (section 7.2).

Class R comparisons hold the binary32 path against the binary64 reference on the same
fixed schedule, inside bounds evaluated at binary32's unit roundoff; class P
comparisons are exact where the operation sequences differ only by powers of two;
the targeted tests reproduce the review's counterexamples (a vanished Raman kernel, a
lost population, a coherent propagator error) and the refusals.
"""

import dataclasses
import math
import warnings
from collections.abc import Sequence
from contextlib import ExitStack
from typing import cast

import numpy as np
import pytest
from scipy import signal

from bounds import EPS_32, UNIT_ROUNDOFF_32, assert_within, higham_gamma, prefix_bound
from precision_cases import capillary, finish, gnlse, loaded, plasma_beam, spatial_kerr
from solprop.backends import registry
from solprop.backends.memory import (
    MLX_GRAPH_BUDGET,
    estimate_memory,
    mlx_bounded,
    mlx_graph_bytes,
    mlx_stage_bytes,
)
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.backends.numpy_ref.single import (
    NumpySingleBackend,
    analytic_single,
    plasma_current_single,
    raman_single,
    running_integral,
    stable_population,
)
from solprop.backends.protocol import Backend
from solprop.backends.registry import (
    available,
    metal_available,
    select,
    select_batch,
)
from solprop.constants import EPS0
from solprop.driver import Simulation, prop_gnlse
from solprop.driver import simulation as simulation_module
from solprop.exceptions import (
    BackendError,
    ConvergenceError,
    DriverError,
    ExponentBudgetWarning,
    TranslationError,
)
from solprop.nonlinear.ionisation import RunningIntegral
from solprop.stats import StatsSpec
from solprop.stepper import LowStorageRungeKutta, PIController, RungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.precision import (
    PrecisionName,
    combine,
    precision_named,
    study_admission,
)
from vector_raman_cases import vector_problem

U32 = UNIT_ROUNDOFF_32


def rescaled(backend: Backend, exponent: int) -> None:
    """Hold a loaded single-precision state at the scale ``2**exponent`` instead."""
    single = cast(NumpySingleBackend, backend)
    physical = single._observed_field(single._require_state())
    single._scale = math.ldexp(1.0, exponent)
    single._scale_exponent = exponent
    controller = single._problem.stepper.controller
    single._controller = dataclasses.replace(
        controller, atol=controller.atol / single._scale
    )
    single._state = (physical / single._scale).astype(np.complex64)


# The interface: settings, admission and refusals.


def test_single_precision_takes_its_defaults_and_refuses_what_it_does_not_run():
    simulation = gnlse("single", steps=None)
    settings = simulation.settings
    assert settings.precision == "single"
    assert isinstance(settings.integrator, RungeKutta)
    assert settings.controller.rtol == 1e-6
    assert simulation.translate().precision.single

    def automatic() -> Simulation:
        return Simulation.from_gnlse(
            0.11, 0.05, [0.0, 0.0, -1e-26], lambda0=835e-9, trange=1e-12,
            wavelength_limits=(500e-9, 2e-6), width=50e-15, peak_power=1.0,
            pulseshape="sech", precision="single",
        )  # fmt: skip

    # "auto" is the GPU backend where MLX and a Metal device are present, the native
    # backend where its build runs single precision (SEP 36 phase 2), and refused
    # otherwise: never a binary64 backend, never the oracle (Q8).
    if metal_available():
        assert automatic().settings.backend == "mlx"
    elif "single" in registry.native_precisions():
        assert automatic().settings.backend == "native"
    else:
        with pytest.raises(DriverError, match="backend='auto'"):
            automatic()
    with pytest.raises(DriverError, match="refuses precision='single'"):
        dataclasses.replace(settings, backend="numba_cpu")
    with pytest.raises(DriverError, match="Adams integrator"):
        dataclasses.replace(settings, integrator=Multistep())
    with pytest.raises(DriverError, match="floor"):
        dataclasses.replace(
            settings, controller=dataclasses.replace(settings.controller, rtol=1e-8)
        )


def test_the_compiled_backends_refuse_a_single_precision_problem():
    """The refusal is the backend's own too, not only the settings' (SEP 33 5.2).

    The native backend runs single precision where its build compiled it (SEP 36
    phase 2; its refusals are ``test_native_single_precision.py``'s).
    """
    if "numba_cpu" not in available():
        pytest.skip("numba_cpu is not installed")
    problem = gnlse("single").translate()
    with pytest.raises(BackendError, match="binary64 only"):
        select("numba_cpu", problem)


def test_a_step_whose_inverse_propagator_overflows_binary32_is_refused():
    """A 1 mm capillary's collar on 0.25 mm steps: |Re dPhi| = 150, past 88.72.

    Binary64 runs it (its limit is 709.78); binary32 would hold ``exp(150)`` in the
    interaction picture's inverse propagator.
    """
    length = 0.001
    schedule = FixedSchedule.uniform(0.0, length, length / 4)
    capillary("double", length=length, schedule=schedule).translate()
    with pytest.raises(TranslationError, match=r"past binary32's range exp\(88\.72\)"):
        capillary("single", length=length, schedule=schedule).translate()


def test_a_step_within_a_tenth_of_binary32s_range_is_warned_of():
    """Seven steps of the same capillary: |Re dPhi| = 86, between 79.8 and 88.72."""
    length = 0.001
    schedule = FixedSchedule.uniform(0.0, length, length / 7)
    with pytest.warns(
        ExponentBudgetWarning, match="within a tenth of binary32's range"
    ):
        capillary("single", length=length, schedule=schedule).translate()


def test_a_noise_model_is_refused_under_single_precision():
    with pytest.raises(Exception, match="noise model"):
        Simulation.from_gnlse(
            0.11, 0.05, [0.0, 0.0, -1e-26], lambda0=835e-9, trange=1e-12,
            wavelength_limits=(500e-9, 2e-6), width=50e-15, peak_power=1.0,
            pulseshape="sech",
            precision="single", backend="numpy_ref", noise="input", seed=1,
        ).translate()  # fmt: skip


def test_the_precision_is_in_every_configuration_and_its_hash():
    double, single = gnlse("double"), gnlse("single")
    assert double.metadata().configuration["precision"] == "double"
    assert single.metadata().configuration["precision"] == "single"
    assert (
        double.metadata().configuration_hash() != single.metadata().configuration_hash()
    )


def test_a_resume_in_the_other_precision_is_refused(tmp_path):
    path = tmp_path / "run.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        keywords = {
            "lambda0": 835e-9, "wavelength_limits": (500e-9, 2e-6), "trange": 1e-12,
            "width": 50e-15, "peak_power": 10.0, "pulseshape": "sech", "n_save": 2,
            "backend": "numpy_ref", "integrator": RungeKutta(), "progress": "none",
        }  # fmt: skip
        prop_gnlse(0.11, 0.01, [0.0, 0.0, -1e-26], output=path, **keywords)
        with pytest.raises(Exception, match="precision"):
            prop_gnlse(
                0.11, 0.01, [0.0, 0.0, -1e-26], output=path, resume=True,
                precision="single", **keywords,
            )  # fmt: skip


# The state, its scale and what crosses to the host.


def test_the_state_is_binary32_at_the_launch_scale_and_observed_exactly():
    simulation = gnlse("single")
    with ExitStack() as cleanup:
        backend = cast(NumpySingleBackend, loaded(simulation, cleanup))
        state = backend._require_state()
        assert state.dtype == np.complex64
        assert backend.scale == 8.0
        # The launch's peak temporal amplitude over the scale lies in [1, 2).
        peak = float(np.max(np.abs(np.fft.ifft(state.astype(np.complex128)))))
        assert 1.0 <= peak < 2.0
        observed = backend.observe().field
        assert observed is not None and observed.dtype == np.complex128
        assert np.array_equal(observed, state.astype(np.complex128) * 8.0)
        # A reload of the observed plane holds the same binary32 state: class P.
        backend.propagate_to(0.02)
        plane = backend.observe().field
        held = backend._require_state().copy()
        assert plane is not None
        backend.set_initial(plane, None, z=0.02)
        assert np.array_equal(backend._require_state(), held)
        # The scale is the launch's, whatever the peak of a state loaded later.
        backend.set_initial(4 * plane, None, z=0.02)
        assert backend.scale == 8.0


def test_a_zero_launch_takes_the_unit_scale_and_stays_zero():
    problem = gnlse("single", gamma=0.0, raman=False).translate()
    zeros = np.zeros_like(problem.initial.field0)
    zeros.setflags(write=False)
    problem = dataclasses.replace(
        problem, initial=dataclasses.replace(problem.initial, field0=zeros)
    )
    backend = cast(NumpySingleBackend, select("numpy_ref", problem))
    try:
        assert backend.scale == 1.0
        backend.set_initial(zeros, None)
        backend.propagate_to(0.05)
        field = backend.observe().field
        assert field is not None and not np.any(field)
    finally:
        backend.close()


def test_scaling_by_a_power_of_two_is_exact():
    """Scaled and unscaled runs differ by powers of two alone: class P (section 4.3)."""
    for simulation in (gnlse("single"), capillary("single")):
        results = []
        for exponent in (None, 0, 11):
            with ExitStack() as cleanup:
                backend = loaded(simulation, cleanup)
                if exponent is not None:
                    rescaled(backend, exponent)
                backend.propagate_to(float(simulation.settings.save_planes[-1]))
                results.append(backend.observe().field)
        assert np.array_equal(results[0], results[1])
        assert np.array_equal(results[0], results[2])


# Class R: the binary32 path against the binary64 reference, same schedule.


@pytest.mark.parametrize("steps", [1, 10])
def test_the_oracle_agrees_with_binary64_within_a_binary32_bound(steps):
    """A step carries the state's rounding and its seven stages' transform rounding.

    Each stage's two transforms of 1024 samples and its pointwise response are
    bounded by ``16 eps ceil(log2 n)`` relative to the stage, which ``dz phi_NL`` (here
    below 0.1) scales into the state; the state itself is rounded once a step. The
    bound is ``steps (2 + 7 * 16 eps log2 n)`` in binary32's ``eps``.
    """
    single = finish(gnlse("single", steps=steps))
    double = finish(gnlse("double", steps=steps))
    n_to = gnlse("double").translate().grids.n_to
    bound = steps * (2 * U32 + 7 * 16 * EPS_32 * math.ceil(math.log2(n_to)))
    assert_within(single, double, bound, what=f"{steps} steps")


def test_the_polarisation_taper_agrees_with_binary64_within_its_bound():
    single = finish(gnlse("single", taper=True))
    double = finish(gnlse("double", taper=True))
    bound = 10 * (2 * U32 + 7 * 16 * EPS_32 * 10)
    assert_within(single, double, bound)


def test_a_capillary_with_plasma_agrees_with_binary64_within_its_bound():
    """16 steps of an ionising field-grid capillary; the scans add their tree depth."""
    simulation = capillary("double")
    n_to = simulation.translate().grids.n_to
    depth = math.ceil(math.log2(n_to))
    bound = 16 * (2 * U32 + 7 * (16 * EPS_32 * depth + 4 * higham_gamma(depth, U32)))
    assert_within(finish(capillary("single")), finish(simulation), bound)


def test_low_storage_agrees_with_binary64_within_its_bound():
    method = LowStorageRungeKutta()
    single = finish(gnlse("single", integrator=method))
    double = finish(gnlse("double", integrator=method))
    bound = 10 * (2 * U32 + 4 * 16 * EPS_32 * 10)
    assert_within(single, double, bound)


def test_a_batch_member_is_its_solo_run():
    """Members agree with their solo runs within the matched-discrete bound."""
    with ExitStack() as cleanup:
        members = tuple(loaded(gnlse("single"), cleanup) for _ in range(2))
        runner = select_batch(cast(Sequence[Backend], members))
        cleanup.callback(runner.close)
        runner.propagate_to(0.05)
        batch = [member.field for member in runner.observe_members()]
    solo = finish(gnlse("single"))
    bound = 10 * (2 * U32 + 7 * 16 * EPS_32 * 10)
    for field in batch:
        assert_within(field, solo, bound)


@pytest.mark.parametrize("route", ["radial", "cartesian"])
def test_the_spatial_routes_agree_with_binary64_within_their_bounds(route):
    """Ten steps of a self-focusing Kerr beam; a transverse transform per stage.

    The radial route's Hankel GEMM over ``n_r`` nodes and the Cartesian route's
    two-dimensional transform add their depth to the temporal transform's.
    """
    fields = {p: finish(spatial_kerr(route, p)) for p in ("double", "single")}
    n_t = spatial_kerr(route, "double").grid.n_t
    transverse = 32 if route == "radial" else 2 * 4
    depth = math.ceil(math.log2(n_t)) + transverse
    bound = 10 * (2 * U32 + 7 * 16 * EPS_32 * depth)
    assert_within(fields["single"], fields["double"], bound)


def test_a_radial_plasma_beam_agrees_with_binary64_within_its_bound():
    """The radial route's own plasma spectrum (held current on nodes), ten steps."""
    fields = {p: finish(plasma_beam(p)) for p in ("double", "single")}
    depth = math.ceil(math.log2(plasma_beam("double").grid.n_t)) + 32
    bound = 10 * (2 * U32 + 7 * (16 * EPS_32 * depth + 4 * higham_gamma(depth, U32)))
    assert_within(fields["single"], fields["double"], bound)


@pytest.mark.parametrize(
    ("family", "material"),
    [
        ("envelope", "H2"),
        ("field", "H2"),
        ("field_real", "H2"),
        ("envelope", "CH4"),
    ],
)
def test_tensor_raman_on_two_components_agrees_with_binary64(family, material):
    """Both helicities and an oblique delayed probe through H2's tensor ladder.

    The real field grid keeps the harmonic in the drive; methane, a spherical top, has
    the isotropic kernel alone.
    """
    problem, grid, modes = vector_problem(family=family, material=material, steps=8)
    fields = {}
    cases: tuple[tuple[PrecisionName, type[NumpyRefBackend]], ...] = (
        ("double", NumpyRefBackend),
        ("single", NumpySingleBackend),
    )
    for precision, backend_class in cases:
        case = dataclasses.replace(problem, precision=precision_named(precision))
        backend = backend_class(case, grid=grid, modes=modes)
        try:
            backend.set_initial(case.initial.field0, None)
            backend.propagate_to(float(case.stepper.save_planes[-1]))
            fields[precision] = backend.observe().field
        finally:
            backend.close()
    depth = math.ceil(math.log2(2 * grid.n_t)) + 4
    assert_within(fields["single"], fields["double"], step_bound_oracle(8, depth))


def step_bound_oracle(steps: int, depth: int) -> float:
    """Return the oracle tests' bound for `steps` steps of transforms of `depth`."""
    return steps * (2 * U32 + 7 * 16 * EPS_32 * depth)


# Targeted tests: the review's counterexamples.


def test_a_molecular_raman_response_survives_narrowing():
    """Nitrogen's per-particle kernel spectrum peaks near 5e-50 (review R1)."""
    simulation = capillary(
        "single", gas="N2", energy=20e-6, wavelength_limits=(200e-9, 3e-6)
    )
    problem = simulation.translate()
    with ExitStack() as cleanup:
        backend = cast(NumpySingleBackend, loaded(simulation, cleanup))
        bundle = backend._stage_scalars(0.0, 0.0)
        wide = NumpySingleBackend.__mro__[1]._raman_spectrum(  # pyright: ignore[reportAttributeAccessIssue]
            backend, bundle.raman_rates, bundle.raman_densities
        )
        backend._raman_kernel = None
        narrowed = backend._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
        exponent = backend.block_exponents["raman_kernel"]
    assert narrowed is not None and wide is not None
    assert float(np.max(np.abs(wide))) < 1e-45
    assert not np.any(wide.astype(np.complex64))
    t = np.arange(problem.grids.n_to) * problem.grids.dt_fine
    field = 3e10 * np.exp(-(((t - t.mean()) / 20e-15) ** 2)) * np.cos(2.35e15 * t)
    density = bundle.raman_densities[0] if bundle.raman_densities else bundle.density
    expected = passes.raman_polarisation(
        field[None, None, :], wide, kind="field", thg=False, density=density
    )
    scale = 2.0**34
    factor = combine(EPS0, density, exponent=2 * 34 + exponent).binary32()
    actual = raman_single(
        (field / scale).astype(np.float32)[None, None, :],
        narrowed,
        factor,
        kind="field",
        thg=False,
    )
    assert np.any(actual)
    bound = 64 * EPS_32 * math.ceil(math.log2(2 * problem.grids.n_to))
    assert_within(actual.astype(np.float64) * scale, expected, bound)


def test_the_stable_populations_keep_small_and_saturated_exposures():
    """Review R4's two cases, against the binary64 forms."""
    exposure = np.array([1e-8, 1e-3, 1.0, 20.0, 60.0], dtype=np.float32)
    for preionisation in (0.0, 0.3):
        fraction, survival = stable_population(exposure, preionisation)
        wide = exposure.astype(np.float64)
        expected_fraction = preionisation + (1 - preionisation) * -np.expm1(-wide)
        expected_survival = (1 - preionisation) * np.exp(-wide)
        assert np.all(
            np.abs(fraction - expected_fraction) <= 4 * U32 * expected_fraction
        )
        assert np.all(
            np.abs(survival - expected_survival) <= 4 * U32 * expected_survival
        )
    # The subtraction the binary64 path keeps loses both in binary32.
    naive = np.float32(1.0) - np.exp(-exposure)
    assert naive[0] == 0.0
    assert np.float32(1.0) - naive[3] == 0.0


def test_the_ionisation_work_is_the_population_it_ionises():
    """``I_p int W sigma dt = I_p (f_end - p)``: the loss's work from the survival."""
    t = np.linspace(-100e-15, 100e-15, 4097)
    dt = float(t[1] - t[0])
    rule = RunningIntegral.trapezoid()
    for peak in (1e6, 1e14, 5e14):
        rate = (peak * np.exp(-((t / 20e-15) ** 2))).astype(np.float32)
        exposure = running_integral(rate, dt, rule)
        fraction, survival = stable_population(exposure, 0.1)
        work = running_integral(rate * survival, dt, rule)[-1]
        depth = math.ceil(math.log2(t.size)) + 4
        # The population gained, ``(1 - p)(-expm1(-X))``: ``f - p`` from the rounded
        # fraction would lose the digits ``p`` holds.
        gained = 0.9 * -math.expm1(-float(exposure[-1]))
        assert gained > 0.0 and float(fraction[-1]) > 0.1
        # The quadrature of d f/dt = W sigma on this grid is second order: the
        # reference integral and its binary32 evaluation are compared at the
        # trapezoid's own error plus the scan's depth.
        wide_exposure = np.concatenate(
            ([0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * dt))
        )
        wide_work = np.sum(
            0.5
            * dt
            * (
                (rate * 0.9 * np.exp(-wide_exposure))[1:]
                + (rate * 0.9 * np.exp(-wide_exposure))[:-1]
            )
        )
        assert abs(work - wide_work) <= 8 * higham_gamma(depth, U32) * wide_work
        assert abs(wide_work - gained) <= 1e-3 * gained


def test_a_running_integral_is_within_its_prefix_bound():
    rng = np.random.default_rng(7)
    values = rng.random((3, 1 << 13)).astype(np.float32)
    dt = 1e-16
    rule = RunningIntegral.trapezoid()
    actual = running_integral(values, dt, rule).astype(np.float64)
    wide = values.astype(np.float64)
    increments = 0.5 * (wide[..., 1:] + wide[..., :-1]) * dt
    expected = np.concatenate(
        (np.zeros((3, 1)), np.cumsum(increments, axis=-1)), axis=-1
    )
    depth = math.ceil(math.log2(values.shape[-1])) + 3
    bound = prefix_bound(increments, depth, u=U32)
    assert np.all(np.abs(actual[..., 1:] - expected[..., 1:]) <= bound)


def test_the_plasma_current_vanishes_without_rate_or_field():
    simulation = capillary("single")
    channels = simulation.translate().responses.plasma_channels
    field = np.zeros((1, 1, 64), dtype=np.float32)
    field[0, 0, 10] = 1e5
    current = plasma_current_single(
        field, channels, (2.5e25,), (0.0,), 1e-16, RunningIntegral.trapezoid()
    )
    assert np.all(np.isfinite(current))
    assert current.dtype == np.float32


def test_the_propagator_drift_is_linear_in_the_step_count():
    """A constant operator, no source, an occupied band: ``c M u`` with ``c <= 1``.

    The binary64-accurate propagator rounds once per step (section 4.1, item 3),
    whatever ``|dPhi|``; the survival record states the measured ``c`` and the
    study's binary32-exponent variant beside it.
    """
    linear = {"gamma": 0.0, "raman": False, "length": 0.5}
    for steps in (10, 100, 1000):
        exact = finish(gnlse("double", steps=steps, **linear))
        actual = finish(gnlse("single", steps=steps, **linear))
        assert_within(actual, exact, steps * U32, what=f"{steps} steps")


# Guards and failures.


def test_a_binary64_array_in_a_pass_is_refused():
    with ExitStack() as cleanup:
        backend = cast(NumpySingleBackend, loaded(gnlse("single"), cleanup))
        wide = NumpySingleBackend.__mro__[1]._prefactor  # pyright: ignore[reportAttributeAccessIssue]
        backend._prefactor = lambda bundle: wide(backend, bundle)
        with pytest.raises(BackendError, match="binary64 array reached"):
            backend.propagate_to(0.05)


@pytest.mark.parametrize("march", ["adaptive", "replayed", "low_storage"])
def test_a_non_finite_source_stops_every_march(march):
    simulation = {
        "adaptive": gnlse("single", steps=None),
        "replayed": gnlse("single"),
        "low_storage": gnlse("single", integrator=LowStorageRungeKutta()),
    }[march]
    with ExitStack() as cleanup:
        backend = cast(NumpySingleBackend, loaded(simulation, cleanup))
        original = backend._column_sources

        def poisoned(field_t, bundle, raman):
            sources = original(field_t, bundle, raman)
            return passes.NonlinearSources(
                sources.polarisation * np.float32(np.nan), sources.current
            )

        backend._column_sources = poisoned
        with pytest.raises(ConvergenceError):
            backend.propagate_to(0.05)


def test_a_table_beyond_binary32_range_is_refused_by_name():
    with ExitStack() as cleanup:
        backend = cast(NumpySingleBackend, loaded(gnlse("single"), cleanup))
        with pytest.raises(BackendError, match="must carry its block exponent"):
            backend._upload("probe", np.array([1e-3, 1e-41]))
        with pytest.raises(BackendError, match="beyond binary32"):
            backend._upload("probe", np.array([1e39]))
        assert backend._upload("probe", np.array([3.0, 1e-41])).dtype == np.float32


def test_the_memory_estimate_counts_binary32_buffers():
    single = estimate_memory(gnlse("single").translate(), "numpy_ref")
    double = estimate_memory(gnlse("double").translate(), "numpy_ref")
    assert single.groups["base"] < double.groups["base"]


# The oracle's own branches: routes, switches and the host side's helpers.


def test_the_single_oracle_refuses_a_double_problem():
    problem = gnlse("double").translate()
    with pytest.raises(BackendError, match="needs a single-precision problem"):
        NumpySingleBackend(problem)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda p: capillary(p, modes=4, length=0.002, energy=20e-6), id="multimode"
        ),
        pytest.param(
            lambda p: capillary(p, envelope=True, plasma=False), id="envelope"
        ),
        pytest.param(
            lambda p: spatial_kerr("radial", p, absorbed=True), id="radial-collars"
        ),
        pytest.param(
            lambda p: capillary(p, envelope=True, plasma=False, thg=True),
            id="envelope-harmonic",
        ),
        pytest.param(lambda p: capillary(p, gas={"Ar": 0.5, "He": 0.5}), id="mixture"),
        pytest.param(
            lambda p: capillary(
                p, gas="N2", thg=True, wavelength_limits=(200e-9, 3e-6)
            ),
            id="field-raman",
        ),
    ],
)
def test_more_routes_agree_with_binary64_within_their_bounds(build):
    """The embedded quadrature, the envelope's harmonic carrier, the radial absorber,
    a two-species plasma and the field grid's Raman drive with its harmonic.

    Each takes a path of the oracle the cases above do not.
    """
    single, double = finish(build("single")), finish(build("double"))
    assert_within(single, double, step_bound_oracle(16, 13 + 32))


def test_the_statistics_are_taken_from_the_widened_state():
    with ExitStack() as cleanup:
        single = loaded(capillary("single"), cleanup)
        double = loaded(capillary("double"), cleanup)
        for backend in (single, double):
            backend.propagate_to(0.001)
        spec = StatsSpec(names=("energy", "electron_density"))
        ours, theirs = single.stats(spec), double.stats(spec)
    assert set(ours) == set(theirs)
    assert_within(np.asarray(ours["energy"]), np.asarray(theirs["energy"]), 1e-5)


def test_the_binary32_exponent_switch_rounds_the_exponent_first():
    """The study's variant: the exponent rounded to binary32 before the exponential."""
    with ExitStack() as cleanup:
        backend = cast(NumpySingleBackend, loaded(gnlse("single"), cleanup))
        backend.binary32_exponent = True
        backend.propagate_to(0.01)
        field = backend.observe().field
    assert field is not None and np.all(np.isfinite(field))


def test_a_non_finite_state_after_a_step_is_refused():
    with ExitStack() as cleanup:
        backend = cast(NumpySingleBackend, loaded(gnlse("single"), cleanup))
        state = np.full(backend._problem.shape, np.nan, dtype=np.complex64)
        with pytest.raises(ConvergenceError, match="not finite after a step"):
            backend._finish(state, 1e-3)


def test_a_weighted_controller_divides_by_its_bins():
    """A per-bin weight makes the controller's scale an array, applied in binary32."""
    n_omega = gnlse("single").translate().shape[-1]
    simulation = gnlse("single", steps=None)
    simulation.settings = dataclasses.replace(
        simulation.settings,
        controller=PIController(rtol=1e-5, atol=1e-6, weights=np.ones(n_omega)),
    )
    field = finish(simulation)
    assert np.all(np.isfinite(field))


def gradient(precision: str) -> Simulation:
    """Return an argon capillary on a pressure gradient, Kerr only, adaptive Adams."""
    return Simulation.from_capillary(
        50e-6,
        0.004,
        "Ar",
        (1.0, 0.5),
        lambda0=800e-9,
        wavelength_limits=(300e-9, 3e-6),
        trange=300e-15,
        fwhm=10e-15,
        energy=20e-6,
        plasma=False,
        n_save=2,
        backend="numpy_ref",
        precision=precision,  # pyright: ignore[reportArgumentType]
        integrator=Multistep(),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda p: gnlse(p, steps=None, integrator=Multistep()), id="constant"
        ),
        pytest.param(gradient, id="gradient"),
    ],
)
def test_adams_runs_in_binary32_under_the_study_admission(build):
    """The study's private route (Q2): the flow, its moments and the prediction rounded.

    Adams is refused publicly; the study measures it. Its binary32 result agrees with
    its binary64 one to the tolerance's truncation; on a gradient the operator is an
    expansion and the predictor carries its compensation.
    """
    with study_admission():
        single, double = finish(build("single")), finish(build("double"))
    assert_within(single, double, 1e-4)


def test_the_intermediate_ranges_record_each_pass():
    """The study's range record: the field, the polarisation and the current."""
    with ExitStack() as cleanup:
        backend = cast(NumpySingleBackend, loaded(capillary("single"), cleanup))
        backend.intermediate_ranges = {}
        backend.propagate_to(0.0005)
        ranges = backend.intermediate_ranges
    assert {"field", "polarisation", "current"} <= set(ranges)
    assert all(0 < low <= high for low, high in ranges.values())


def test_a_zero_launch_records_no_range():
    with ExitStack() as cleanup:
        backend = cast(NumpySingleBackend, loaded(gnlse("single"), cleanup))
        backend.set_initial(np.zeros(backend._problem.shape, dtype=np.complex128), None)
        backend.intermediate_ranges = {}
        backend.propagate_to(0.005)
        assert backend.intermediate_ranges == {}


def test_the_envelope_plasma_runs_in_binary32_on_the_oracle():
    """Two steps of a 1 mm envelope-plasma capillary on 256 samples, both precisions."""
    length = 0.001

    def build(precision: str) -> Simulation:
        return capillary(
            precision,
            envelope=True,
            plasma=True,
            length=length,
            trange=150e-15,
            wavelength_limits=(300e-9, 2e-6),
            schedule=FixedSchedule.uniform(0.0, length, length / 8),
        )

    with ExitStack() as cleanup:
        single, double = (loaded(build(p), cleanup) for p in ("single", "double"))
        for backend in (single, double):
            backend.propagate_to(length / 4)
        fields = [backend.observe().field for backend in (single, double)]
    assert fields[0] is not None and fields[1] is not None
    assert_within(fields[0], fields[1], 1e-5)


def test_without_mlx_single_precision_lists_the_cpu_backends_only(monkeypatch):
    real = registry._satisfied
    monkeypatch.setattr(
        registry,
        "_satisfied",
        lambda entry: entry is not registry.BACKENDS["mlx"] and real(entry),
    )
    # The oracle, and the native backend where it is installed (whose build refuses
    # single precision by name when it did not compile it).
    native = ("native",) if "native" in available() else ()
    assert available("single") == ("numpy_ref", *native)
    assert not registry.metal_available()
    assert registry.statistics_refusal("mlx", ["beam_spectrum"]) is None


def test_auto_resolves_to_mlx_where_metal_is_present(monkeypatch):
    monkeypatch.setattr(simulation_module, "metal_available", lambda: True)
    assert simulation_module._reduced_backend("auto", "single") == "mlx"
    assert simulation_module._reduced_backend("numpy_ref", "single") == "numpy_ref"


def test_auto_resolves_to_native_where_its_build_runs_single(monkeypatch):
    monkeypatch.setattr(simulation_module, "metal_available", lambda: False)
    monkeypatch.setattr(
        simulation_module, "native_precisions", lambda: ("double", "single")
    )
    assert simulation_module._reduced_backend("auto", "single") == "native"
    with pytest.raises(DriverError, match="native extra"):
        simulation_module._reduced_backend("auto", "mixed")


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: gnlse("single"), id="gnlse"),
        pytest.param(lambda: capillary("single"), id="plasma"),
        pytest.param(lambda: capillary("single", modes=4), id="multimode"),
        pytest.param(
            lambda: spatial_kerr("radial", "single", absorbed=True), id="radial"
        ),
        pytest.param(lambda: spatial_kerr("cartesian", "single"), id="cartesian"),
        pytest.param(
            lambda: capillary("single", envelope=True, plasma=True), id="envelope"
        ),
    ],
)
def test_the_mlx_estimate_adds_its_device_rows(build):
    """The device's rows are counted without MLX installed: they are arithmetic."""
    problem = build().translate()
    host = estimate_memory(problem, "numpy_ref").total
    device = estimate_memory(problem, "mlx").total
    assert device > host
    assert mlx_graph_bytes(problem) > 0 and mlx_stage_bytes(problem) > 0


def test_a_step_graph_past_the_budget_is_evaluated_stage_by_stage():
    assert mlx_bounded(7, MLX_GRAPH_BUDGET)
    assert not mlx_bounded(7, 1)


@pytest.mark.parametrize("n", [255, 256])
def test_the_binary32_analytic_signal_is_the_hilbert_transforms(n):
    """Odd and even lengths: the one-sided weights of :func:`scipy.signal.hilbert`."""
    values = np.random.default_rng(n).standard_normal(n).astype(np.float32)
    expected = signal.hilbert(values.astype(np.float64))
    assert_within(analytic_single(values), expected, 64 * U32)
