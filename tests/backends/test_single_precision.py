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
from legacy_geometry import LEGACY_GRID, bare_absorber
from solprop import fourier
from solprop.backends import registry
from solprop.backends.batch import LockstepRunner
from solprop.backends.driver import StepDriver
from solprop.backends.memory import (
    MLX_GRAPH_BUDGET,
    mlx_bounded,
    mlx_graph_bytes,
    mlx_stage_bytes,
)
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.backends.numpy_ref.batch import NumpyBatchBackend
from solprop.backends.numpy_ref.single import (
    NumpySingleBackend,
    analytic_single,
    plasma_current_single,
    raman_single,
    running_integral,
    stable_population,
)
from solprop.backends.protocol import Backend, Observation
from solprop.backends.registry import (
    available,
    metal_available,
    select,
    select_batch,
)
from solprop.boundaries import absorber
from solprop.constants import EPS0, C
from solprop.driver import Simulation, prop_gnlse
from solprop.driver import simulation as simulation_module
from solprop.driver.simulation import SimulationSettings
from solprop.exceptions import (
    BackendError,
    BoundaryWarning,
    ConvergenceError,
    DriverError,
    ExponentBudgetWarning,
    TranslationError,
)
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.monitor import BoundaryCheck
from solprop.nonlinear.ionisation import RunningIntegral
from solprop.nonlinear.optmat_raman import gas_raman
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.output.schema import RUN_SUMMARY_ANNOTATION, PlaneStore
from solprop.physdata import gamma3
from solprop.physdata.selection import MaterialSelection
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import LowStorageRungeKutta, PIController, RungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.precision import (
    PrecisionName,
    combine,
    precision_named,
)
from vector_raman_cases import vector_problem

U32 = UNIT_ROUNDOFF_32


def gnlse(  # noqa: PLR0913 - the case's own knobs
    precision: str,
    *,
    length: float = 0.05,
    steps: int | None = 10,
    gamma: float = 0.11,
    power: float = 100.0,
    raman: bool = True,
    integrator: RungeKutta | LowStorageRungeKutta | Multistep | None = None,
    beta0: float = 0.0,
    taper: bool = False,
) -> Simulation:
    """Return a 1024-bin soliton fibre, replayed on a uniform schedule by default."""
    schedule = (
        None
        if steps is None
        else FixedSchedule.uniform(0.0, length, length / steps, save_planes=[length])
    )
    return Simulation.from_gnlse(
        gamma,
        length,
        [beta0, 0.0, -1.183e-26, 8.1e-41],
        wavelength=835e-9,
        # The former 1024-bin window and no collar: the bounds below count its bins,
        # and a replayed schedule is refused beside the default collar.
        grid=EnvelopeGrid.from_limits(
            length, 835e-9, (500e-9, 2e-6), 1e-12, **LEGACY_GRID
        ),
        absorber=False,
        width=50e-15,
        peak_power=power,
        shape="sech",
        save_planes=[length],
        backend="numpy_ref",
        precision=precision,  # pyright: ignore[reportArgumentType]
        schedule=schedule,
        integrator=RungeKutta() if integrator is None else integrator,
        raman=raman,
        polarisation_taper=taper,
        noise=None,
    )


def capillary(
    precision: str, gas: str | dict[str, float] = "Ar", **overrides
) -> Simulation:
    """Return a short field-grid capillary, ionising in argon, on 16 fixed steps.

    The split-step collar is unconditionally stable, so the steps need not stay inside
    the in-stage collar's limit, which a short capillary's strong collar lowers.
    """
    length = overrides.pop("length", 0.004)
    keywords = {
        "wavelength": 800e-9,
        "wavelength_limits": (150e-9, 3e-6),
        "trange": 300e-15,
        "fwhm": 10e-15,
        "energy": 60e-6,
        "n_save": 2,
        "backend": "numpy_ref",
        "precision": precision,
        "integrator": RungeKutta(),
        "schedule": FixedSchedule.uniform(0.0, length, length / 16),
        "absorber_scheme": "split_step",
    } | overrides
    return Simulation.from_capillary(
        50e-6, length, gas, 1.0, **{"noise": None, **keywords}
    )


def loaded(simulation: Simulation, cleanup: ExitStack | None = None) -> Backend:
    """Return the simulation's backend with its launch loaded."""
    problem = simulation.translate()
    backend = select("numpy_ref", problem, grid=simulation.grid)
    if cleanup is not None:
        cleanup.callback(backend.close)
    backend.set_initial(problem.initial.field0, None)
    return backend


def finish(simulation: Simulation) -> np.ndarray:
    """Run to the last save plane and return the observed field."""
    backend = loaded(simulation)
    try:
        backend.propagate_to(float(simulation.settings.save_planes[-1]))
        field = backend.observe().field
        assert field is not None
        return field
    finally:
        backend.close()


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
    assert cast("PIController", settings.controller).rtol == 1e-6
    # A capillary takes its own default, at the floor, not the waveguide's 1e-5.
    controller = capillary("single").settings.controller
    assert cast("PIController", controller).rtol == 1e-6
    assert simulation.translate().precision.single

    def automatic() -> Simulation:
        return Simulation.from_gnlse(
            0.11, 0.05, [0.0, 0.0, -1e-26], wavelength=835e-9, trange=1e-12,
            wavelength_limits=(500e-9, 2e-6), width=50e-15, peak_power=1.0,
            shape="sech", precision="single", noise=None,
        )  # fmt: skip

    # "auto" is the GPU backend where MLX and a Metal device are present, and refused
    # otherwise: never a binary64 backend, never the oracle (Q8).
    if metal_available():
        assert automatic().settings.backend == "mlx"
    else:
        with pytest.raises(DriverError, match="backend='auto'"):
            automatic()
    with pytest.raises(DriverError, match="binary64 only"):
        dataclasses.replace(settings, backend="native")
    # The Adams integrator is admitted (#342); the default stays Runge-Kutta.
    assert isinstance(
        dataclasses.replace(settings, integrator=Multistep()).integrator, Multistep
    )
    with pytest.raises(DriverError, match="floor"):
        dataclasses.replace(
            settings,
            controller=dataclasses.replace(
                cast("PIController", settings.controller), rtol=1e-8
            ),
        )


@pytest.mark.parametrize("name", ["native"])
def test_the_compiled_backends_refuse_a_single_precision_problem(name):
    """The refusal is the backend's own too, not only the settings' (SEP 33 5.2)."""
    if name not in available():
        pytest.skip(f"{name} is not installed")
    problem = gnlse("single").translate()
    with pytest.raises(BackendError, match="binary64 only"):
        select(name, problem)


def test_the_oracle_applies_the_settings_admission_to_a_problem_past_them():
    """``registry.select`` on a translated problem meets the settings' refusals too.

    The settings refuse a tolerance below the floor under single precision; a problem
    replaced after translation reached the backend without them. The Adams integrator
    is admitted (#342).
    """
    problem = gnlse("single", steps=None).translate()
    stepper = problem.stepper
    adams = dataclasses.replace(
        problem, stepper=dataclasses.replace(stepper, integrator=Multistep())
    )
    tight = dataclasses.replace(
        problem,
        stepper=dataclasses.replace(
            stepper, controller=dataclasses.replace(stepper.controller, rtol=1e-8)
        ),
    )
    with pytest.raises(BackendError, match="below the single-precision floor"):
        select("numpy_ref", tight)
    select("numpy_ref", adams).close()


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


@pytest.mark.parametrize(
    "integrator",
    [LowStorageRungeKutta(), Multistep()],
    ids=["low_storage", "adams"],
)
def test_a_forward_only_march_takes_the_damped_step_runge_kutta_cannot(integrator):
    """The low-storage march and the exponential Adams flow form no inverse.

    A damped bin's forward flow underflows towards zero in binary32, harmlessly, so the
    step the Runge-Kutta interaction picture is refused (|Re dPhi| = 150) is admitted;
    only growth would overflow them.
    """
    length = 0.001
    schedule = FixedSchedule.uniform(0.0, length, length / 4)
    capillary(
        "single", length=length, schedule=schedule, integrator=integrator
    ).translate()


def overflowing_beam(precision: str) -> Simulation:
    """Return a radial Kerr beam whose first trial step is the whole millimetre.

    The default collar over that step reaches ``|Re dPhi| = 459``: inside binary64's
    range, past binary32's 88.72, in the interaction picture's inverse propagator --
    a Runge-Kutta trial's; the exponential Adams flow forms no inverse.
    """
    length = 1e-3
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 60e-6
    )
    settings = SimulationSettings(
        save_planes=[length],
        dz_max=length,
        dz_init=length,
        precision=precision,  # pyright: ignore[reportArgumentType]
        backend="numpy_ref",
        integrator=RungeKutta(),
    )
    return Simulation.from_radial(
        grid,
        RadialGrid(6e-4, 16),
        SpatialMedium(),
        [KerrResponse(1e-25, thg=False)],
        pulse,
        settings=settings,
    )


def run_to_end(simulation: Simulation, backend: str) -> Observation:
    """Propagate on `backend` to the last save plane and return its observation."""
    problem = simulation.translate()
    with ExitStack() as cleanup:
        made = select(backend, problem, grid=simulation.grid)
        cleanup.callback(made.close)
        made.set_initial(problem.initial.field0, None)
        made.propagate_to(float(simulation.settings.save_planes[-1]))
        return made.observe()


def test_an_adaptive_trial_that_overflows_binary32_is_rejected_and_shrunk():
    """A trial past ``exp(88.72)`` is a rejection, not the end of the run.

    Binary64 takes the whole millimetre in one step. In binary32 the first trials'
    propagators overflow and their norms are infinite: the controller rejects them and
    retries at its smallest factor, and the history records them as infinity rather
    than refusing the record at the plane.
    """
    single = run_to_end(overflowing_beam("single"), "numpy_ref")
    double = run_to_end(overflowing_beam("double"), "numpy_ref")
    history = single.step_history
    assert double.step_history.n_rejected == 0
    assert history.n_rejected >= 1
    assert not bool(history.accepted[0])
    assert history.error_norm[0] == np.inf
    accepted = history.accepted
    assert np.all(np.isfinite(history.error_norm[accepted]))
    assert np.all(history.error_norm[accepted] <= 1.0)
    # The survival record's field envelope (single-precision guide): both runs are
    # adaptive at the default tolerance, so they are held to it, not to a rounding
    # bound.
    assert_within(single.require_field(), double.require_field(), 1e-3)


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
            0.11, 0.05, [0.0, 0.0, -1e-26], wavelength=835e-9, trange=1e-12,
            wavelength_limits=(500e-9, 2e-6), width=50e-15, peak_power=1.0,
            shape="sech",
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
            "wavelength": 835e-9, "wavelength_limits": (500e-9, 2e-6), "trange": 1e-12,
            "width": 50e-15, "peak_power": 10.0, "shape": "sech", "n_save": 2,
            "backend": "numpy_ref", "integrator": RungeKutta(), "progress": "none",
        }  # fmt: skip
        prop_gnlse(
            0.11, 0.01, [0.0, 0.0, -1e-26], output=path, **{"noise": None, **keywords}
        )
        with pytest.raises(Exception, match="precision"):
            prop_gnlse(
                0.11, 0.01, [0.0, 0.0, -1e-26], output=path, resume=True,
                precision="single", **{"noise": None, **keywords},
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


def spatial_kerr(
    route: str,
    precision: str,
    *,
    steps: int = 10,
    absorbed: bool = False,
    energy: float = 3e-4,
    chi3: float = 1e-25,
) -> Simulation:
    """Return a self-focusing Kerr beam on the radial or Cartesian route.

    `absorbed` adds the grid's collars: the transverse window's in-stage rate beside
    the temporal one.
    """
    length = 0.1
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False, **LEGACY_GRID
    )
    spaces = {
        "radial": RadialGrid(2e-3, 32),
        "cartesian": FreeGrid(2e-3, 16, 2e-3, 16),
        "line": LineGrid(2e-3, 16),
    }
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=50e-15, energy=energy), 300e-6
    )
    settings = SimulationSettings(
        controller=PIController(rtol=1e-6, atol=0.0),
        save_planes=[length],
        dz_max=length,
        dz_init=length / steps,
        schedule=FixedSchedule.uniform(0.0, length, length / steps),
        backend="numpy_ref",
        integrator=RungeKutta(),
        precision=precision,  # pyright: ignore[reportArgumentType]
        absorber=(
            absorber(grid, boundary_length=100 * length)
            if absorbed
            else bare_absorber(grid)
        ),
    )
    constructor = Simulation.from_radial if route == "radial" else Simulation.from_free
    return constructor(
        grid,
        spaces[route],
        SpatialMedium(),
        [KerrResponse(chi3, thg=False)],
        pulse,
        settings=settings,
    )


def plasma_beam(
    precision: str, *, route: str = "radial", gas: str = "Ar"
) -> Simulation:
    """Return a short filament on the radial (or Cartesian) route, ten steps.

    Nitrogen adds its molecular Raman response beside the plasma.
    """
    length = 1e-3
    grid = FieldGrid.from_limits(length, 800e-9, (350e-9, 2e-6), 150e-15, **LEGACY_GRID)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=2e-4), 100e-6
    )
    responses: list[object] = [
        KerrResponse(gamma3(gas, wavelength=800e-9), thg=False),
        PlasmaResponse(gas, OptmatRate(gas, 800e-9)),
    ]
    if gas == "N2":
        selection = MaterialSelection(carrier_wavelength=800e-9)
        responses.append(gas_raman(gas, selection=selection, thg=False))
    settings = SimulationSettings(
        controller=PIController(rtol=1e-6, atol=0.0),
        save_planes=[length],
        dz_max=length,
        dz_init=length / 10,
        schedule=FixedSchedule.uniform(0.0, length, length / 10),
        backend="numpy_ref",
        integrator=RungeKutta(),
        ionisation_field_range=(1e7, 1e11),
        precision=precision,  # pyright: ignore[reportArgumentType]
        absorber=bare_absorber(grid),
    )
    if route == "cartesian":
        return Simulation.from_free(
            grid,
            FreeGrid(1e-3, 16, 1e-3, 16),
            SpatialMedium(gas, 1.0),
            responses,
            pulse,
            settings=settings,
            noise=None,
        )
    return Simulation.from_radial(
        grid,
        RadialGrid(1e-3, 32),
        SpatialMedium(gas, 1.0),
        responses,
        pulse,
        settings=settings,
        noise=None,
    )


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


# The oracle's own branches: routes, switches and the host side's helpers.


def test_the_single_oracle_refuses_a_double_problem():
    problem = gnlse("double").translate()
    with pytest.raises(BackendError, match="needs a single problem"):
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


# The split-step collar.


def collar_case(kind: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a two-mode state on 256 samples, a collar's rate and a step.

    The collar rises quadratically over the outer eighth of the window on each side
    to ``alpha h / 2 = 0.5``; a Gaussian pulse sits in the middle, so the factor is
    exactly one where the light is.
    """
    n_t, dz = 256, 1e-4
    t = np.arange(n_t) - n_t / 2
    edge = np.clip((np.abs(t) - 0.375 * n_t) / (0.125 * n_t), 0.0, 1.0)
    rate = 1e4 * edge**2
    rng = np.random.default_rng(79)
    pulse = np.exp(-((t / 20.0) ** 2))
    samples = pulse * rng.standard_normal((2, n_t))
    if kind == "envelope":
        samples = samples + 1j * pulse * rng.standard_normal((2, n_t))
    state = fourier.forward(samples, kind=kind)[None, :, None, :]  # pyright: ignore[reportArgumentType]
    return state, rate, dz


@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_the_split_collar_keeps_binary32_in_binary32(kind, monkeypatch):
    state, rate, dz = collar_case(kind)
    # Binary64 keeps its form, operation for operation.
    wide = passes.apply_temporal_rate(state, rate, dz, kind=kind)
    product = fourier.inverse(state, kind=kind) * np.exp(-0.5 * rate * dz)
    assert wide.dtype == np.complex128
    assert np.array_equal(wide, fourier.forward(product, kind=kind))
    narrow = state.astype(np.complex64)
    # Every transform the pass takes of a binary32 state is of binary32 operands.
    operands: list[np.dtype] = []
    for name in ("forward", "inverse"):
        original = getattr(fourier, name)

        def spy(values, *, kind, original=original):
            operands.append(np.asarray(values).dtype)
            return original(values, kind=kind)

        monkeypatch.setattr(passes.fourier, name, spy)
    absorbed = passes.apply_temporal_rate(narrow, rate, dz, kind=kind)
    monkeypatch.undo()
    assert absorbed.dtype == np.complex64
    assert len(operands) == 2
    assert set(operands) <= {np.dtype(np.float32), np.dtype(np.complex64)}
    # Against the binary64 pass on the same binary32 state: the correction's round
    # trip, two binary32 transforms of the oracle tests' 16 eps log2 n each and the
    # rounded factor and product, scaled by the correction's largest share
    # |f - 1| = 1 - exp(-0.5); and the final addition's rounding.
    change = float(np.max(-np.expm1(-0.5 * rate * dz)))
    transform = 16 * EPS_32 * math.ceil(math.log2(rate.size))
    widened = narrow.astype(np.complex128)
    exact = passes.apply_temporal_rate(widened, rate, dz, kind=kind)
    assert_within(absorbed, exact, 2 * U32 + change * (2 * transform + 2 * U32))
    # Without a collar the correction is exactly nothing.
    unchanged = passes.apply_temporal_rate(narrow, np.zeros_like(rate), dz, kind=kind)
    assert np.array_equal(unchanged, narrow)


def test_a_binary64_collar_in_the_finish_is_refused(monkeypatch):
    # The finish used to narrow whatever its collar returned; a binary64 result is now
    # a pass that left binary32, refused by name.
    widening = passes.apply_temporal_rate

    def widened(state, rate, dz, *, kind):
        return widening(state.astype(np.complex128), rate, dz, kind=kind)

    with ExitStack() as cleanup:
        backend = cast(NumpySingleBackend, loaded(capillary("single"), cleanup))
        state = backend._require_state()
        assert backend._finish(state, 1e-4).dtype == np.complex64
        monkeypatch.setattr(passes, "apply_temporal_rate", widened)
        with pytest.raises(BackendError, match="the finished state as complex128"):
            backend._finish(state, 1e-4)


def low_storage_split(route: str, precision: str) -> Simulation:
    """Return a low-storage march under the split-step collar: a field-grid capillary
    (Kerr only, 16 steps) or a Cartesian Kerr beam with its collars (10 steps), whose
    finish transforms each row across the plane and back around the collar."""
    if route == "capillary":
        return capillary(precision, plasma=False, integrator=LowStorageRungeKutta())
    simulation = spatial_kerr("cartesian", precision, absorbed=True)
    simulation.settings = dataclasses.replace(
        simulation.settings,
        integrator=LowStorageRungeKutta(),
        absorber_scheme="split_step",
    )
    return simulation


@pytest.mark.parametrize("route", ["capillary", "cartesian"])
def test_low_storage_with_the_split_collar_stays_in_binary32(route, monkeypatch):
    """The low-storage register step commits through the same checked finish."""
    original = passes.apply_temporal_rate
    results: list[np.dtype] = []

    def spy(state, rate, dz, *, kind):
        absorbed = original(state, rate, dz, kind=kind)
        results.append(absorbed.dtype)
        return absorbed

    monkeypatch.setattr(passes, "apply_temporal_rate", spy)
    single = finish(low_storage_split(route, "single"))
    assert results and set(results) == {np.dtype(np.complex64)}
    monkeypatch.undo()
    double = finish(low_storage_split(route, "double"))
    simulation = low_storage_split(route, "double")
    steps = 16 if route == "capillary" else 10
    transverse = 0 if route == "capillary" else 2 * 4
    depth = math.ceil(math.log2(simulation.translate().grids.n_to)) + transverse
    # The four stages' transforms of the low-storage test, and the collar's round
    # trip counted as a fifth: its rounding is further scaled by |f - 1| < 1.
    bound = steps * (2 * U32 + 5 * 16 * EPS_32 * depth)
    assert_within(single, double, bound)
    if route == "cartesian":
        # The plane's rows are written back into its own complex64 buffer, so a
        # widened collar would be narrowed there, not refused: the spy above is
        # this route's check.
        return

    def widened(state, rate, dz, *, kind):
        return original(state.astype(np.complex128), rate, dz, kind=kind)

    monkeypatch.setattr(passes, "apply_temporal_rate", widened)
    with ExitStack() as cleanup:
        backend = loaded(low_storage_split(route, "single"), cleanup)
        with pytest.raises(BackendError, match="the finished state as complex128"):
            backend.propagate_to(float(simulation.settings.save_planes[-1]))


def test_a_single_precision_batch_takes_each_member_s_own_trial():
    # The reference's batch arithmetic over the member axis is binary64: the
    # single-precision members step in turn, each through its binary32 trial.
    members = tuple(loaded(gnlse("single")) for _ in range(2))
    runner = select_batch(cast(Sequence[Backend], members))
    try:
        assert type(runner) is LockstepRunner
    finally:
        runner.close()


@pytest.mark.parametrize("order", [("single", "double"), ("double", "single")])
def test_a_batch_of_mixed_precisions_is_refused(order):
    with ExitStack() as cleanup:
        mixed = tuple(loaded(gnlse(precision), cleanup) for precision in order)
        with pytest.raises(BackendError, match="mixed precisions"):
            select_batch(cast(Sequence[Backend], mixed))
        # The shared member validation, whichever runner is built.
        with pytest.raises(BackendError, match="mixed precisions"):
            LockstepRunner(cast(Sequence[StepDriver[object]], mixed))


def test_the_reference_s_binary64_batch_refuses_a_single_member():
    with ExitStack() as cleanup:
        member = cast(NumpySingleBackend, loaded(gnlse("single"), cleanup))
        with pytest.raises(BackendError, match="refuses a single-precision member"):
            NumpyBatchBackend([member])


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
        wavelength=800e-9,
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
        noise=None,
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
def test_adams_runs_in_binary32(build):
    """The flow, its moments and the prediction rounded once (#342).

    Its binary32 result agrees with its binary64 one to the tolerance's truncation; on
    a gradient the operator is an expansion and the predictor carries its compensation.
    """
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


def test_without_mlx_single_precision_lists_the_oracle_only(monkeypatch):
    real = registry._satisfied
    monkeypatch.setattr(
        registry,
        "_satisfied",
        lambda entry: entry is not registry.BACKENDS["mlx"] and real(entry),
    )
    assert available("single") == ("numpy_ref",)
    assert not registry.metal_available()
    assert registry.statistics_refusal("mlx", ["beam_spectrum"]) is None


def test_auto_resolves_to_mlx_where_metal_is_present(monkeypatch):
    monkeypatch.setattr(simulation_module, "metal_available", lambda: True)
    assert simulation_module._single_backend("auto") == "mlx"
    assert simulation_module._single_backend("numpy_ref") == "numpy_ref"


def test_a_step_graph_past_the_budget_is_evaluated_stage_by_stage():
    assert mlx_bounded(7, MLX_GRAPH_BUDGET)
    assert not mlx_bounded(7, 1)


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
def test_the_mlx_graph_sizes_are_arithmetic(build):
    """The sizes the device's limits are set from need no MLX installed."""
    problem = build().translate()
    assert mlx_graph_bytes(problem) > 0 and mlx_stage_bytes(problem) > 0


@pytest.mark.parametrize("n", [255, 256])
def test_the_binary32_analytic_signal_is_the_hilbert_transforms(n):
    """Odd and even lengths: the one-sided weights of :func:`scipy.signal.hilbert`."""
    values = np.random.default_rng(n).standard_normal(n).astype(np.float32)
    expected = signal.hilbert(values.astype(np.float64))
    assert_within(analytic_single(values), expected, 64 * U32)


# A whole run, as a user sees it.


def drifting_pulse(backend: str) -> Simulation:
    """Return a single-precision fibre run whose pulse walks deep into the collar.

    A ten-kilowatt Gaussian pulse, offset in frequency so that the group-velocity
    walk-off carries it 0.65 of the window over a metre of 200 fixed steps, under the
    split-step collar, its boundary check relaxed so the walk is recorded, not
    stopped. The nonlinearity is scaled down with the power: the walk is linear.
    """
    length, trange, beta2, peak_power = 1.0, 1e-12, 2e-26, 1e4
    grid = EnvelopeGrid.from_limits(length, 1e-6, (0.8e-6, 1.25e-6), trange)
    omega0 = 2 * math.pi * C / 1e-6
    offset = 0.65 * trange / (beta2 * length)
    simulation = Simulation.from_gnlse(
        1e-9 / peak_power, length, [0.0, 0.0, beta2], wavelength=1e-6, grid=grid,
        pulses=[GaussianPulse(
            wavelength=2 * math.pi * C / (omega0 + offset), fwhm=200e-15,
            peak_power=peak_power,
        )],
        noise=None, raman=False, shock=False, n_save=3, precision="single",
        backend=backend, integrator=RungeKutta(), rtol=1e-5,
        schedule=FixedSchedule.uniform(0.0, length, length / 200),
    )  # fmt: skip
    simulation.settings = dataclasses.replace(
        simulation.settings,
        boundary_check=BoundaryCheck(budget=1.0, stop_factor=1.0),
        absorber_scheme="split_step",
    )
    return simulation


def check_a_whole_run(backend: str) -> None:
    """Run :func:`drifting_pulse` through ``Simulation.run`` and check what it reports.

    The summary names the precision and the backend, the saved field is widened to
    binary64 (observation is exact), and the boundary monitor's record counts every
    step and reads as removed what the run lost.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        run = drifting_pulse(backend).run(progress="none")
    assert isinstance(run, PlaneStore)
    (summary,) = cast(
        "list[dict[str, list[list[str]]]]", run.annotations[RUN_SUMMARY_ANNOTATION]
    )
    rows = {row[0]: row[1] for row in summary["rows"]}
    assert rows["precision"] == "single"
    assert rows["backend"] == backend
    field = np.asarray(run["field"])
    assert field.dtype == np.complex128
    assert field.shape[0] == 3 and np.all(np.isfinite(field))
    outcome = run.outcome
    assert outcome is not None and outcome.execution is not None
    record = outcome.execution["boundary_monitor"]
    assert isinstance(record, dict)
    assert record["steps"] == 200
    assert record["launch_energy"] > 0.0
    energy = np.asarray(run["stats/energy"])[:, 0, 0]
    lost = float(1.0 - energy[-1] / energy[0])
    assert lost > 0.5
    # Binary32's rounding over 200 steps: each step's removal is a non-negative sum,
    # within gamma of its few hundred samples of itself, and the states it is taken
    # from agree with the energies' to the march's rounding.
    assert record["removed_fraction"] == pytest.approx(lost, rel=1e-4)
    assert 0.0 < record["collar_max"] <= 1.0


@pytest.mark.medium
def test_a_whole_single_precision_run_reports_what_it_ran():
    check_a_whole_run("numpy_ref")
