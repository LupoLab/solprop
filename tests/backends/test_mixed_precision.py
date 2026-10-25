# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The mixed-precision path of numpy_ref, SEP 36's oracle (section 7.2).

Class R comparisons hold the mixed path against the binary64 reference on the same
fixed schedule, inside the bound of SEP 36 section 7.1: per stage, the source's
binary32 error ``kappa u32 + E_pass`` relative to the stage value; per step, the stages
weighted by the solution weights and carried in by the step's nonlinear phase
``dz ||k|| / ||A||``, which is read from the binary64 reference's own stages before the
comparison; over the run, their sum. Class P comparisons are exact where the mixed path
does exactly what the reference does (a source-free state path) or where only powers of
two differ. Independent witnesses hold the oracle against physics, not against itself,
and the targeted tests reproduce what the review asked for: a strongly damped bin, a
non-unit launch scale, and the dtype guards on both sides of the source.
"""

import dataclasses
import math
import warnings
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from typing import cast

import numpy as np
import pytest

from bounds import (
    EPS,
    UNIT_ROUNDOFF_32,
    assert_within,
    mixed_run_bound,
    mixed_step_bound,
    rate_condition,
    source_pass_bound,
)
from precision_cases import (
    capillary,
    finish,
    gnlse,
    loaded,
    plasma_beam,
    spatial_kerr,
)
from solprop.backends import memory
from solprop.backends.memory import estimate_memory, mixed_precision_rows
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.mixed import NumpyMixedBackend
from solprop.backends.protocol import Backend
from solprop.backends.registry import available, select, select_batch
from solprop.boundaries import absorber
from solprop.driver import Simulation, prop_gnlse
from solprop.exceptions import BackendError, DriverError, TranslationError
from solprop.stats import StatsSpec
from solprop.stepper import DOPRI5, LowStorageRungeKutta, RungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.ionisation import RateTable
from solprop.translate.linop import delta_phi
from solprop.translate.precision import MIXED, study_admission

U32 = UNIT_ROUNDOFF_32

# Kerr and Raman are cubic in the field: a relative change of the input moves the
# source by three times as much (SEP 36 section 7.1, item 1).
CUBIC = 3.0


Build = Callable[[str], Simulation]
"""A case, as a function of its precision."""


@dataclasses.dataclass(frozen=True)
class Reference:
    """What SEP 36's bound reads from the binary64 reference, before any mixed run.

    Attributes
    ----------
    phases : numpy.ndarray
        Each step's nonlinear phase ``dz max_i ||N_i|| / ||A_j||``: the largest
        lab-frame source of its evaluations over the state at its anchor, plus the
        step's in-stage absorber exponent (:func:`absorber_rate`).
    norms : numpy.ndarray
        The state's norm at each step's anchor and at the end, which carry a step's
        relative error into a comparison's normwise one at any step.
    fields : numpy.ndarray
        Every field magnitude the ionisation rates were evaluated at, V/m.
    """

    phases: np.ndarray
    norms: np.ndarray
    fields: np.ndarray


def reference(build: Build) -> Reference:
    """Run the binary64 reference on the case's schedule and record its phases.

    The lab-frame source of an evaluation is its stage value with the inverse
    propagator divided out: SEP 36 section 7.1 bounds a stage's error relative to it,
    and the inverse propagator scales absolute bounds bin by bin (item 1).
    """
    simulation = build("double")
    problem = simulation.translate()
    method = problem.stepper.integrator
    per_step = (
        method.evaluations
        if isinstance(method, LowStorageRungeKutta)
        else DOPRI5.n_stages
    )
    sources: list[float] = []
    anchors = [float(np.linalg.norm(problem.initial.field0))]
    fields: list[np.ndarray] = []
    rate = RateTable.__call__

    def evaluated(table: RateTable, amplitude):
        fields.append(np.abs(np.asarray(amplitude, dtype=np.float64)).ravel())
        return rate(table, amplitude)

    with ExitStack() as cleanup:
        backend = cast(NumpyRefBackend, loaded(simulation, cleanup))
        source, finished = backend._nonlinear, backend._finish

        def recorded(lab, bundle, inverse, **keywords):
            stage = source(lab, bundle, inverse, **keywords)
            lab_frame = np.divide(
                stage, inverse, out=np.zeros_like(stage), where=inverse != 0
            )
            sources.append(float(np.linalg.norm(lab_frame)))
            return stage

        def kept(state, dz):
            result = finished(state, dz)
            anchors.append(float(np.linalg.norm(result)))
            return result

        backend._nonlinear = recorded
        backend._finish = kept
        RateTable.__call__ = evaluated
        cleanup.callback(setattr, RateTable, "__call__", rate)
        backend.propagate_to(float(simulation.settings.save_planes[-1]))
    schedule = problem.stepper.schedule
    assert schedule is not None
    steps = np.asarray(schedule.steps, dtype=np.float64)
    assert len(sources) == per_step * steps.size
    largest = np.max(np.reshape(sources, (steps.size, per_step)), axis=1)
    norms = np.asarray(anchors, dtype=np.float64)
    return Reference(
        phases=steps * (largest / norms[:-1] + absorber_rate(problem)),
        norms=norms,
        fields=np.concatenate(fields) if fields else np.zeros(0),
    )


def absorber_rate(problem) -> float:
    """Return the largest field rate of the in-stage collars, ``max rate / 2``, 1/m.

    An in-stage collar's term ``-rate E / 2`` is formed from the binary32 transform of
    the whole field, whose rounding is spread over every sample: its error is the
    largest rate times the field's, not a fraction of the term itself. SEP 36 section
    4.2 scales it by the step's absorber exponent, so ``dz`` times this rate joins the
    step's phase. Zero without an in-stage collar.
    """
    responses = problem.responses
    rate = 0.0
    if (
        responses.temporal_absorber_fine is not None
        and responses.absorber_scheme == "nonlinear"
    ):
        rate += float(np.max(responses.temporal_absorber_fine))
    geometry = problem.transform.geometry
    if geometry is not None and geometry.kind == "free":
        rate += float(np.max(geometry.absorber_rate, initial=0.0))
    return 0.5 * rate


def plasma_kappa(record: Reference, problem) -> float:
    """Return the plasma's condition number over the fields the reference met.

    ``1 + max |d ln W / d ln |E||`` (SEP 36 section 7.1, item 1) at every field
    magnitude the binary64 reference evaluated a rate at where the rate is at least
    ``u32`` of its largest: a sample whose rate is below that contributes to the
    current less than binary32 rounds its largest term, so its relative error does not
    enter the normwise bound.
    """
    kappa = 1.0
    for channel in problem.responses.plasma_channels:
        rates = np.asarray(channel.rate(record.fields), dtype=np.float64)
        live = rates >= U32 * float(np.max(rates))
        kappa = max(kappa, rate_condition(channel.rate, record.fields[live]))
    return kappa


def bound_of(
    build: Build,
    *,
    kappa: float | None = None,
    scans: int = 0,
    transverse: int = 0,
) -> float:
    """Return SEP 36's mixed bound of a fixed-schedule case against binary64.

    `kappa` defaults to the cubic responses' 3, or the plasma's condition number
    (:func:`plasma_kappa`) where the case ionises, with its two running integrals.
    The binary64 side's own rounding is the run bound's state-path term (item 4).
    """
    problem = build("double").translate()
    record = reference(build)
    if kappa is None:
        plasma = bool(problem.responses.plasma_channels)
        kappa = plasma_kappa(record, problem) if plasma else CUBIC
        scans = 2 if plasma else scans
    samples = problem.grids.n_to
    passes = source_pass_bound(samples, scans=scans, transverse=transverse)
    method = problem.stepper.integrator
    if isinstance(method, LowStorageRungeKutta):
        # The low-storage march's evaluations, whose weights in the step sum to one.
        weights, stages = np.ones(1), method.evaluations
    else:
        weights, stages = DOPRI5.solution_weights, DOPRI5.n_stages
    scale = record.norms[:-1] / record.norms[-1]
    steps = [
        weight
        * mixed_step_bound(
            phase, kappa=kappa, passes=passes, weights=weights, stages=stages
        )
        for phase, weight in zip(record.phases, scale, strict=True)
    ]
    return mixed_run_bound(steps, samples=samples, stages=stages)


# A bound above a tenth of the field's class T envelope (1e-3) is uninformative, and
# the comparison runs on a shorter window instead (SEP 36 section 7.1).
INFORMATIVE = 1e-4


def compare(build: Build, bound: float, what: str = "") -> float:
    """Assert a case's mixed result is within an informative `bound` of binary64's."""
    assert bound < INFORMATIVE, f"{what}: the bound {bound:.2e} is uninformative"
    mixed, double = finish(build("mixed")), finish(build("double"))
    return assert_within(mixed, double, bound, what=what)


# The interface: settings, admission and refusals.


def test_mixed_precision_takes_its_defaults_and_refuses_what_it_does_not_run():
    simulation = gnlse("mixed", steps=None)
    settings = simulation.settings
    assert settings.precision == "mixed"
    assert isinstance(settings.integrator, RungeKutta)
    # The larger of the binary64 default and the floor: 1e-8 for the GNLSE, 1e-6 for
    # a waveguide (SEP 36 section 5).
    assert settings.controller.rtol == 1e-8
    assert capillary("mixed").settings.controller.rtol == 1e-6
    assert simulation.translate().precision is MIXED
    for backend in ("numba_cpu", "native", "mlx"):
        with pytest.raises(DriverError, match="refuses precision='mixed'"):
            dataclasses.replace(settings, backend=backend)
    with pytest.raises(DriverError, match="Adams integrator"):
        dataclasses.replace(settings, integrator=Multistep())
    with pytest.raises(DriverError, match="mixed-precision floor 1e-08"):
        dataclasses.replace(
            settings, controller=dataclasses.replace(settings.controller, rtol=1e-9)
        )
    assert available("mixed") == ("numpy_ref",)


def test_auto_is_refused_without_a_native_build_that_runs_mixed():
    """``auto`` never resolves to the oracle, and native has no mixed build yet."""
    with pytest.raises(DriverError, match="native extra"):
        Simulation.from_gnlse(
            0.11, 0.05, [0.0, 0.0, -1e-26], lambda0=835e-9, trange=1e-12,
            wavelength_limits=(500e-9, 2e-6), width=50e-15, peak_power=1.0,
            pulseshape="sech", precision="mixed",
        )  # fmt: skip


@pytest.mark.parametrize("name", ["numba_cpu", "native"])
def test_the_compiled_backends_refuse_a_mixed_problem(name):
    """The refusal is the backend's own too, not only the settings' (SEP 36 5)."""
    if name not in available():
        pytest.skip(f"{name} is not installed")
    problem = gnlse("mixed").translate()
    with pytest.raises(BackendError, match="refuses precision='mixed'"):
        select(name, problem)


def test_the_mixed_oracle_refuses_another_precision():
    for precision in ("double", "single"):
        problem = gnlse(precision).translate()
        with pytest.raises(BackendError, match="needs a mixed-precision problem"):
            NumpyMixedBackend(problem)


def test_the_routes_single_precision_refuses_are_refused_under_mixed():
    problem = spatial_kerr("cartesian", "mixed").translate()
    statistics = StatsSpec(names=("beam_spectrum",))
    with pytest.raises(TranslationError, match="precision='mixed' refuses the far"):
        dataclasses.replace(problem, stats_spec=statistics).validate()


@pytest.mark.parametrize("model", ["input", "modified"])
def test_a_noise_model_is_refused_under_mixed_until_its_study(model):
    """Phase 5 admits noise on what it qualifies; the study reaches it before then."""

    def build() -> Simulation:
        return Simulation.from_gnlse(
            0.11, 0.05, [0.0, 0.0, -1e-26], lambda0=835e-9, trange=1e-12,
            wavelength_limits=(500e-9, 2e-6), width=50e-15, peak_power=1.0,
            pulseshape="sech", precision="mixed", backend="numpy_ref",
            noise=model, seed=1,
        )  # fmt: skip

    with pytest.raises(TranslationError, match="noise model until its noise study"):
        build().translate()
    with study_admission():
        assert build().translate().initial.noise_model == model


def test_the_precision_is_in_every_configuration_and_its_hash():
    hashes = set()
    for precision in ("double", "single", "mixed"):
        metadata = gnlse(precision).metadata()
        assert metadata.configuration["precision"] == precision
        hashes.add(metadata.configuration_hash())
    assert len(hashes) == 3


def test_a_resume_in_another_precision_is_refused(tmp_path):
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
                precision="mixed", **keywords,
            )  # fmt: skip


def test_a_batch_admits_members_of_one_precision():
    with ExitStack() as cleanup:
        members = (loaded(gnlse("mixed"), cleanup), loaded(gnlse("double"), cleanup))
        with pytest.raises(BackendError, match="one precision"):
            select_batch(cast(Sequence[Backend], members))


# The state, its scale and the stored rows.


def test_the_state_and_its_rows_are_binary64_and_observed_exactly():
    """The state is physical binary64; a reload is class P with no scaling."""
    simulation = gnlse("mixed")
    with ExitStack() as cleanup:
        backend = cast(NumpyMixedBackend, loaded(simulation, cleanup))
        state = backend._require_state()
        assert state.dtype == np.complex128
        # The source's input is narrowed at the launch's scale: peak 10 W^0.5 -> 8.
        assert backend.scale == 8.0
        backend.propagate_to(0.02)
        assert all(stage.dtype == np.complex128 for stage in backend._last_stages)
        plane = backend.observe().field
        held = backend._require_state().copy()
        assert plane is not None and plane.dtype == np.complex128
        assert np.array_equal(plane, held)
        backend.set_initial(plane, None, z=0.02)
        assert np.array_equal(backend._require_state(), held)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: gnlse("mixed"), id="gnlse"),
        pytest.param(lambda: capillary("mixed"), id="plasma"),
        pytest.param(
            lambda: capillary("mixed", gas="N2", wavelength_limits=(200e-9, 3e-6)),
            id="raman",
        ),
    ],
)
def test_scaling_by_a_power_of_two_is_exact(build):
    """The narrowing's scale enters and leaves exactly: class P (SEP 33 4.3)."""
    simulation = build()
    results = []
    for exponent in (None, 0, 11):
        with ExitStack() as cleanup:
            backend = cast(NumpyMixedBackend, loaded(simulation, cleanup))
            if exponent is not None:
                backend._scale = math.ldexp(1.0, exponent)
                backend._scale_exponent = exponent
            backend.propagate_to(float(simulation.settings.save_planes[-1]))
            results.append(backend.observe().field)
    assert np.array_equal(results[0], results[1])
    assert np.array_equal(results[0], results[2])


# Class R: the oracle against binary64 over one step and over a few.


def stage_source(backend: NumpyRefBackend, field: np.ndarray, z: float) -> np.ndarray:
    """Return a backend's lab-frame source at a binary64 state, physical units."""
    bundle = backend._stage_scalars(z, z)
    return backend._nonlinear(field, bundle, backend._lab_frame)


@pytest.mark.parametrize("exponent", [None, 30])
@pytest.mark.parametrize(
    ("build", "transverse"),
    [
        pytest.param(lambda p: gnlse(p, raman=False), 0, id="kerr"),
        pytest.param(gnlse, 0, id="kerr-raman"),
        pytest.param(capillary, 0, id="plasma"),
        pytest.param(
            lambda p: capillary(p, gas="N2", wavelength_limits=(200e-9, 3e-6)),
            0,
            id="plasma-raman",
        ),
        pytest.param(lambda p: spatial_kerr("radial", p), 32, id="radial"),
        pytest.param(lambda p: spatial_kerr("cartesian", p), 8, id="cartesian"),
    ],
)
def test_a_source_evaluation_is_within_its_stage_bound(build, transverse, exponent):
    """Item 1 directly: the same binary64 input, a binary32 source, one widening.

    With ``exponent`` the narrowing's scale is ``2**30`` rather than the launch's (a
    few powers of two in these cases' state units): the physical source comes back
    from the scaled one, the non-unit scale witness of SEP 36 section 7.2.
    """
    simulation = build("mixed")
    problem = simulation.translate()
    plasma = bool(problem.responses.plasma_channels)
    kappa = plasma_kappa(reference(build), problem) if plasma else CUBIC
    with ExitStack() as cleanup:
        mixed = cast(NumpyMixedBackend, loaded(simulation, cleanup))
        double = cast(NumpyRefBackend, loaded(build("double"), cleanup))
        if exponent is not None:
            mixed._scale = math.ldexp(1.0, exponent)
            mixed._scale_exponent = exponent
        field = mixed._require_state()
        actual = stage_source(mixed, field, 0.0)
        expected = stage_source(double, field, 0.0)
    passes = source_pass_bound(
        problem.grids.n_to, scans=2 if plasma else 0, transverse=transverse
    )
    assert actual.dtype == np.complex128
    assert_within(actual, expected, kappa * U32 + passes)


@pytest.mark.parametrize("steps", [1, 10])
def test_the_oracle_agrees_with_binary64_within_the_mixed_bound(steps):
    """The fibre route with Kerr, Raman and shock: one 5 mm step, and ten.

    The single step is the ten-step run's first, a nonlinear phase of 0.055: one step of
    the whole 5 cm (0.55) would carry the bound's transport factor past informative.
    """

    def build(p: str) -> Simulation:
        return gnlse(p, steps=steps, length=0.005 * steps)

    compare(build, bound_of(build))


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda p: gnlse(p, taper=True), id="taper"),
        pytest.param(
            lambda p: gnlse(p, integrator=LowStorageRungeKutta()), id="low-storage"
        ),
        pytest.param(
            lambda p: capillary(p, modes=4, length=0.002, energy=20e-6, plasma=False),
            id="multimode",
        ),
        pytest.param(
            lambda p: capillary(p, envelope=True, plasma=False), id="envelope"
        ),
        pytest.param(
            lambda p: capillary(p, envelope=True, plasma=False, thg=True),
            id="envelope-harmonic",
        ),
    ],
)
def test_the_cubic_routes_agree_with_binary64_within_the_mixed_bound(build):
    """Each takes a path of the oracle the soliton fibre does not."""
    compare(build, bound_of(build))


@pytest.mark.parametrize(
    ("build", "transverse"),
    [
        pytest.param(lambda p: spatial_kerr("radial", p), 32, id="radial"),
        pytest.param(lambda p: spatial_kerr("cartesian", p), 8, id="cartesian"),
        pytest.param(lambda p: spatial_kerr("line", p), 4, id="line"),
    ],
)
def test_the_spatial_routes_agree_with_binary64_within_the_mixed_bound(
    build, transverse
):
    """The radial Hankel product over 32 nodes, the Cartesian and line transforms."""
    compare(build, bound_of(build, transverse=transverse))


def ionising(precision: str, **overrides) -> Simulation:
    """Return the ionising capillary's first 0.25 mm on 64 steps.

    Its first 0.25 mm step has a nonlinear phase of 0.5 and the plasma's condition
    number reaches 17, so over the class R tests' 16 steps of 4 mm the bound's transport
    factor ``(1 + kappa phi)**7`` makes it uninformative; this shorter window, with a
    phase near 0.008 per step, is the one it is compared on (SEP 36 section 7.1).
    """
    length = 2.5e-4
    schedule = FixedSchedule.uniform(0.0, length, length / 64)
    return capillary(precision, length=length, schedule=schedule, **overrides)


@pytest.mark.parametrize(
    ("build", "transverse"),
    [
        pytest.param(ionising, 0, id="capillary"),
        pytest.param(
            lambda p: ionising(p, gas="N2", wavelength_limits=(200e-9, 3e-6)),
            0,
            id="capillary-raman",
        ),
        pytest.param(
            lambda p: ionising(p, gas={"Ar": 0.5, "He": 0.5}), 0, id="mixture"
        ),
        pytest.param(plasma_beam, 32, id="radial-filament"),
    ],
)
def test_the_field_grid_plasma_agrees_with_binary64_within_the_mixed_bound(
    build, transverse
):
    """The plasma's condition number is read from its rate (SEP 36 7.1, item 1)."""
    compare(build, bound_of(build, transverse=transverse))


def test_the_envelope_plasma_runs_on_the_mixed_oracle():
    """Two steps of a 1 mm envelope-plasma capillary on 256 samples.

    The oracle's cycle moments are binary64 host evaluations of the narrowed field
    (SEP 36 section 4.5), not a matched binary32 sequence, so the comparison is the
    class T envelope SEP 33's test fixed for this case.
    """
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
        mixed, double = (loaded(build(p), cleanup) for p in ("mixed", "double"))
        for backend in (mixed, double):
            backend.propagate_to(length / 4)
        fields = [backend.observe().field for backend in (mixed, double)]
    assert fields[0] is not None and fields[1] is not None
    assert_within(fields[0], fields[1], 1e-5)


def test_a_batch_member_is_its_solo_run():
    """Members agree with their solo runs within the mixed bound."""
    with ExitStack() as cleanup:
        members = tuple(loaded(gnlse("mixed"), cleanup) for _ in range(2))
        runner = select_batch(cast(Sequence[Backend], members))
        cleanup.callback(runner.close)
        runner.propagate_to(0.05)
        batch = [member.field for member in runner.observe_members()]
    solo = finish(gnlse("mixed"))
    bound = 2 * bound_of(gnlse)
    for field in batch:
        assert_within(field, solo, bound)


# Independent witnesses: the oracle against physics.


@pytest.mark.parametrize("precision", ["single", "mixed"])
def test_dispersion_free_self_phase_modulation_is_the_closed_form(precision):
    """``A(t) exp(i gamma |A(t)|**2 L)``: the phase ``P gamma L`` per sample.

    The binary64 run's own departure from the closed form (the march's truncation) is
    measured first; the reduced run may depart by that plus its precision's bound.
    """
    length = 0.05

    def build(p: str) -> Simulation:
        return gnlse(p, raman=False, shock=False, betas=[0.0, 0.0, 0.0], length=length)

    problem = build("double").translate()
    start = np.fft.ifft(problem.initial.field0[0, 0, 0])
    exact = start * np.exp(1j * 0.11 * np.abs(start) ** 2 * length)

    def time(field: np.ndarray) -> np.ndarray:
        return np.fft.ifft(field[0, 0, 0])

    double = assert_within(time(finish(build("double"))), exact, 1.0) * 1.0
    if precision == "mixed":
        budget = bound_of(build)
    else:
        depth = math.ceil(math.log2(problem.grids.n_to))
        budget = 10 * (2 * U32 + 7 * 16 * 2 * U32 * depth)
    assert_within(time(finish(build(precision))), exact, double + budget)


@pytest.mark.parametrize("precision", ["single", "mixed"])
def test_lossless_kerr_propagation_conserves_the_photon_number(precision):
    """Kerr with self-steepening conserves ``sum |A(omega)|**2 / omega``."""
    simulation = gnlse(precision, raman=False)
    problem = simulation.translate()
    omega = problem.grids.omega + problem.grids.omega0
    active = problem.grids.active_mask

    def photons(field: np.ndarray) -> float:
        return float(np.sum(np.abs(field[..., active]) ** 2 / omega[active]))

    start = photons(problem.initial.field0)
    double = abs(photons(finish(gnlse("double", raman=False))) / start - 1)
    change = abs(photons(finish(simulation)) / start - 1)
    budget = (
        2 * bound_of(lambda p: gnlse(p, raman=False))
        if precision == "mixed"
        else 2 * 10 * (2 * U32 + 7 * 16 * 2 * U32 * 10)
    )
    assert change <= double + budget


# The linear part: binary64 under mixed precision, both absorber schemes.


def vacuum(precision: str, scheme: str) -> Simulation:
    """Return an evacuated capillary: its collar under `scheme`, and no source.

    Under the in-stage scheme the collar's reference length is 0.1 m, a rate of
    126 1/m and an absorber exponent of 0.016 per step. The default collar's 6.3e4 1/m
    (2.6 per step at the in-stage limit) makes SEP 36's transport factor
    ``(1 + kappa phi)**7`` exceed 1e3, so its bound is uninformative over any window
    (section 7.1) and the class T study qualifies it.
    """
    length = 0.004
    collar = None
    if scheme == "nonlinear":
        grid = capillary(precision, pressure=0.0, plasma=False).grid
        collar = absorber(grid, boundary_length=0.1)
    return capillary(
        precision,
        pressure=0.0,
        plasma=False,
        absorber_scheme=scheme,
        absorber=collar,
        length=length,
        schedule=FixedSchedule.uniform(0.0, length, length / 16),
    )


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda p: gnlse(p, gamma=0.0, raman=False, length=0.5), id="fibre"
        ),
        pytest.param(
            lambda p: vacuum(p, "split_step"),
            id="fibre-split-step",
        ),
        pytest.param(lambda p: spatial_kerr("cartesian", p, chi3=None), id="cartesian"),
        pytest.param(
            lambda p: spatial_kerr(
                "cartesian",
                p,
                chi3=None,
                absorbed=True,
                scheme="split_step",
                window_factor=0.0,
            ),
            id="cartesian-split-step",
        ),
    ],
)
def test_a_source_free_run_agrees_with_binary64_within_binary64_bounds(build):
    """No source, or a split-step collar: every operation is the reference's.

    The split-step collar transforms the state in binary64 (SEP 36 section 4.2). The
    Cartesian grid's spatial collar is always an in-stage term, so its split-step case
    takes the grid without one; the vacuum capillary's Kerr term has no density.
    """
    problem = build("mixed").translate()
    geometry = problem.transform.geometry
    assert geometry is None or not np.any(geometry.absorber_rate)
    assert problem.stepper.schedule is not None
    steps = len(problem.stepper.schedule.steps)
    depth = math.ceil(math.log2(problem.grids.n_to)) + 8
    bound = steps * (2 * EPS + 4 * 16 * EPS * depth)
    compare(build, bound)


def test_an_in_stage_temporal_collar_agrees_within_the_mixed_bound():
    """The evacuated capillary's in-stage collar, a binary32 source term (kappa 1)."""

    def build(precision: str) -> Simulation:
        return vacuum(precision, "nonlinear")

    compare(build, bound_of(build, kappa=1.0))


@pytest.mark.parametrize(
    ("build", "transverse"),
    [
        pytest.param(
            lambda p: spatial_kerr("cartesian", p, chi3=None, absorbed=True),
            8,
            id="cartesian",
        ),
    ],
)
def test_in_stage_spatial_collars_agree_within_the_mixed_bound(build, transverse):
    """The Cartesian grid's spatial collar beside a weak temporal one (kappa 1)."""
    compare(build, bound_of(build, kappa=1.0, transverse=transverse))


# Damping: no restriction under mixed precision.


def test_a_strongly_damped_bin_runs_and_agrees_with_binary64():
    """A 1 mm capillary's collar on 0.25 mm steps: |Re dPhi| = 150 per step.

    Single precision refuses it (``exp(150)`` overflows binary32 in the inverse
    propagator); under mixed precision the inverse propagator multiplies binary64
    values only, and the stored stage values hold ``exp(150)`` times a small source.
    """
    length = 0.001
    schedule = FixedSchedule.uniform(0.0, length, length / 4)

    def build(precision: str) -> Simulation:
        return capillary(precision, length=length, schedule=schedule, plasma=False)

    simulation = build("mixed")
    problem = simulation.translate()
    with ExitStack() as cleanup:
        backend = cast(NumpyMixedBackend, loaded(simulation, cleanup))
        backend.propagate_to(length / 4)
        damped = max(float(np.max(np.abs(stage))) for stage in backend._last_stages)
    assert damped > float(np.finfo(np.float32).max)
    step = delta_phi(problem.linear, 0.0, length / 4)
    assert float(np.max(np.abs(np.real(step)))) > 149.0
    compare(build, bound_of(build))


# The dtype guards, adversarially.


def test_a_binary64_array_inside_the_source_is_refused():
    with ExitStack() as cleanup:
        backend = cast(NumpyMixedBackend, loaded(gnlse("mixed"), cleanup))
        wide = NumpyRefBackend._prefactor
        backend._prefactor = lambda bundle: wide(backend, bundle)
        with pytest.raises(BackendError, match="binary64 array reached the binary32"):
            backend.propagate_to(0.05)


def test_a_binary32_array_in_a_stored_row_is_refused():
    with ExitStack() as cleanup:
        backend = cast(NumpyMixedBackend, loaded(gnlse("mixed"), cleanup))
        source = backend._nonlinear

        def narrowed(lab, bundle, inverse, **keywords):
            return source(lab, bundle, inverse, **keywords).astype(np.complex64)

        backend._nonlinear = narrowed
        with pytest.raises(BackendError, match="binary32 array reached a row"):
            backend.propagate_to(0.05)


def test_a_binary32_stage_input_and_adams_state_are_refused():
    with ExitStack() as cleanup:
        backend = cast(NumpyMixedBackend, loaded(gnlse("mixed"), cleanup))
        state = backend._require_state().astype(np.complex64)
        with pytest.raises(BackendError, match="the stage input"):
            stage_source(backend, state, 0.0)
        with pytest.raises(BackendError, match="the Adams state"):
            backend._adams_adopt(state, 1e-3)


# The study's routes and the memory estimate.


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda p: gnlse(p, steps=None, integrator=Multistep()), id="constant"
        ),
        pytest.param(lambda p: gnlse(p, steps=None), id="runge-kutta"),
    ],
)
def test_adaptive_marches_run_under_the_study_admission(build):
    """Adams is refused publicly; the study measures it (Q2) against binary64 Adams.

    At the mixed floor both reach the tolerance's truncation; the envelope is the
    class T one SEP 33 fixed for its own Adams measurement.
    """
    with study_admission():
        mixed, double = finish(build("mixed")), finish(build("double"))
    assert_within(mixed, double, 1e-4)


def test_the_memory_estimate_keeps_the_stored_rows_binary64():
    """SEP 36 section 6.7: the rows beside the state stay, the source's halve."""
    problem = gnlse("mixed").translate()
    double = memory._reference_base(problem)
    rows = mixed_precision_rows(problem, double)
    for name in ("state", "stages", "propagators", "accumulator_solution"):
        assert rows[name] == double[name]
    assert rows["fine_spectrum"] == double["fine_spectrum"] // 2
    assert rows["mixed_narrowed_input"] == 8 * math.prod(problem.shape)
    totals = {
        p: estimate_memory(gnlse(p).translate(), "numpy_ref").total
        for p in ("double", "single", "mixed")
    }
    assert totals["single"] < totals["mixed"] < totals["double"]
