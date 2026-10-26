# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Single precision on the native backend's fibre route (SEP 36 phase 2, section 7.2).

Class R comparisons hold native's binary32 run against the ``numpy_ref`` oracle's on the
same fixed schedule with a fixed-order method, inside the sum of both sides' bounds at
binary32's unit roundoff (SEP 36 section 7.1): each side is within the oracle tests'
bound of the binary64 result, so the two are within twice it. Native accumulates its
running integrals and its norms in binary64 (section 4.5), so its own side's bound is
the oracle's. The class P tests are exact; the targeted tests are SEP 33's on native;
and the routes phase 3 adds are refused by name.
"""

import dataclasses
import math
import warnings
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from typing import cast

import h5py
import numpy as np
import pytest

from bounds import EPS_32, UNIT_ROUNDOFF_32, assert_within, higham_gamma
from precision_cases import capillary, gnlse, plasma_beam, spatial_kerr
from solprop.backends.memory import estimate_memory
from solprop.backends.protocol import Backend
from solprop.backends.reduced import LaunchScale
from solprop.backends.registry import available, native_precisions, select, select_batch
from solprop.constants import EPS0
from solprop.driver import Simulation, prop_gnlse
from solprop.exceptions import (
    BackendError,
    ConvergenceError,
    DriverError,
    TranslationError,
)
from solprop.log import NullReporter
from solprop.stats import PLASMA_INTEGRATION_STATISTICS, StatsSpec
from solprop.stepper import LowStorageRungeKutta, PIController, RungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.precision import combine, precision_named
from vector_raman_cases import vector_problem

pytestmark = pytest.mark.skipif(
    "native" not in available() or "single" not in native_precisions(),
    reason="the native backend is not installed or its build has no single precision",
)

U32 = UNIT_ROUNDOFF_32


def loaded(simulation: Simulation, backend: str, cleanup: ExitStack) -> Backend:
    """Return the simulation's backend `backend` with its launch loaded."""
    problem = simulation.translate()
    selected = select(backend, problem, grid=simulation.grid)
    cleanup.callback(selected.close)
    selected.set_initial(problem.initial.field0, None)
    return selected


def finish(simulation: Simulation, backend: str) -> np.ndarray:
    """Run to the last save plane on `backend` and return the observed field."""
    with ExitStack() as cleanup:
        selected = loaded(simulation, backend, cleanup)
        selected.propagate_to(float(simulation.settings.save_planes[-1]))
        return selected.observe().require_field().copy()


def step_bound(steps: int, depth: int) -> float:
    """Return the oracle tests' bound for `steps` steps of transforms of `depth`."""
    return steps * (2 * U32 + 7 * 16 * EPS_32 * depth)


def plasma_bound(steps: int, depth: int) -> float:
    """The same with the plasma's running integrals of `depth` (SEP 25 section 4.6)."""
    return steps * (2 * U32 + 7 * (16 * EPS_32 * depth + 4 * higham_gamma(depth, U32)))


def two_sides(bound: float) -> float:
    """Return the bound between two runs each within `bound` of the binary64 one."""
    return 2 * bound


# Class R: native against the oracle, both in binary32, on the same schedule.

FIBRE = math.ceil(math.log2(gnlse("double").translate().grids.n_to))


@pytest.mark.parametrize(
    ("build", "bound"),
    [
        pytest.param(lambda p: gnlse(p, steps=1), step_bound(1, FIBRE), id="one-step"),
        pytest.param(gnlse, step_bound(10, FIBRE), id="raman"),
        pytest.param(lambda p: gnlse(p, raman=False), step_bound(10, FIBRE), id="kerr"),
        pytest.param(lambda p: gnlse(p, taper=True), step_bound(10, FIBRE), id="taper"),
        pytest.param(
            lambda p: gnlse(p, integrator=LowStorageRungeKutta()),
            step_bound(10, FIBRE),
            id="low-storage",
        ),
        pytest.param(capillary, plasma_bound(16, 13), id="field-plasma-band"),
        pytest.param(
            lambda p: capillary(p, plasma_integration="trapezoid"),
            plasma_bound(16, 13),
            id="field-plasma-trapezoid",
        ),
        pytest.param(
            lambda p: capillary(p, gas={"Ar": 0.5, "He": 0.5}),
            plasma_bound(16, 13),
            id="plasma-mixture",
        ),
        pytest.param(
            lambda p: capillary(p, modes=4, length=0.002, energy=20e-6),
            step_bound(16, 13 + 32),
            id="multimode",
        ),
        pytest.param(
            lambda p: capillary(p, envelope=True, plasma=False),
            step_bound(16, 13 + 32),
            id="envelope",
        ),
        pytest.param(
            lambda p: capillary(p, envelope=True, plasma=False, thg=True),
            step_bound(16, 13 + 32),
            id="envelope-harmonic",
        ),
        pytest.param(
            lambda p: capillary(
                p, gas="N2", thg=True, wavelength_limits=(200e-9, 3e-6)
            ),
            step_bound(16, 13 + 32),
            id="field-raman",
        ),
        pytest.param(
            # The in-stage collar's explicit stability limit asks for 40 steps here.
            lambda p: capillary(
                p,
                absorber_scheme="nonlinear",
                plasma=False,
                schedule=FixedSchedule.uniform(0.0, 0.004, 0.004 / 40),
            ),
            step_bound(40, 13),
            id="in-stage-absorber",
        ),
    ],
)
def test_native_agrees_with_the_oracle_within_both_sides_bounds(build, bound):
    """The fibre route's capabilities, each against the oracle (SEP 36 section 7.2)."""
    native = finish(build("single"), "native")
    oracle = finish(build("single"), "numpy_ref")
    assert_within(native, oracle, two_sides(bound))


@pytest.mark.parametrize("weighted", [False, True])
def test_an_adaptive_run_keeps_its_accuracy_envelope(weighted):
    """The controller reads binary64 norms of the scaled state against ``atol / s``.

    Adaptive runs take their own steps, so native and the binary64 reference are
    compared within an accuracy envelope (SEP 25), ten times the relative tolerance,
    not a rounding bound; the per-bin weights take the weighted controller's path.
    """
    reference = finish(gnlse("double", steps=None), "numpy_ref")
    simulation = gnlse("single", steps=None)
    rtol = simulation.settings.controller.rtol
    if weighted:
        n_omega = gnlse("single").translate().shape[-1]
        rtol = 1e-5
        simulation.settings = dataclasses.replace(
            simulation.settings,
            controller=PIController(rtol=rtol, atol=1e-6, weights=np.ones(n_omega)),
        )
    assert_within(finish(simulation, "native"), reference, 10 * rtol)


@pytest.mark.parametrize(
    ("family", "material"),
    [("envelope", "H2"), ("field", "H2"), ("field_real", "H2"), ("envelope", "CH4")],
)
def test_tensor_raman_agrees_with_the_oracle(family, material):
    """Both helicities through H2's tensor ladder, and methane's isotropic kernel."""
    problem, grid, modes = vector_problem(family=family, material=material, steps=8)
    case = dataclasses.replace(problem, precision=precision_named("single"))
    fields = {}
    for name in ("native", "numpy_ref"):
        with ExitStack() as cleanup:
            backend = select(name, case, grid=grid, modes=modes)
            cleanup.callback(backend.close)
            backend.set_initial(case.initial.field0, None)
            backend.propagate_to(float(case.stepper.save_planes[-1]))
            fields[name] = backend.observe().require_field().copy()
    depth = math.ceil(math.log2(2 * grid.n_t)) + 4
    assert_within(
        fields["native"], fields["numpy_ref"], two_sides(step_bound(8, depth))
    )


def test_a_batch_member_is_its_solo_run():
    """Lockstep members of one precision agree with their solo runs (section 6.6)."""
    with ExitStack() as cleanup:
        members = tuple(loaded(gnlse("single"), "native", cleanup) for _ in range(2))
        runner = select_batch(cast(Sequence[Backend], members))
        cleanup.callback(runner.close)
        runner.propagate_to(0.05)
        batch = [member.field for member in runner.observe_members()]
    solo = finish(gnlse("single"), "native")
    for field in batch:
        assert_within(field, solo, step_bound(10, FIBRE))


def test_a_batch_refuses_members_of_two_precisions():
    with ExitStack() as cleanup:
        members = (
            loaded(gnlse("single"), "native", cleanup),
            loaded(gnlse("double"), "native", cleanup),
        )
        with pytest.raises(Exception, match="one precision"):
            select_batch(cast(Sequence[Backend], members)).propagate_to(0.005)


# Class P: exact relations of the scaled state.


@pytest.mark.parametrize("build", [gnlse, capillary], ids=["gnlse", "capillary"])
def test_scaling_by_a_power_of_two_is_exact(build, monkeypatch):
    """Runs at other launch scales differ by powers of two alone (SEP 33 section 4.3).

    The state is held as ``a / s`` and every coefficient carries ``s``'s powers exactly,
    so the observed planes are the same bits whatever ``s``. A first run plans its
    transforms before FFTW's process-local wisdom holds the measured plans a later run
    reuses, so it may round differently (SEP 25: no bitwise agreement across plans);
    it runs first, and the scales are compared on the same plans.
    """
    from solprop_native import backend as native_backend  # noqa: PLC0415

    simulation = build("single")
    finish(simulation, "native")
    natural = LaunchScale.of(simulation.translate())
    results = []
    for exponent in (natural.exponent, 0, 11):
        pinned = LaunchScale(math.ldexp(1.0, exponent), exponent)
        launch = type("Pinned", (), {"of": staticmethod(lambda _, s=pinned: s)})
        monkeypatch.setattr(native_backend, "LaunchScale", launch)
        results.append(finish(simulation, "native"))
    assert np.array_equal(results[0], results[1])
    assert np.array_equal(results[0], results[2])


def test_the_state_is_held_at_the_launch_scale_and_observed_exactly():
    """The launch divided by ``s`` exactly and rounded once, and observed times ``s``.

    The oracle holds the same binary32 state, so the two observations are equal.
    """
    simulation = gnlse("single")
    with ExitStack() as cleanup:
        backend = loaded(simulation, "native", cleanup)
        observed = backend.observe().require_field()
        oracle = loaded(simulation, "numpy_ref", cleanup)
        assert np.array_equal(observed, oracle.observe().require_field())
        # Every observed value is a binary32 value times the power of two ``s = 8``.
        assert np.array_equal(
            (observed / 8.0).astype(np.complex64).astype(np.complex128) * 8.0, observed
        )
        # A reload of an observed plane holds the same binary32 state.
        backend.propagate_to(0.02)
        plane = backend.observe().require_field().copy()
        backend.set_initial(plane, None, z=0.02)
        assert np.array_equal(backend.observe().require_field(), plane)


def test_a_zero_launch_stays_zero():
    simulation = gnlse("single", raman=False)
    with ExitStack() as cleanup:
        backend = loaded(simulation, "native", cleanup)
        backend.set_initial(np.zeros(simulation.translate().shape, np.complex128), None)
        backend.propagate_to(0.05)
        assert not np.any(backend.observe().require_field())


# Targeted tests: SEP 33's on native.


def test_a_molecular_raman_response_survives_narrowing():
    """Nitrogen's kernel spectrum of order 1e-50 is narrowed with its exponent kept.

    Rounded directly it would vanish; native's nitrogen run then differs from the same
    run without the molecule's response by far more than their bound.
    """

    def nitrogen(precision: str, *, raman: bool = True) -> Simulation:
        return capillary(
            precision,
            gas="N2",
            energy=20e-6,
            wavelength_limits=(200e-9, 3e-6),
            raman=raman,
        )

    bound = two_sides(step_bound(16, 13 + 32))
    native = finish(nitrogen("single"), "native")
    assert_within(native, finish(nitrogen("single"), "numpy_ref"), bound)
    # The response changes the field far beyond the rounding sixteen steps can carry.
    without = finish(nitrogen("single", raman=False), "native")
    change = np.linalg.norm(native - without) / np.linalg.norm(native)
    assert change > 1e3 * 16 * U32


def test_the_plasma_populations_agree_with_the_oracle():
    """The ionised fraction and the electron density in SEP 33's stable forms."""
    spec = StatsSpec(names=("energy", "electron_density"))
    results = {}
    for name in ("native", "numpy_ref"):
        with ExitStack() as cleanup:
            backend = loaded(capillary("single"), name, cleanup)
            backend.propagate_to(0.004)
            results[name] = backend.stats(spec)
    for key in ("energy", "electron_density"):
        assert_within(
            np.asarray(results["native"][key]),
            np.asarray(results["numpy_ref"][key]),
            two_sides(plasma_bound(16, 13)),
            what=key,
        )


def test_the_plasma_integration_diagnostics_are_formed_in_binary64():
    """SEP 18's run diagnostics scan binary64 rows of the physical field (SEP 33 4.5).

    The undershoot and the boundary level are compared with the oracle's as the
    populations are. The work defect ``|paid - freed|`` is a difference of nearly
    equal integrals, so it is compared with native's binary64 run instead: binary32
    rows would floor it at their rounding, about ``u32`` times the paid work, far above
    the run's own defect.
    """
    spec = StatsSpec(names=PLASMA_INTEGRATION_STATISTICS)
    results = {}
    runs = (("single", "native"), ("single", "numpy_ref"), ("double", "native"))
    for precision, name in runs:
        with ExitStack() as cleanup:
            backend = loaded(capillary(precision), name, cleanup)
            backend.propagate_to(0.004)
            results[precision, name] = {
                key: np.asarray(value) for key, value in backend.stats(spec).items()
            }
    native, oracle = results["single", "native"], results["single", "numpy_ref"]
    bound = two_sides(plasma_bound(16, 13))
    undershoot = native["plasma_undershoot"], oracle["plasma_undershoot"]
    assert_within(*undershoot, bound, what="plasma_undershoot")
    # The pulse leaves no fraction at the window's edges: a zero level, in both.
    assert not np.any(oracle["plasma_boundary_level"])
    assert not np.any(native["plasma_boundary_level"])
    defect = native["plasma_work_defect"]
    reference = results["double", "native"]["plasma_work_defect"]
    assert np.all(reference > 0.0)
    assert_within(defect, reference, 1e-2, what="plasma_work_defect")


def test_the_propagator_drift_is_linear_in_the_step_count():
    """A constant operator and no source: each step's binary32 propagator.

    SEP 36 section 4.3's propagator is within ``10 u32`` of the binary64 value and the
    state's product adds its own rounding (``2 u32`` for a complex product), so the
    drift after ``M`` steps is within ``12 M u32``.
    """
    linear = {"gamma": 0.0, "raman": False, "length": 0.5}
    for steps in (10, 100, 1000):
        exact = finish(gnlse("double", steps=steps, **linear), "numpy_ref")
        actual = finish(gnlse("single", steps=steps, **linear), "native")
        assert_within(actual, exact, 12 * steps * U32, what=f"{steps} steps")


@pytest.mark.parametrize("march", ["adaptive", "replayed", "low_storage"])
def test_a_non_finite_state_stops_every_march(march):
    simulation = {
        "adaptive": gnlse("single", steps=None),
        "replayed": gnlse("single"),
        "low_storage": gnlse("single", integrator=LowStorageRungeKutta()),
    }[march]
    with ExitStack() as cleanup:
        backend = loaded(simulation, "native", cleanup)
        field = simulation.translate().initial.field0.copy()
        # The launch's strongest bin, which the band keeps.
        field[..., int(np.argmax(np.abs(field[0, 0, 0])))] = np.nan
        backend.set_initial(field, None)
        with pytest.raises((ConvergenceError, BackendError)):
            backend.propagate_to(0.05)


def test_a_resume_in_the_other_precision_is_refused(tmp_path):
    path = tmp_path / "run.h5"
    keywords = {
        "lambda0": 835e-9, "wavelength_limits": (500e-9, 2e-6), "trange": 1e-12,
        "width": 50e-15, "peak_power": 10.0, "pulseshape": "sech", "n_save": 2,
        "backend": "native", "integrator": RungeKutta(), "progress": "none",
    }  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prop_gnlse(0.11, 0.01, [0.0, 0.0, -1e-26], output=path, **keywords)
        with pytest.raises(Exception, match="precision"):
            prop_gnlse(
                0.11, 0.01, [0.0, 0.0, -1e-26], output=path, resume=True,
                precision="single", **keywords,
            )  # fmt: skip


class _StoppedError(RuntimeError):
    """What an interrupted run raises, standing in for a machine going away."""


class _StopAfter(NullReporter):
    """Stop a run once it has saved `after` planes."""

    def __init__(self, after: int) -> None:
        self.after = after
        self.seen = 0

    def plane(self, z, summary):
        self.seen += 1
        if self.seen >= self.after:
            raise _StoppedError


def _resumable(backend: str) -> Simulation:
    """A single-precision soliton fibre on a replayed schedule with four planes."""
    length = 0.02
    planes = list(np.linspace(0.0, length, 5)[1:])
    return Simulation.from_gnlse(
        0.11, length, [0.0, 0.0, -1.183e-26, 8.1e-41], lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6), trange=1e-12, width=50e-15,
        peak_power=100.0, pulseshape="sech", save_planes=planes, backend=backend,
        precision="single", integrator=RungeKutta(),
        schedule=FixedSchedule.uniform(0.0, length, length / 16, save_planes=planes),
    )  # fmt: skip


@pytest.mark.parametrize(
    ("first", "second"), [("numpy_ref", "native"), ("native", "numpy_ref")]
)
def test_a_run_continues_across_backends_in_the_same_precision(tmp_path, first, second):
    """Class T: a file started on one backend resumes on the other, both binary32.

    The resume key holds the problem and its precision, not the backend; the continued
    run is the same replayed march, so it agrees with the uninterrupted run within the
    bound between two binary32 runs of its sixteen steps.
    """
    whole, part = tmp_path / "whole.h5", tmp_path / "part.h5"
    _resumable(first).run(whole, progress="none")
    with pytest.raises(_StoppedError):
        _resumable(first).run(part, progress=_StopAfter(2))
    _resumable(second).run(part, resume=True, progress="none")
    assert np.array_equal(_stored(whole, "z"), _stored(part, "z"))
    bound = two_sides(step_bound(16, FIBRE))
    assert_within(_stored(part, "field"), _stored(whole, "field"), bound)


def _stored(path, key: str) -> np.ndarray:
    """Return one stored array of a closed output file."""
    with h5py.File(path, "r") as file:
        dataset = file[key]
        assert isinstance(dataset, h5py.Dataset)
        return np.asarray(dataset[()])


@pytest.mark.parametrize("build", [gnlse, capillary], ids=["gnlse", "capillary"])
def test_single_precision_takes_less_memory_by_the_ledger_and_the_estimate(build):
    """The buffers native allocates, and solprop's estimate of them, both shrink.

    The FFT planners' measured heap growth is left out: it depends on what the
    process planned before, not on the problem.
    """
    ledgers, estimates = {}, {}
    for precision in ("single", "double"):
        simulation = build(precision)
        estimates[precision] = estimate_memory(simulation.translate(), "native").total
        with ExitStack() as cleanup:
            ledger = loaded(simulation, "native", cleanup).memory_ledger()
        ledger.pop("fft_plan_heap_growth", None)
        ledgers[precision] = sum(ledger.values())
    assert ledgers["single"] < ledgers["double"]
    assert estimates["single"] < estimates["double"]


def test_the_memory_ledger_counts_binary32_buffers():
    """The state's rows are half binary64's; the operator's terms keep their size."""
    ledgers = {}
    for precision in ("single", "double"):
        with ExitStack() as cleanup:
            ledgers[precision] = loaded(
                gnlse(precision), "native", cleanup
            ).memory_ledger()
    single, double = ledgers["single"], ledgers["double"]
    # The planners' measured heap growth depends on what the process planned before.
    for ledger in (single, double):
        ledger.pop("fft_plan_heap_growth", None)
    for row in ("state", "stages", "fine_spectrum", "projected_spectrum"):
        assert 2 * single[row] == double[row], row
    assert single["fixed_phi_terms"] == double["fixed_phi_terms"]
    assert sum(double.values()) / 2 < sum(single.values()) < sum(double.values())


# Refusals: what phase 3 adds, and a build without single precision.


def spatial_depth(build: Callable[[str], Simulation], transverse: int) -> int:
    """The temporal transform's depth plus the route's transverse one."""
    return math.ceil(math.log2(build("double").grid.n_t)) + transverse


@pytest.mark.parametrize(
    ("build", "transverse"),
    [
        pytest.param(lambda p: spatial_kerr("radial", p), 32, id="radial"),
        pytest.param(
            lambda p: spatial_kerr("radial", p, absorbed=True), 32, id="radial-collars"
        ),
        pytest.param(lambda p: spatial_kerr("cartesian", p), 8, id="cartesian"),
        pytest.param(lambda p: spatial_kerr("line", p), 4, id="line"),
        pytest.param(
            lambda p: spatial_kerr("cartesian", p, absorbed=True),
            8,
            id="cartesian-collars",
        ),
        pytest.param(
            lambda p: spatial_kerr(
                "cartesian", p, absorbed=True, scheme="split_step", window_factor=0.0
            ),
            8,
            id="cartesian-split-step",
        ),
        pytest.param(
            lambda p: spatial_kerr("cartesian", p, integrator=LowStorageRungeKutta()),
            8,
            id="cartesian-low-storage",
        ),
        pytest.param(
            lambda p: spatial_kerr(
                "cartesian", p, absorbed=True, integrator=LowStorageRungeKutta()
            ),
            8,
            id="cartesian-low-storage-collars",
        ),
    ],
)
def test_the_spatial_routes_agree_with_the_oracle(build, transverse):
    """Ten steps of a self-focusing Kerr beam; a transverse transform per stage.

    The radial route's Hankel GEMM over ``n_r`` nodes and the free-space routes'
    transforms add their depth to the temporal transform's (SEP 33's oracle bound).
    """
    bound = two_sides(step_bound(10, spatial_depth(build, transverse)))
    native = finish(build("single"), "native")
    assert_within(native, finish(build("single"), "numpy_ref"), bound)


@pytest.mark.parametrize("route", ["radial", "cartesian"])
def test_a_plasma_beam_agrees_with_the_oracle(route):
    """The spatial routes' plasma (the held current on nodes), ten steps."""

    def build(p: str) -> Simulation:
        return plasma_beam(p, route=route)

    bound = two_sides(plasma_bound(10, spatial_depth(build, 32)))
    native = finish(build("single"), "native")
    assert_within(native, finish(build("single"), "numpy_ref"), bound)


SPATIAL_STATISTICS = (
    "energy",
    "on_axis_intensity",
    "peak_intensity",
    "on_axis_spectrum",
    "beam_radius",
    "fwhm_r",
)


@pytest.mark.parametrize("route", ["radial", "cartesian", "line"])
def test_the_spatial_statistics_are_binary64_reductions_of_the_state(route):
    """Each statistic reads the physical state widened to binary64 (SEP 36 4.5).

    The oracle reads its own state the same way; the two states are within the class
    R bound of the fields, and a statistic quadratic in the field within twice it.
    """
    spec = StatsSpec(names=SPATIAL_STATISTICS)
    results = {}
    for name in ("native", "numpy_ref"):
        with ExitStack() as cleanup:
            backend = loaded(spatial_kerr(route, "single"), name, cleanup)
            backend.propagate_to(0.05)
            results[name] = backend.stats(spec)
    transverse = {"radial": 32, "cartesian": 8, "line": 4}[route]
    bound = 2 * two_sides(step_bound(5, spatial_depth(
        lambda p: spatial_kerr(route, p), transverse
    )))  # fmt: skip
    for key in SPATIAL_STATISTICS:
        assert_within(
            np.asarray(results["native"][key], dtype=np.float64),
            np.asarray(results["numpy_ref"][key], dtype=np.float64),
            bound,
            what=key,
        )


def test_the_envelope_plasma_is_refused_until_built():
    build = capillary("single", envelope=True, plasma=True)
    with pytest.raises(BackendError, match="phase 3"):
        select("native", build.translate())


def test_a_build_without_single_precision_refuses_it_by_name(monkeypatch):
    import solprop_native  # noqa: PLC0415
    from solprop_native import backend as native_backend  # noqa: PLC0415

    record = solprop_native.build_info()
    monkeypatch.setattr(
        native_backend,
        "build_info",
        lambda: dataclasses.replace(record, precisions=("double",)),
    )
    with pytest.raises(BackendError, match="SOLPROP_NATIVE_PRECISIONS"):
        select("native", gnlse("single").translate())


def test_a_device_build_refuses_single_precision_until_phase_four(monkeypatch):
    """A device build, compile-only single included, runs binary64 alone (SEP 36)."""
    import solprop_native  # noqa: PLC0415
    from solprop_native import backend as native_backend  # noqa: PLC0415

    record = dataclasses.replace(
        solprop_native.build_info(), precisions=("double",), device_backend="cuda"
    )
    monkeypatch.setattr(native_backend, "build_info", lambda: record)
    with pytest.raises(BackendError, match=r"cuda build refuses .*phase 4"):
        select("native", gnlse("single").translate())


def test_the_adams_march_is_refused_under_single_precision():
    with pytest.raises(DriverError, match="Adams integrator"):
        Simulation.from_gnlse(
            0.11, 0.05, [0.0, 0.0, -1e-26], lambda0=835e-9, trange=1e-12,
            wavelength_limits=(500e-9, 2e-6), width=50e-15, peak_power=1.0,
            pulseshape="sech", precision="single", backend="native",
            integrator=Multistep(),
        )  # fmt: skip


@pytest.mark.parametrize("values", [[1e-3, 1e-41], [1e39]], ids=["subnormal", "beyond"])
def test_a_table_beyond_binary32_range_is_refused_as_the_oracle_refuses_it(values):
    """Native checks each table the oracle narrows, by the oracle's name and words."""
    from solprop_native import backend as native_backend  # noqa: PLC0415

    from solprop.backends.numpy_ref.single import NumpySingleBackend  # noqa: PLC0415

    problem = gnlse("single").translate()
    table = np.asarray(values)
    payload = {"prefactor_terms": np.empty(0), "time_window": table}
    with pytest.raises(BackendError) as native:
        native_backend._precision_payload(problem, payload)  # pyright: ignore[reportPrivateUsage]
    with ExitStack() as cleanup:
        oracle = cast(NumpySingleBackend, loaded(gnlse("single"), "numpy_ref", cleanup))
        with pytest.raises(BackendError) as expected:
            oracle._upload("t_window", table)  # pyright: ignore[reportPrivateUsage]
    assert str(native.value) == str(expected.value)


def _coefficient_cases() -> list[tuple[list[float], int]]:
    """Factors and exponents over binary64, binary32's edges and the physical scales."""
    rng = np.random.default_rng(36)
    cases: list[tuple[list[float], int]] = [
        ([], 0),
        ([0.0, 3.0e25], 40),
        ([-0.0, 1.0], 0),
        ([-2.5e-22, 2.4e25], 30),
        ([EPS0, 2.4e25], -164),
        ([1.0e300, 1.0e300], 0),
        ([1.0e-300, 1.0e-300], 0),
        ([1.0], 2_000),
        ([1.0], -2_000),
    ]
    tiny, huge = float(np.finfo(np.float32).tiny), float(np.finfo(np.float32).max)
    for edge in (tiny, huge):
        for ulps in range(-3, 4):
            cases.append(([math.ldexp(1.0, 0) + ulps * 2.0**-52, edge], 0))
            cases.append(([float(np.nextafter(edge, 0.0)) * (1 + ulps * 2.0**-30)], 0))
    for _ in range(400):
        count = int(rng.integers(1, 3))
        factors = [
            float(rng.choice([-1.0, 1.0]) * 10.0 ** rng.uniform(-60.0, 60.0))
            for _ in range(count)
        ]
        cases.append((factors, int(rng.integers(-160, 161))))
    return cases


def test_native_coefficients_are_the_oracles_bit_for_bit():
    """Class P: native forms every Kerr and Raman coefficient as solprop's ``combine``.

    ``combine(...).binary32()`` rounds once from the same binary64 mantissa and
    exponent, so native and the oracle apply the same scalars, and refuse the same
    coefficients with the same ``TranslationError``.
    """
    import solprop_native._core as core  # noqa: PLC0415  # pyright: ignore[reportPrivateUsage]

    refused = 0
    for factors, exponent in _coefficient_cases():
        try:
            # The oracle's narrowing warns on its way to an overflow refusal.
            with np.errstate(over="ignore"):
                expected: float | str = combine(*factors, exponent=exponent).binary32()
        except TranslationError as error:
            expected = str(error)
        try:
            found: float | str = core.combined_binary32(factors, exponent)
        except TranslationError as error:
            found = str(error)
        if isinstance(expected, str):
            refused += 1
            assert found == expected, (factors, exponent)
        else:
            assert isinstance(found, float), (factors, exponent, found)
            assert np.float32(found).tobytes() == np.float32(expected).tobytes(), (
                factors,
                exponent,
            )
    assert refused > 10
