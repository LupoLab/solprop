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
from collections.abc import Sequence
from contextlib import ExitStack
from typing import cast

import numpy as np
import pytest

from bounds import EPS_32, UNIT_ROUNDOFF_32, assert_within, higham_gamma
from precision_cases import capillary, gnlse, plasma_beam, spatial_kerr
from solprop.backends.protocol import Backend
from solprop.backends.registry import available, native_precisions, select, select_batch
from solprop.driver import Simulation, prop_gnlse
from solprop.exceptions import BackendError, ConvergenceError, DriverError
from solprop.stats import StatsSpec
from solprop.stepper import LowStorageRungeKutta, PIController, RungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.precision import precision_named
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


def test_the_memory_ledger_counts_binary32_buffers():
    """The state's rows are half binary64's; the operator's terms keep their size."""
    ledgers = {}
    for precision in ("single", "double"):
        with ExitStack() as cleanup:
            ledgers[precision] = loaded(
                gnlse(precision), "native", cleanup
            ).memory_ledger()
    single, double = ledgers["single"], ledgers["double"]
    for row in ("state", "stages", "fine_spectrum", "projected_spectrum"):
        assert 2 * single[row] == double[row], row
    assert single["fixed_phi_terms"] == double["fixed_phi_terms"]
    assert sum(double.values()) / 2 < sum(single.values()) < sum(double.values())


# Refusals: what phase 3 adds, and a build without single precision.


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: spatial_kerr("radial", "single"), id="radial"),
        pytest.param(lambda: spatial_kerr("cartesian", "single"), id="cartesian"),
        pytest.param(lambda: spatial_kerr("line", "single"), id="line"),
        pytest.param(lambda: plasma_beam("single"), id="radial-plasma"),
        pytest.param(
            lambda: capillary("single", envelope=True, plasma=True),
            id="envelope-plasma",
        ),
    ],
)
def test_the_phase_three_routes_are_refused_by_name(build):
    with pytest.raises(BackendError, match="phase 3"):
        select("native", build().translate())


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


def test_the_adams_march_is_refused_under_single_precision():
    with pytest.raises(DriverError, match="Adams integrator"):
        Simulation.from_gnlse(
            0.11, 0.05, [0.0, 0.0, -1e-26], lambda0=835e-9, trange=1e-12,
            wavelength_limits=(500e-9, 2e-6), width=50e-15, peak_power=1.0,
            pulseshape="sech", precision="single", backend="native",
            integrator=Multistep(),
        )  # fmt: skip
