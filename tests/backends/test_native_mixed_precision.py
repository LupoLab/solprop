# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Mixed precision on the native backend (SEP 36 phase 3, section 7.2).

Class R comparisons hold native's mixed path against the mixed oracle (``numpy_ref``
under ``precision="mixed"``) on the same fixed schedule. Each side is within SEP 36
section 7.1's mixed bound of the binary64 reference, so the two are within the sum of
both sides' bounds; native's running integrals are binary64 (its side's term is the
binary64 one, item 4), so twice the oracle's bound is conservative. The bound is
:func:`backends.test_mixed_precision.bound_of`'s, evaluated from the binary64
reference before any comparison. The linear-part witnesses hold the source-free paths
to binary64 bounds, and the targeted tests reproduce SEP 33's on native.
"""

import dataclasses
import math
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from typing import cast

import h5py
import numpy as np
import pytest

from backends.test_mixed_precision import INFORMATIVE, bound_of, ionising, vacuum
from bounds import EPS, EPS_32, UNIT_ROUNDOFF_32, assert_within
from precision_cases import capillary, envelope_plasma, gnlse, plasma_beam, spatial_kerr
from solprop.backends.driver import StepDriver
from solprop.backends.protocol import Backend
from solprop.backends.reduced import LaunchScale
from solprop.backends.registry import native_precisions, select, select_batch
from solprop.driver import Simulation
from solprop.exceptions import BackendError, ConvergenceError
from solprop.noise import NoiseModel
from solprop.stats import StatsSpec
from solprop.stepper import LowStorageRungeKutta, RungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.linop import delta_phi
from solprop.translate.precision import PrecisionName, precision_named, study_admission
from vector_raman_cases import vector_problem

pytestmark = pytest.mark.skipif(
    "mixed" not in native_precisions(), reason="no native build runs mixed precision"
)

Build = Callable[[str], Simulation]


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


def two_sides(bound: float) -> float:
    """Return the bound between two runs each within `bound` of the binary64 one."""
    return 2 * bound


def against_oracle(build: Build, bound: float, what: str = "") -> float:
    """Assert native mixed is within two sides of `bound` of the mixed oracle."""
    assert bound < INFORMATIVE, f"{what}: the bound {bound:.2e} is uninformative"
    native = finish(build("mixed"), "native")
    oracle = finish(build("mixed"), "numpy_ref")
    return assert_within(native, oracle, two_sides(bound), what=what)


# Class R: native mixed against the mixed oracle, every capability of the fibre route.


@pytest.mark.parametrize("steps", [1, 10])
def test_the_fibre_agrees_with_the_oracle(steps):
    """The soliton fibre with Kerr, Raman and shock: one 5 mm step, and ten."""

    def build(p: str) -> Simulation:
        return gnlse(p, steps=steps, length=0.005 * steps)

    against_oracle(build, bound_of(build))


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda p: gnlse(p, raman=False), id="kerr"),
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
def test_the_cubic_routes_agree_with_the_oracle(build):
    against_oracle(build, bound_of(build))


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(ionising, id="capillary"),
        pytest.param(
            lambda p: ionising(p, gas="N2", wavelength_limits=(200e-9, 3e-6)),
            id="capillary-raman",
        ),
        pytest.param(lambda p: ionising(p, gas={"Ar": 0.5, "He": 0.5}), id="mixture"),
        pytest.param(
            lambda p: ionising(p, plasma_integration="trapezoid"), id="trapezoid"
        ),
    ],
)
def test_the_field_grid_plasma_agrees_with_the_oracle(build):
    """The plasma's condition number is read from its rate (SEP 36 7.1, item 1)."""
    against_oracle(build, bound_of(build))


def test_an_in_stage_temporal_collar_agrees_with_the_oracle():
    """The evacuated capillary's in-stage collar, a binary32 source term (kappa 1)."""

    def build(precision: str) -> Simulation:
        return vacuum(precision, "nonlinear")

    against_oracle(build, bound_of(build, kappa=1.0))


@pytest.mark.parametrize(
    ("family", "material"),
    [("envelope", "H2"), ("field", "H2"), ("field_real", "H2"), ("envelope", "CH4")],
)
def test_tensor_raman_agrees_with_the_oracle(family, material):
    """Both helicities through H2's tensor ladder, and methane's isotropic kernel.

    Bounded by SEP 33's binary32 bound of eight steps on each side, which holds the
    mixed path's binary32 source and leaves its binary64 state's rounding out.
    """
    problem, grid, modes = vector_problem(family=family, material=material, steps=8)
    case = dataclasses.replace(problem, precision=precision_named("mixed"))
    fields = {}
    for name in ("native", "numpy_ref"):
        with ExitStack() as cleanup:
            backend = select(name, case, grid=grid, modes=modes)
            cleanup.callback(backend.close)
            backend.set_initial(case.initial.field0, None)
            backend.propagate_to(float(case.stepper.save_planes[-1]))
            fields[name] = backend.observe().require_field().copy()
    depth = math.ceil(math.log2(2 * grid.n_t)) + 4
    bound = 8 * (2 * UNIT_ROUNDOFF_32 + 7 * 16 * EPS_32 * depth)
    assert_within(fields["native"], fields["numpy_ref"], two_sides(bound))


def test_a_batch_member_is_its_solo_run():
    """Lockstep members of one precision agree with their solo runs (section 6.6)."""
    with ExitStack() as cleanup:
        members = tuple(loaded(gnlse("mixed"), "native", cleanup) for _ in range(2))
        runner = select_batch(cast(Sequence[Backend], members))
        cleanup.callback(runner.close)
        runner.propagate_to(0.05)
        batch = [member.field for member in runner.observe_members()]
    solo = finish(gnlse("mixed"), "native")
    for field in batch:
        assert_within(field, solo, two_sides(bound_of(gnlse)))


def test_a_batch_refuses_members_of_two_precisions():
    with ExitStack() as cleanup:
        members = (
            loaded(gnlse("mixed"), "native", cleanup),
            loaded(gnlse("single"), "native", cleanup),
        )
        with pytest.raises(Exception, match="one precision"):
            select_batch(cast(Sequence[Backend], members)).propagate_to(0.005)


# The linear part: binary64 under mixed precision, both absorber schemes.


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda p: gnlse(p, gamma=0.0, raman=False, length=0.5), id="fibre"
        ),
        pytest.param(lambda p: vacuum(p, "split_step"), id="fibre-split-step"),
    ],
)
def test_a_source_free_run_agrees_with_binary64_within_binary64_bounds(build):
    """No source, or a split-step collar whose transforms are binary64 (4.2).

    Every operation is binary64 native's, so native mixed is held to the binary64
    run of the same backend within a binary64 bound.
    """
    problem = build("mixed").translate()
    assert problem.stepper.schedule is not None
    steps = len(problem.stepper.schedule.steps)
    depth = math.ceil(math.log2(problem.grids.n_to)) + 8
    bound = steps * (2 * EPS + 4 * 16 * EPS * depth)
    mixed, double = finish(build("mixed"), "native"), finish(build("double"), "native")
    assert_within(mixed, double, bound)


# Damping and scale witnesses.


def test_a_strongly_damped_bin_runs_and_agrees_with_the_oracle():
    """A 1 mm capillary's collar on 0.25 mm steps: |Re dPhi| = 150 per step.

    Single precision refuses it; under mixed precision the inverse propagator
    multiplies binary64 values only.
    """
    length = 0.001
    schedule = FixedSchedule.uniform(0.0, length, length / 4)

    def build(precision: str) -> Simulation:
        return capillary(precision, length=length, schedule=schedule, plasma=False)

    step = delta_phi(build("mixed").translate().linear, 0.0, length / 4)
    assert float(np.max(np.abs(np.real(step)))) > 149.0
    against_oracle(build, bound_of(build))


def pinned(monkeypatch, exponent: int) -> None:
    """Hold native's launch scale at ``2**exponent`` instead of the launch's."""
    from solprop_native import backend as native_backend  # noqa: PLC0415

    scale = LaunchScale(math.ldexp(1.0, exponent), exponent)
    launch = type("Pinned", (), {"of": staticmethod(lambda _, s=scale: s)})
    monkeypatch.setattr(native_backend, "LaunchScale", launch)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(gnlse, id="gnlse"),
        pytest.param(capillary, id="plasma"),
        pytest.param(
            lambda p: capillary(p, gas="N2", wavelength_limits=(200e-9, 3e-6)),
            id="raman",
        ),
    ],
)
def test_scaling_by_a_power_of_two_is_exact(build, monkeypatch):
    """The narrowing's scale enters and leaves exactly: class P (SEP 33 4.3).

    A first run plans the transforms before FFTW's process-local wisdom holds what
    later runs reuse, so it runs first and the scales are compared on the same plans.
    """
    simulation = build("mixed")
    finish(simulation, "native")
    natural = LaunchScale.of(simulation.translate()).exponent
    results = []
    for exponent in (natural, 0, 11):
        pinned(monkeypatch, exponent)
        results.append(finish(simulation, "native"))
    assert np.array_equal(results[0], results[1])
    assert np.array_equal(results[0], results[2])


def test_a_launch_scale_of_two_to_the_thirty_gives_the_physical_source(monkeypatch):
    """SEP 36 section 7.2's scale witness: ``s = 2**30`` instead of the launch's.

    The physical result comes back from the scaled source: native at ``s = 2**30`` is
    within two sides of the oracle's bound of the oracle at its own scale.
    """

    def build(p: str) -> Simulation:
        return gnlse(p, steps=10, length=0.05)

    oracle = finish(build("mixed"), "numpy_ref")
    pinned(monkeypatch, 30)
    native = finish(build("mixed"), "native")
    assert_within(native, oracle, two_sides(bound_of(build)))


# The state and its rows.


def test_the_state_is_physical_and_observed_exactly():
    """A mixed state is binary64 and unscaled: a reload of a plane is class P."""
    simulation = gnlse("mixed")
    with ExitStack() as cleanup:
        backend = loaded(simulation, "native", cleanup)
        problem = simulation.translate()
        launch = backend.observe().require_field()
        assert launch.dtype == np.complex128
        assert np.array_equal(
            launch, problem.initial.field0 * problem.grids.active_mask
        )
        backend.propagate_to(0.02)
        plane = backend.observe().require_field().copy()
        backend.set_initial(plane, None, z=0.02)
        assert np.array_equal(backend.observe().require_field(), plane)


def test_an_adams_run_agrees_with_binary64_adams_under_the_study_admission():
    """Adams is refused publicly; the study measures it (Q2) like with like."""

    def build(p: str) -> Simulation:
        return gnlse(p, steps=None, integrator=Multistep())

    with study_admission():
        mixed = finish(build("mixed"), "native")
        double = finish(build("double"), "native")
        oracle = finish(build("mixed"), "numpy_ref")
    assert_within(mixed, double, 1e-4)
    assert_within(mixed, oracle, 1e-4)


def test_an_adams_snapshot_round_trip_is_exact():
    """The state and the history rows are binary64: class P, no scaling (SEP 36 5)."""
    with study_admission():
        simulation = gnlse("mixed", steps=None, integrator=Multistep())
        with ExitStack() as cleanup:
            first = cast(StepDriver, loaded(simulation, "native", cleanup))
            first.propagate_to(0.02)
            snapshot = first.march_snapshot()
            assert snapshot is not None
            second = cast(StepDriver, loaded(simulation, "native", cleanup))
            second.set_initial(snapshot.state, None, z=snapshot.z)
            assert second.restore_march(snapshot) is None
            again = second.march_snapshot()
    assert again is not None
    assert snapshot.state.dtype == np.complex128
    assert np.array_equal(again.state, snapshot.state)
    assert set(again.rows) == set(snapshot.rows)
    for name, rows in snapshot.rows.items():
        assert np.array_equal(again.rows[name], rows), name


@pytest.mark.parametrize("march", ["adaptive", "replayed", "low_storage"])
def test_a_non_finite_state_stops_every_march(march):
    simulation = {
        "adaptive": gnlse("mixed", steps=None),
        "replayed": gnlse("mixed"),
        "low_storage": gnlse("mixed", integrator=LowStorageRungeKutta()),
    }[march]
    with ExitStack() as cleanup:
        backend = loaded(simulation, "native", cleanup)
        field = simulation.translate().initial.field0.copy()
        field[..., int(np.argmax(np.abs(field[0, 0, 0])))] = np.nan
        backend.set_initial(field, None)
        with pytest.raises((ConvergenceError, BackendError)):
            backend.propagate_to(0.05)


def test_a_modified_noise_run_agrees_with_the_oracle_under_the_study_admission(
    tmp_path,
):
    """The realisation joins the binary64 stage input before its narrowing (4.4).

    The same seed draws the same realisation on each backend; the driver hands it to
    the backend. The quiet run shows the noise reached the field.
    """

    def run(
        p: PrecisionName, backend: str, noise: NoiseModel | None = "modified"
    ) -> np.ndarray:
        simulation = Simulation.from_gnlse(
            0.11, 0.02, [0.0, 0.0, -1.183e-26, 8.1e-41], lambda0=835e-9,
            wavelength_limits=(500e-9, 2e-6), trange=1e-12, width=50e-15,
            peak_power=100.0, pulseshape="sech", save_planes=[0.02],
            backend=backend, precision=p, noise=noise,
            seed=None if noise is None else 3,
            integrator=RungeKutta(),
            schedule=FixedSchedule.uniform(0.0, 0.02, 0.002, save_planes=[0.02]),
        )  # fmt: skip
        path = tmp_path / f"{p}-{backend}-{noise}.h5"
        simulation.run(path, progress="none")
        with h5py.File(path, "r") as file:
            dataset = file["field"]
            assert isinstance(dataset, h5py.Dataset)
            return np.asarray(dataset[()])[-1]

    with study_admission():
        native = run("mixed", "native")
        oracle = run("mixed", "numpy_ref")
        double = run("double", "native")
        quiet = run("mixed", "native", None)
    assert assert_within(native, quiet, 1.0) > 0.0
    bound = two_sides(bound_of(lambda p: gnlse(p, length=0.02, steps=10)))
    assert_within(native, oracle, bound)
    assert_within(native, double, bound)


# Memory: the ledger's representative ratios (SEP 36 section 6.7).


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(gnlse, id="fibre"),
        pytest.param(lambda p: capillary(p, modes=4), id="multimode"),
    ],
)
def test_the_ledger_keeps_the_stored_rows_binary64_and_halves_the_source(build):
    """Mixed keeps every stored row's size and halves the source's workspaces."""
    ledgers = {}
    for precision in ("double", "single", "mixed"):
        with ExitStack() as cleanup:
            ledger = loaded(build(precision), "native", cleanup).memory_ledger()
        ledger.pop("fft_plan_heap_growth", None)
        ledgers[precision] = ledger
    double, mixed = ledgers["double"], ledgers["mixed"]
    for row in ("state", "stages", "propagators", "accumulator_solution"):
        assert mixed[row] == double[row], row
    for row in ("fine_spectrum", "node_field", "node_polarisation"):
        assert 2 * mixed[row] == double[row], row
    totals = {p: sum(ledger.values()) for p, ledger in ledgers.items()}
    assert totals["single"] < totals["mixed"] < totals["double"]


def ledger_total(simulation: Simulation) -> tuple[int, dict[str, int]]:
    """Return a native state's ledger total, its planners' heap growth left out."""
    with ExitStack() as cleanup:
        ledger = loaded(simulation, "native", cleanup).memory_ledger()
    ledger.pop("fft_plan_heap_growth", None)
    return sum(ledger.values()), ledger


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda p: spatial_kerr("radial", p), id="radial"),
        pytest.param(lambda p: spatial_kerr("cartesian", p), id="cartesian"),
        pytest.param(lambda p: spatial_kerr("line", p), id="line"),
    ],
)
def test_the_spatial_ledgers_order_the_modes(build):
    """The spatial routes' node chunks and transforms are the source's (6.7)."""
    totals = {p: ledger_total(build(p))[0] for p in ("double", "single", "mixed")}
    assert totals["single"] < totals["mixed"] < totals["double"]


def test_the_cartesian_split_step_holds_a_binary64_transverse_volume():
    """Mixed's collar transforms the binary64 state in a volume of its own (4.2)."""

    def build(p: str, scheme: str) -> Simulation:
        return spatial_kerr(
            "cartesian", p, absorbed=True, scheme=scheme, window_factor=0.0
        )

    split = ledger_total(build("mixed", "split_step"))[1]["transverse_scratch"]
    in_stage = ledger_total(build("mixed", "nonlinear"))[1]["transverse_scratch"]
    double = ledger_total(build("double", "split_step"))[1]["transverse_scratch"]
    # The binary32 source volume, half binary64's, and beside it one of binary64.
    assert 2 * in_stage == double
    assert split == in_stage + double


def test_the_envelope_plasma_agrees_within_its_class_t_envelope():
    """SEP 36 section 6.5: the cycle moments on a binary32 quadrature, binary64 sums.

    Neither the oracle's binary64 moments of the narrowed field nor native's binary32
    quadrature is a matched sequence of the other (section 4.5), so the comparison is
    the class T envelope SEP 33's oracle test fixed for this case, two of eight steps:
    native against binary64 and against the oracle, the populations' statistics too.
    """
    length = 0.001
    spec = StatsSpec(names=("energy", "electron_density"))
    fields, stats = {}, {}
    for precision, name in (("mixed", "native"), ("mixed", "numpy_ref"),
                            ("double", "native")):  # fmt: skip
        with ExitStack() as cleanup:
            backend = loaded(envelope_plasma(precision), name, cleanup)
            backend.propagate_to(length / 4)
            fields[precision, name] = backend.observe().require_field().copy()
            stats[precision, name] = backend.stats(spec)
    native = fields["mixed", "native"]
    assert_within(native, fields["double", "native"], 1e-5)
    assert_within(native, fields["mixed", "numpy_ref"], 1e-5)
    for key in ("energy", "electron_density"):
        assert_within(
            np.asarray(stats["mixed", "native"][key]),
            np.asarray(stats["double", "native"][key]),
            1e-5,
            what=key,
        )


# The spatial routes (SEP 36 section 5.1, phase 3).


@pytest.mark.parametrize(
    ("build", "transverse"),
    [
        pytest.param(lambda p: spatial_kerr("radial", p), 32, id="radial"),
        pytest.param(lambda p: spatial_kerr("cartesian", p), 8, id="cartesian"),
        pytest.param(lambda p: spatial_kerr("line", p), 4, id="line"),
        pytest.param(
            lambda p: spatial_kerr("cartesian", p, integrator=LowStorageRungeKutta()),
            8,
            id="cartesian-low-storage",
        ),
        pytest.param(
            lambda p: plasma_beam(p, route="radial"), 32, id="radial-filament"
        ),
        pytest.param(
            lambda p: plasma_beam(p, route="cartesian"), 8, id="cartesian-filament"
        ),
    ],
)
def test_the_spatial_routes_agree_with_the_oracle(build, transverse):
    """The radial Hankel product over 32 nodes, the Cartesian and line transforms."""
    against_oracle(build, bound_of(build, transverse=transverse))


@pytest.mark.parametrize(
    ("build", "transverse"),
    [
        pytest.param(
            lambda p: spatial_kerr("radial", p, chi3=None, absorbed=True),
            32,
            id="radial",
        ),
        pytest.param(
            lambda p: spatial_kerr("cartesian", p, chi3=None, absorbed=True),
            8,
            id="cartesian",
        ),
        pytest.param(
            lambda p: spatial_kerr(
                "cartesian",
                p,
                chi3=None,
                absorbed=True,
                integrator=LowStorageRungeKutta(),
            ),
            8,
            id="cartesian-low-storage",
        ),
    ],
)
def test_in_stage_spatial_collars_agree_with_the_oracle(build, transverse):
    """Spatial collars beside a weak temporal one, binary32 source terms (kappa 1)."""
    against_oracle(build, bound_of(build, kappa=1.0, transverse=transverse))


@pytest.mark.parametrize(
    "build",
    [
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
def test_a_source_free_cartesian_run_agrees_with_binary64_within_binary64_bounds(
    build,
):
    """No source, or the split-step collar, whose transverse and temporal transforms
    are binary64 under Mixed (its own binary64 volume, SEP 36 sections 4.2 and 6.7)."""
    problem = build("mixed").translate()
    assert problem.stepper.schedule is not None
    steps = len(problem.stepper.schedule.steps)
    depth = math.ceil(math.log2(problem.grids.n_to)) + 8
    bound = steps * (2 * EPS + 4 * 16 * EPS * depth)
    mixed, double = finish(build("mixed"), "native"), finish(build("double"), "native")
    assert_within(mixed, double, bound)
