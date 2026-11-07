# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Julia backend experiment against the NumPy oracle, on short benchmark windows.

Optional: skipped unless ``solprop[julia]`` is installed and ``julia/bootstrap.py`` has
prepared the runtime. ``SOLPROP_JULIA_DEVICE=cuda`` runs the same tests on an NVIDIA GPU
(``pytest -m gpu``). Every kernel body is checked to be one a GPU could take
(``SOLPROP_JULIA_CHECK``), so that a CPU run catches what would fail on a device.
"""

import importlib.util
import os

import numpy as np
import pytest
from benchmarks.builders import build
from benchmarks.registry import CASES

from solprop.backends import select
from solprop.exceptions import BackendError

os.environ.setdefault("SOLPROP_JULIA_CHECK", "1")
# Bounds checked: an @inbounds index slip raises here instead of corrupting memory.
os.environ.setdefault("PYTHON_JULIACALL_CHECK_BOUNDS", "yes")
DEVICE = os.environ.get("SOLPROP_JULIA_DEVICE", "cpu")
pytestmark = [pytest.mark.medium] + ([pytest.mark.gpu] if DEVICE != "cpu" else [])
# An Apple GPU has no binary64: on Metal only the binary32 tests run, with the
# operator's exponents and the Adams flow in double-float (julia/src/dfloat.jl).
binary64 = pytest.mark.skipif(DEVICE == "metal", reason="Apple GPUs have no binary64")
# The wide types a binary32 run can form its exponents in on this device.
WIDE = ["df32"] if DEVICE == "metal" else ["binary64", "df32"]


@pytest.fixture(scope="module")
def runtime():
    """Start the runtime once per worker process, or skip without one."""
    if importlib.util.find_spec("juliacall") is None:
        pytest.skip("solprop[julia] is not installed")
    from solprop.backends.julia._runtime import runtime as start  # noqa: PLC0415

    try:
        return start()
    except BackendError as exc:
        pytest.skip(f"the Julia runtime is not prepared: {exc}")


# (case, stepper, fraction): each a few steps of a focus route (docs/guide/performance/
# julia.md). The fractions keep the NumPy oracle to seconds.
WINDOWS = [
    ("rdw", "rk", 0.002),
    ("rdw", "default", 0.002),
    ("argon_blueshift", "default", 0.002),
    ("h2_raman_6ps", "default", 0.0005),
    ("radial_filament_256", "default", 0.002),
    ("line", "default", 0.005),
    ("kerr_beam_192", "default", 1.0),
]


# Statistics that measure rounding: the quadrature's error estimate and the plasma
# column's work balance.
ROUNDING_ESTIMATES = frozenset(("quadrature_error", "plasma_work_defect"))


def _run(name, configuration):
    """Propagate a window on one backend; return its field, steps and statistics."""
    simulation = build(configuration | {"backend": name})
    problem = simulation.translate()
    engine = select(
        name,
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    engine.set_initial(problem.initial.field0, None, z=problem.initial.z)
    engine.propagate_to(float(configuration["stop"]))
    observation = engine.observe(field=True)
    steps = (observation.n_steps, observation.n_rejected)
    statistics = engine.stats(problem.stats_spec)
    engine.close()
    return observation.field, steps, statistics


# The Cartesian beam's fused silica is evaluated past Malitson's range, by design of the
# benchmark case; the warning says so and is not this test's subject.
@binary64
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.parametrize(("case", "stepper", "fraction"), WINDOWS)
def test_julia_matches_the_reference(runtime, case, stepper, fraction):
    """Same steps, and the same field to rounding, as numpy_ref on the same contract."""
    del runtime
    configuration = CASES[case].configuration(
        stepper, 2, "julia", fraction=fraction, noise="none"
    )
    field, steps, statistics = _run("julia", configuration)
    reference, reference_steps, reference_statistics = _run("numpy_ref", configuration)
    assert steps == reference_steps
    # The engine forms the Poynting set's transforms and the populations itself: each
    # statistic to rounding of its own largest value (a spectrum's tails are rounding).
    assert set(statistics) == set(reference_statistics)
    for name, value in reference_statistics.items():
        if name in ROUNDING_ESTIMATES:
            # A difference of nearly equal sums, at their rounding here (a work defect
            # of 1e-13 to 1e-12): two orders of summation agree on its order only.
            ours, theirs = float(statistics[name][0]), float(value[0])
            assert ours == theirs or 0.5 <= ours / theirs <= 2.0, (name, ours, theirs)
            continue
        largest = float(np.max(np.abs(value), initial=0.0))
        np.testing.assert_allclose(
            statistics[name], value, rtol=1e-9, atol=1e-9 * largest, err_msg=name
        )
    # Matched discrete methods on one step sequence: the fields differ by the rounding
    # of their transforms and sums, which the native/julia A/B measures at <= 1e-13
    # over these windows; 1e-10 leaves three orders for longer accumulation and is
    # still far below the controller's 1e-6 relative tolerance.
    scale = float(np.linalg.norm(reference))
    assert scale > 0
    assert float(np.linalg.norm(field - reference)) <= 1e-10 * scale


@binary64
def _fixed_capillary(backend, *, integrator=None, fixed=True):
    """A short ionising capillary, on 16 fixed steps (a batch runs a fixed schedule) or
    adaptive."""
    from solprop.driver import Simulation  # noqa: PLC0415
    from solprop.stepper import FixedSchedule, RungeKutta  # noqa: PLC0415

    length = 0.004
    schedule = FixedSchedule.uniform(0.0, length, length / 16) if fixed else None
    return Simulation.from_capillary(
        50e-6,
        length,
        "Ar",
        1.0,
        wavelength=800e-9,
        wavelength_limits=(150e-9, 3e-6),
        trange=300e-15,
        fwhm=10e-15,
        energy=60e-6,
        n_save=2,
        backend=backend,
        integrator=RungeKutta() if integrator is None else integrator,
        schedule=schedule,
        absorber_scheme="split_step",
        noise=None,
    )


@binary64
def test_julia_modal_low_storage_matches_the_reference(runtime):
    """RK4IP on a capillary (a modal route) replays the reference's schedule."""
    del runtime
    from contextlib import ExitStack  # noqa: PLC0415

    from solprop.stepper import LowStorageRungeKutta  # noqa: PLC0415

    stop, fields = 0.004, {}
    with ExitStack() as cleanup:
        for name in ("numpy_ref", "julia"):
            simulation = _fixed_capillary(name, integrator=LowStorageRungeKutta())
            engine = _loaded(simulation, name, cleanup)
            engine.propagate_to(stop)
            fields[name] = engine.observe(field=True).field
    scale = float(np.linalg.norm(fields["numpy_ref"]))
    assert float(np.linalg.norm(fields["julia"] - fields["numpy_ref"])) <= 1e-10 * scale


# The band reaches below O2's qualified dispersion range; the warning says so and is
# not this test's subject.
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@binary64
def test_julia_two_raman_components_match_the_reference(runtime):
    """Two Raman components (air's, split into an N2 and an O2 half) sum into the
    engine's one kernel and match the reference's two convolutions."""
    del runtime
    import dataclasses  # noqa: PLC0415
    from contextlib import ExitStack  # noqa: PLC0415

    from solprop.driver import Simulation  # noqa: PLC0415
    from solprop.stepper import FixedSchedule, RungeKutta  # noqa: PLC0415
    from solprop.translate.problem import Flags  # noqa: PLC0415

    length = 0.004
    simulation = Simulation.from_capillary(
        50e-6,
        length,
        {"N2": 0.79, "O2": 0.21},
        1.0,
        wavelength=800e-9,
        wavelength_limits=(200e-9, 3e-6),
        trange=300e-15,
        fwhm=10e-15,
        energy=40e-6,
        n_save=2,
        backend="numpy_ref",
        plasma=False,
        raman=True,
        noise=None,
        integrator=RungeKutta(),
        absorber_scheme="split_step",
        schedule=FixedSchedule.uniform(0.0, length, length / 16),
    )
    problem = simulation.translate()
    (term,) = problem.responses.raman_terms
    terms = tuple(
        dataclasses.replace(term, fraction=term.fraction / 2, label=label)
        for label in ("N2", "O2")
    )
    responses = dataclasses.replace(problem.responses, raman_terms=terms)
    problem = dataclasses.replace(
        problem,
        responses=responses,
        flags=Flags.from_content(
            shape=problem.shape,
            linear=problem.linear,
            transform=problem.transform,
            responses=responses,
            initial=problem.initial,
        ),
    )
    fields = {}
    with ExitStack() as cleanup:
        for name in ("numpy_ref", "julia"):
            engine = select(
                name,
                problem,
                grid=simulation.grid,
                modes=simulation.modes,
                profiles=simulation.profiles,
            )
            cleanup.callback(engine.close)
            engine.set_initial(problem.initial.field0, None, z=problem.initial.z)
            engine.propagate_to(length)
            fields[name] = engine.observe(field=True).field
    scale = float(np.linalg.norm(fields["numpy_ref"]))
    assert float(np.linalg.norm(fields["julia"] - fields["numpy_ref"])) <= 1e-10 * scale


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.filterwarnings("ignore::solprop.exceptions.GasFlowWarning")
@binary64
def test_julia_taper_matches_the_reference(runtime):
    """A 20 % radius taper with a pressure gradient: the synthesis scaled by the taper,
    the projection by its reciprocal, on ten fixed steps (the native test's case)."""
    del runtime
    from dataclasses import replace  # noqa: PLC0415

    from solprop.boundaries import absorber  # noqa: PLC0415
    from solprop.driver import Simulation  # noqa: PLC0415
    from solprop.profiles import Linear  # noqa: PLC0415
    from solprop.stepper import FixedSchedule  # noqa: PLC0415

    length = 0.1

    def field(backend):
        simulation = Simulation.from_capillary(
            Linear(100e-6, 80e-6, length),
            length,
            "Ar",
            (5.0, 0.2),
            wavelength=800e-9,
            fwhm=30e-15,
            energy=10e-6,
            wavelength_limits=(300e-9, 2e-6),
            trange=400e-15,
            noise=None,
            vacuum_end="floor",
            plasma=False,
            n_save=3,
            backend=backend,
            schedule=FixedSchedule(np.full(10, length / 10)),
        )
        simulation.settings = replace(
            simulation.settings,
            absorber=absorber(simulation.grid, boundary_length=10.0),
            boundary_check=None,
        )
        return np.asarray(simulation.run()["field"])[-1]

    expected, ours = field("numpy_ref"), field("julia")
    scale = float(np.linalg.norm(expected))
    assert float(np.linalg.norm(ours - expected)) <= 1e-10 * scale


@binary64
def test_julia_tsit5_matches_the_reference(runtime):
    """An adaptive TSIT5 window takes the reference's steps and field."""
    del runtime
    from contextlib import ExitStack  # noqa: PLC0415

    from solprop.stepper import RungeKutta  # noqa: PLC0415
    from solprop.stepper.tableau import TSIT5  # noqa: PLC0415

    stop, observed = 0.004, {}
    with ExitStack() as cleanup:
        for name in ("numpy_ref", "julia"):
            simulation = _fixed_capillary(
                name, integrator=RungeKutta(TSIT5), fixed=False
            )
            engine = _loaded(simulation, name, cleanup)
            engine.propagate_to(stop)
            observation = engine.observe(field=True)
            observed[name] = (
                observation.field,
                observation.n_steps,
                observation.n_rejected,
            )
    field, steps, rejected = observed["julia"]
    expected, expected_steps, expected_rejected = observed["numpy_ref"]
    assert (steps, rejected) == (expected_steps, expected_rejected)
    scale = float(np.linalg.norm(expected))
    assert float(np.linalg.norm(field - expected)) <= 1e-10 * scale


def _loaded(simulation, name, cleanup):
    """The simulation's backend, its launch loaded."""
    problem = simulation.translate()
    engine = select(
        name,
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    cleanup.callback(engine.close)
    engine.set_initial(problem.initial.field0, None, z=problem.initial.z)
    return engine


@binary64
def test_julia_batch_members_are_their_solo_runs(runtime):
    """Lockstep members (`select_batch`) match their solo runs and the reference."""
    del runtime
    from contextlib import ExitStack  # noqa: PLC0415

    from solprop.backends.registry import select_batch  # noqa: PLC0415

    stop = 0.004
    with ExitStack() as cleanup:
        reference = _loaded(_fixed_capillary("numpy_ref"), "numpy_ref", cleanup)
        reference.propagate_to(stop)
        expected = reference.observe(field=True).field
        members = [
            _loaded(_fixed_capillary("julia"), "julia", cleanup) for _ in range(2)
        ]
        runner = select_batch(members)
        runner.propagate_to(stop)
        fields = [member.field for member in runner.observe_members()]
    scale = float(np.linalg.norm(expected))
    for field in fields:
        assert float(np.linalg.norm(field - expected)) <= 1e-10 * scale


def test_julia_reports_its_engine(runtime):
    """Metadata names the runtime and the sources; the ledger counts owned arrays."""
    del runtime
    configuration = CASES["rdw"].configuration("rk", 1, "julia", fraction=0.001)
    simulation = build(configuration | {"noise": "none"})
    problem = simulation.translate()
    engine = select("julia", problem, grid=simulation.grid, modes=simulation.modes)
    metadata = engine.metadata()
    assert metadata["experimental"] is True
    assert len(metadata["julia_source_digest"]) == 64
    assert metadata["device"].startswith(DEVICE)
    ledger = engine.memory_ledger()
    assert ledger["work.y"] == 16 * problem.shape[1] * problem.shape[3]
    engine.close()


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.parametrize("wide", WIDE)
@pytest.mark.parametrize(("case", "stepper", "fraction"), WINDOWS[:6])
def test_julia_single_precision_tracks_the_reference(
    runtime, monkeypatch, case, stepper, fraction, wide
):
    """Binary32 (SEP 33) agrees with the binary32 reference on the same contract.

    A single-precision run's grid is its own, so the reference is numpy_ref's binary32
    path, not the binary64 run. The steps may differ: binary32's error estimate is
    noisier near rtol's floor. The operator's exponents and the Adams flow are formed
    in binary64 or in double-float (``SOLPROP_JULIA_WIDE``), rounded once either way.
    """
    del runtime
    monkeypatch.setenv("SOLPROP_JULIA_WIDE", wide)
    configuration = CASES[case].configuration(
        stepper, 2, "julia", fraction=fraction, noise="none"
    ) | {"precision": "single"}
    field, _, _ = _run("julia", configuration)
    reference, _, _ = _run("numpy_ref", configuration)
    # SEP 33's whole-run field envelope is 1e-3; over these few steps two binary32
    # implementations differ by their rounding, ~1e-6 (PR #543's records), so 1e-4
    # holds two orders in hand and still catches a mis-scaled response (O(1)).
    scale = float(np.linalg.norm(reference))
    assert float(np.linalg.norm(field - reference)) <= 1e-4 * scale


def test_julia_double_float_phi_is_binary64s(runtime):
    """The double-float phi functions agree with the binary64 ones to well inside
    binary32's rounding, across the series and recurrence branches and a long phase.

    The pair holds about 2^-48; the recurrence past |x| = 3 amplifies an error by up to
    about k! / |x|^k (4e-13 relative at order 9 just past the join), and the phase's
    reduction keeps about 2^-48 of the phase. 2^-36 bounds both and is still 2^12
    inside the binary32 rounding each value then takes.
    """
    generator = np.random.default_rng(5)
    radii = np.concatenate([10.0 ** generator.uniform(-6, 0.5, 40), [2.9, 3.1, 30.0]])
    angles = generator.uniform(-np.pi, np.pi, radii.size)
    points = [*(radii * np.exp(1j * angles)), -40 + 2e3j, -1e-3 + 1e4j, 0.0]
    for order in (1, 3, 6, 9):
        for x in points:
            exact = np.asarray(runtime.phi_snapshot(complex(x), order))
            pair = np.asarray(runtime.phi_snapshot_df(complex(x), order))
            bound = 2.0**-36 * max(1.0, abs(x.imag)) * np.abs(exact) + 1e-300
            np.testing.assert_array_less(
                np.abs(pair - exact), bound, err_msg=f"{x} {order}"
            )


def test_julia_double_float_propagators_round_once(runtime):
    """exp(+-z) by the double-float pair is within a few binary32 ulps of binary64's,
    for phases up to 1e6 rad (the mlx backend's measured bound is 5 u)."""
    generator = np.random.default_rng(7)
    re = generator.uniform(-20, 5, 2000)
    im = np.concatenate(
        [generator.uniform(-10, 10, 1000), generator.uniform(-1e6, 1e6, 1000)]
    )
    for z in re + 1j * im:
        forward, back = runtime.exp_pair_snapshot_df(complex(z))
        for value, exact in ((forward, np.exp(z)), (back, np.exp(-z))):
            assert abs(value - exact) <= 8 * 2.0**-24 * abs(exact), (z, value, exact)


@binary64
def test_julia_poynting_statistics_match_the_host(runtime):
    """The engine's analytic signals give the shared host function's statistics."""
    del runtime
    from solprop.stats import _poynting_statistics  # noqa: PLC0415

    configuration = CASES["rdw"].configuration(
        "default", 2, "julia", fraction=0.002, noise="none"
    )
    simulation = build(configuration)
    problem = simulation.translate()
    engine = select(
        "julia",
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    engine.set_initial(problem.initial.field0, None, z=problem.initial.z)
    engine.propagate_to(float(configuration["stop"]))
    state = engine.observe(field=True).field[0]
    axis = engine._on_axis_synthesis(simulation.modes)
    spec = problem.stats_spec
    ours = engine._poynting(spec, simulation.grid, state, axis)
    host = _poynting_statistics(spec, simulation.grid, state, engine._weight, axis)
    assert set(ours) == set(host)
    for name, value in host.items():
        # The same transforms in another library and order: rounding of the analytic
        # signals' sums, far below the 1e-10 the fields are held to.
        np.testing.assert_allclose(ours[name], value, rtol=1e-11, atol=0, err_msg=name)
    engine.close()


@binary64
def test_julia_helpers_are_the_python_forms_bit_for_bit(runtime):
    """The record's Chebyshev rows and Adams basis equal PLAIN's to the last bit."""
    del runtime
    from solprop._helpers import PLAIN  # noqa: PLC0415
    from solprop.backends.julia._helpers import JULIA  # noqa: PLC0415

    generator = np.random.default_rng(7)
    edges = np.cumsum(generator.uniform(0.1, 1.0, 6))
    lower, upper = edges[0] - 1e-12, edges[-1] + 1e-12
    z = np.concatenate([edges, generator.uniform(edges[0], edges[-1], 9), [np.nan]])
    for terms in (1, 2, 3, 17):
        for dtype in (np.float64, np.complex128):
            c = generator.standard_normal((5, terms, 4)).astype(dtype)
            if dtype == np.complex128:
                c += 1j * generator.standard_normal(c.shape)
            ours, theirs = np.empty((z.size, 4), dtype), np.empty((z.size, 4), dtype)
            assert JULIA.points(z, edges, lower, upper, c, ours) == PLAIN.points(
                z, edges, lower, upper, c, theirs
            )
            np.testing.assert_array_equal(ours, theirs)
    outside = np.array([edges[1], edges[-1] + 1.0])
    out = np.empty((2, 4))
    c = generator.standard_normal((5, 3, 4))
    assert JULIA.points(outside, edges, lower, upper, c, out) == 1
    nodes = np.array([0.0, -0.93, -1.81, -2.77, -3.6, -4.55, -5.4])
    for count in range(1, nodes.size + 1):
        for upper, integrate in ((1.0, False), (1.0, True), (0.37, True)):
            ours = JULIA.basis(np.ascontiguousarray(nodes[:count]), upper, integrate)
            theirs = PLAIN.basis(np.ascontiguousarray(nodes[:count]), upper, integrate)
            np.testing.assert_array_equal(ours[0], theirs[0])
            if integrate:
                np.testing.assert_array_equal(ours[1], theirs[1])


@binary64
@pytest.mark.parametrize("compensated", [False, True])
def test_julia_trial_tables_are_the_shared_ones_bit_for_bit(runtime, compensated):
    """The engine's own trial tables equal backends.multistep.trial_tables'."""
    from juliacall import Main  # noqa: PLC0415

    from solprop._helpers import PLAIN  # noqa: PLC0415
    from solprop.backends.multistep import trial_tables  # noqa: PLC0415

    nodes = [0.0, -0.93, -1.81, -2.77, -3.6, -4.55]
    host = Main.seval(
        """(configure, nodes, order, orders, compensated) -> begin
            sizes = [0, 0, 0]
            h = (predictor=zeros(8, 8), corrector=zeros(8, 8), defects=zeros(8, 16),
                 slots=zeros(Int, 8), spans=zeros(8), cpredictor=zeros(8),
                 ccorrector=zeros(8), cdefects=zeros(8, 8), rate=zeros(1))
            adams = (host=h, device=h, sizes=sizes, compensated=compensated)
            engine = (work=(adams=adams,),)
            configure(engine, nodes, order, orders, zeros(Int, 8), [0.5])
            h
        end"""
    )
    for order, orders in ((3, [3, 2, 4]), (4, [4, 3, 5]), (1, [1, 2])):
        h = host(
            runtime.configure_adams_nodes_b,
            np.asarray(nodes),
            order,
            np.asarray(orders),
            compensated,
        )
        shared = trial_tables(
            nodes=nodes,
            order=order,
            orders=orders,
            compensated=compensated,
            helpers=PLAIN,
        )
        width, count = 1 + max(orders), len(orders)
        np.testing.assert_array_equal(
            np.asarray(h.predictor)[:order, :order], shared[0]
        )
        np.testing.assert_array_equal(
            np.asarray(h.corrector)[:width, :width], shared[1]
        )
        np.testing.assert_array_equal(
            np.asarray(h.defects)[:count, : 2 * width], shared[2]
        )
        if compensated:
            np.testing.assert_array_equal(np.asarray(h.spans)[: width - 1], shared[3])
            np.testing.assert_array_equal(np.asarray(h.cpredictor)[:order], shared[4])
            np.testing.assert_array_equal(
                np.asarray(h.ccorrector)[: order + 1], shared[5]
            )
            np.testing.assert_array_equal(
                np.asarray(h.cdefects)[:count, :width], shared[6]
            )


@binary64
def test_julia_stage_scalars_are_the_bundles_bit_for_bit(runtime):
    """The engine's stage scalars, from its own tables, equal scalars_at's.

    rdw's gas fill varies along the capillary: the operator's and prefactor's
    expansions, the Kerr and plasma densities (the focus cases' only varying one).
    """
    del runtime
    case = "rdw"
    from solprop._helpers import PLAIN  # noqa: PLC0415
    from solprop.translate.stage import scalars_at  # noqa: PLC0415

    configuration = CASES[case].configuration(
        "default", 1, "julia", fraction=0.002, noise="none"
    )
    simulation = build(configuration)
    problem = simulation.translate()
    engine = select("julia", problem, grid=simulation.grid, modes=simulation.modes)
    assert engine._varying
    span = float(configuration["stop"])
    z, dz = 0.37 * span, 0.21 * span
    engine._prepare_stages(z, dz)
    path = engine._engine.work.path
    count = len(problem.stepper.tableau.nodes)
    positions = [z] * (count - 1) + [z + dz]
    bundles = scalars_at(problem, z, positions, helpers=PLAIN)
    kerr = problem.responses.kerr_coefficients
    expected = {
        "delta": np.stack([b.phi_delta for b in bundles], axis=1),
        "weights": np.stack([b.prefactor for b in bundles], axis=1),
        "material": np.array(
            [
                [
                    0.0 if kerr is None else kerr.coefficient * b.kerr_density,
                    b.plasma_densities[0] if b.plasma_densities else 0.0,
                    b.preionisation,
                    b.taper_scale,
                ]
                for b in bundles
            ]
        ).T,
    }
    for name, value in expected.items():
        held = np.asarray(getattr(path, name))[:, :count]
        np.testing.assert_array_equal(held, value, err_msg=name)
    engine.close()


def test_julia_vector_maths_is_within_an_ulp_or_two(runtime):
    """vexp and vlog (julia/src/vmath.jl) against the C library's functions.

    Each is within one ulp of the correctly rounded value (measured against
    BigFloat when written); the C library's within one more, so two bound the gap.
    """
    from juliacall import Main  # noqa: PLC0415

    apply = Main.seval("(f, x) -> f.(x)")
    generator = np.random.default_rng(11)
    x = np.concatenate(
        [-745 + 1454 * generator.random(100_000), generator.standard_normal(10_000)]
    )
    ours = np.asarray(apply(runtime.vexp, x))
    np.testing.assert_array_less(
        np.abs(ours - np.exp(x)), 2 * np.spacing(np.exp(x)) + 1e-320
    )
    x = np.exp(-700 + 1400 * generator.random(100_000))
    ours = np.asarray(apply(runtime.vlog, x))
    np.testing.assert_array_less(
        np.abs(ours - np.log(x)), 2 * np.abs(np.spacing(np.log(x)))
    )
    specials = np.asarray(
        apply(runtime.vexp, np.array([np.nan, np.inf, -np.inf, 800.0]))
    )
    assert np.isnan(specials[0])
    assert specials[1:].tolist() == [np.inf, 0.0, np.inf]
    assert np.isnan(float(apply(runtime.vlog, np.array([np.nan]))[0]))


def _march(configuration, march):
    """Run a window with the named march; return the field, history and ledger."""
    os.environ["SOLPROP_JULIA_MARCH"] = march
    try:
        simulation = build(configuration)
        problem = simulation.translate()
        engine = select(
            "julia",
            problem,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        )
    finally:
        del os.environ["SOLPROP_JULIA_MARCH"]
    assert engine._adams.native == (march == "julia")
    engine.set_initial(problem.initial.field0, None, z=problem.initial.z)
    engine.propagate_to(float(configuration["stop"]))
    field = engine.observe(field=True).field
    ledger = engine._adams.ledger
    record = (
        [tuple(attempt[:4]) for attempt in engine._history],
        (ledger.startup, ledger.predictor, ledger.rejected, ledger.accepted_steps),
        (ledger.rejected_steps, ledger.restarts, ledger.failure_restarts),
        list(ledger.orders),
        engine._split_removed,
        None if engine._split_found is None else engine._split_found.tolist(),
    )
    engine.close()
    return field, record


@binary64
@pytest.mark.parametrize(
    ("case", "fraction"),
    [
        ("rdw", 0.002),
        ("argon_blueshift", 0.002),
        ("ctc_reduced", 0.002),
        ("radial_filament_256", 0.002),
    ],
)
def test_julia_march_is_the_python_march(runtime, monkeypatch, case, fraction):
    """Julia's Adams march takes the Python march's decisions on the same engine.

    The same attempts, ledger and accepted orders, the same split collar sums (radial),
    and the same field bit for bit;
    estimated FFT plans, since a measured plan may differ from one process to the next.
    """
    del runtime
    monkeypatch.setenv("SOLPROP_FFTW_PLAN", "estimate")
    configuration = CASES[case].configuration(
        "default", 2, "julia", fraction=fraction, noise="none"
    )
    field, record = _march(configuration, "julia")
    reference, reference_record = _march(configuration, "host")
    assert record == reference_record
    np.testing.assert_array_equal(field, reference)


@binary64
@pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")
@pytest.mark.parametrize(
    ("case", "fraction"), [("radial_filament_256", 0.002), ("line", 0.005)]
)
def test_julia_first_window_compiles_nothing(runtime, case, fraction):
    """After the benchmark worker's warm-up, a timed run compiles (almost) nothing.

    The warm-up propagates without a boundary monitor, so the monitor's bindings and
    the split collar's march first run inside the window: each bridge type compiled
    there cost up to 0.25 s of a radial window until the construction warmed them.
    """
    del runtime
    from benchmarks.worker import timed_run  # noqa: PLC0415
    from juliacall import Main  # noqa: PLC0415

    from solprop.driver.run import field_retention  # noqa: PLC0415

    configuration = CASES[case].configuration(
        "default", 2, "julia", fraction=fraction, noise="none"
    ) | {"warmup": 0.02}
    simulation = build(configuration)
    problem = simulation.translate()
    engine = select(
        "julia",
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    retention = field_retention(
        problem, simulation.settings.fields, simulation.settings.field_planes
    )
    engine.set_initial(problem.initial.field0, None, z=problem.initial.z)
    engine.propagate_to(configuration["warmup"] * configuration["stop"])
    engine.observe(field=True)
    engine.stats(problem.stats_spec)
    Main.seval("Base.cumulative_compile_timing(true)")
    compiled = Main.seval("() -> Base.cumulative_compile_time_ns()[1]")
    compiled()
    before = compiled()
    timed_run(simulation, problem, engine, retention, simulation._boundary_monitor())
    # Some tens of milliseconds of inference the compile trace does not list remain
    # (about 21 on radial); a bridge type left to the window cost 50 to 250.
    assert (compiled() - before) / 1e6 < 50.0
    engine.close()
