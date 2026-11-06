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
@pytest.mark.parametrize(("case", "stepper", "fraction"), WINDOWS[:6])
def test_julia_single_precision_tracks_the_reference(runtime, case, stepper, fraction):
    """Binary32 (SEP 33) agrees with the binary32 reference on the same contract.

    A single-precision run's grid is its own, so the reference is numpy_ref's binary32
    path, not the binary64 run. The steps may differ: binary32's error estimate is
    noisier near rtol's floor.
    """
    del runtime
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
