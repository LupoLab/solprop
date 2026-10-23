# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The multimode pipeline of the reference backend.

``docs/dev/architecture.md`` section 8 for any node count: one batched inverse
transform, the synthesis GEMM onto a chunk of quadrature nodes, the fused response
pass, the projection GEMM accumulated over chunks, one batched forward transform.
Mode-averaged propagation is the one-node case of exactly that code, which is
what the first test asserts, and the embedded coarse rule gives the quadrature alarm
for nothing.
"""

import logging
import math
import warnings
from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest

from bounds import EPS, assert_within, higham_gamma
from convergence_records import Column, Study, record
from solprop import fourier
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numba_cpu import backend as numba_backend
from solprop.backends.numba_cpu import fft as numba_fft
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.constants import EPS0, C
from solprop.exceptions import (
    BackendError,
    MaterialRangeWarning,
    TranslationError,
)
from solprop.grid import FieldGrid
from solprop.modes import MarcatiliMode, ModeCollection, transverse_quadrature
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec, analytic_field
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.linop import delta_phi
from solprop.translate.problem import assemble

LENGTH = (
    0.02  # m: long enough for a measurable nonlinear phase, short enough to be fast
)
ENERGY = 2e-6  # J
CHI3 = 2.2e-52  # m^2/V^2, argon-like

# The quadrature converges spectrally on the Kerr quartic of a Bessel profile, so a rule
# of this order is exact to the reference's own rounding
# (docs/guide/waveguides/modes.md). The
# embedded estimate of a three-mode set falls 2.365e-1, 3.316e-2, 1.451e-4, 1.776e-8 and
# 4.264e-15 at n_r = 8, 16, 24, 32 and 48, so a set needs the higher order that a single
# HE_11 does not. Those numbers are now the ladder below rather than a hand measurement:
# they live in tests/data/convergence/quadrature_alarm.json and are the second figure of
# docs/dev/accuracy.md section 2.
CONVERGED_N_R = 24
CONVERGED_N_R_SET = 32


@pytest.fixture(scope="module")
def grid():
    """A field grid wide enough to resolve the third harmonic of an 800 nm pulse."""
    return FieldGrid.from_limits(0.2, 800e-9, (300e-9, 3e-6), 100e-15)


@pytest.fixture(scope="module")
def capillary():
    """A lossless helium-filled capillary, so a Kerr run conserves energy exactly."""
    return MarcatiliMode(125e-6, "He", 1.0, cladding=1.45, loss=False)


def modes_of(capillary, count):
    """Return the first `count` HE_1m modes of one capillary."""
    return [
        MarcatiliMode(
            capillary.radius,
            capillary.gas,
            capillary.pressure,
            m=m,
            cladding=capillary.cladding,
            loss=False,
        )
        for m in range(1, count + 1)
    ]


def build(grid, modes, field0, **overrides):
    """Assemble a problem on this grid, silencing helium's extrapolation warning."""
    request = {
        "controller": PIController(rtol=1e-11, atol=1e-16),
        "save_planes": [LENGTH],
        "dz_max": LENGTH / 8,
        "dz_init": 1e-4,
        "responses": [KerrResponse(chi3=CHI3, thg=True)],
    } | overrides
    responses = request.pop("responses")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return assemble(grid, modes, responses, field0, **request)


def propagate(backend_class, problem, grid, distance=LENGTH):
    """Run a problem to a position and return the plane and the backend."""
    backend = backend_class(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    backend.propagate_to(distance)
    return backend.observe(), backend


def peak_gap(got, want):
    """Return the largest deviation as a fraction of the reference's largest value."""
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


def quadrature_alarm(backend):
    """Read the last scored alarm from the implementation's owner."""
    return backend._alarm


def pulse(grid, count=1, energy=ENERGY):
    """Return a Gaussian pulse repeated into `count` modes, in the canonical layout."""
    built = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=energy).build(grid)
    return np.repeat(built, count, axis=0)


# --------------------------------------------------- one node is the mode-averaged case


@pytest.mark.parametrize("forced", [None, "small", "large"])
@pytest.mark.parametrize("modal", [False, True])
def test_modal_kernel_regime_keeps_overrides_and_the_trial(
    grid, capillary, monkeypatch, modal, forced, record_property
):
    """Changing node dispatch preserves the trial and allocates no extra buffers.

    The two regimes are one trial step by two dispatches, which may order the
    reductions differently: seven stages, each a transform's 32 eps a level and the
    projection's and synthesis's ``gamma(2 n_nodes + 2)`` (the vector-route bound
    below).
    """
    monkeypatch.setattr(numba_fft, "MODAL_REGIME_ELEMENTS", 1)
    mode_input = [capillary] if modal else capillary
    options = (
        {
            "quadrature": transverse_quadrature(
                [capillary], n_r=8, radial_rule="gauss_kronrod"
            )
        }
        if modal
        else {}
    )
    problem = build(grid, mode_input, pulse(grid), **options)
    fields = []
    ledgers = []
    for regime in ("small", forced):
        backend = NumbaCpuBackend(
            problem,
            grid=grid,
            threads=1 if regime == "small" else 2,
            fftw_threads=1,
            blas_threads=1,
            plan_flag="estimate",
            regime=regime,
        )
        try:
            backend.set_initial(problem.initial.field0, None)
            backend._trial_step(0.0, problem.stepper.dz_init)
            fields.append(backend._solution.copy())
            ledgers.append(backend.memory_ledger())
            metadata = backend.metadata()
            assert metadata["kernel_regime_elements"] == (
                1 if modal else numba_fft.REGIME_ELEMENTS
            )
            expected = regime or ("large" if modal else "small")
            assert metadata["regime"] == expected
            assert metadata["parallel_nodes"] == (expected == "large")
        finally:
            backend.close()
    n_nodes = problem.transform.n_nodes
    bound = 7 * (
        32 * EPS * math.ceil(math.log2(grid.n_to)) + higham_gamma(2 * n_nodes + 2)
    )
    assert_within(fields[1], fields[0], bound, what="trial step")
    # Allocator growth during planning is a process measurement, not an owned buffer.
    # Retain it separately: even two serial constructions need not grow the same heap.
    for index, ledger in enumerate(ledgers):
        growth = ledger.pop("fft_plan_heap_growth")
        record_property(f"plan_heap_growth_{index}", growth)
        assert isinstance(growth, int) and growth >= 0
    assert ledgers[0] == ledgers[1]


def projection_comparison(output, expected, bound):
    """Assert the entrywise envelope and retain the most strained comparison."""
    gap = np.abs(output - expected)
    assert np.all(gap <= bound)
    assert np.all(gap[bound == 0] == 0)
    ratios = np.divide(gap, bound, out=np.zeros_like(gap), where=bound > 0)
    worst = np.unravel_index(int(np.argmax(ratios)), ratios.shape)
    return {"error": float(gap[worst]), "bound": float(bound[worst])}


@pytest.mark.parametrize("layout", ["scalar", "state", "rows"])
@pytest.mark.parametrize("zero", [False, True])
def test_stacked_projection_preserves_rules_chunks_and_owned_storage(
    grid, capillary, layout, zero, record_property, monkeypatch
):
    """Compare both rules against separate products with a prior error envelope."""
    monkeypatch.setattr(numba_backend, "_STACKED_PROJECTION_MIN_NODES", 1)
    modes = modes_of(capillary, 2)
    if layout == "rows":
        modes = ModeCollection.with_polarisations(modes)
    start = pulse(grid, len(modes))
    if layout == "state":
        start = np.repeat(start, 2, axis=1) / math.sqrt(2)
    rule = transverse_quadrature(modes, n_r=4, radial_rule="gauss_kronrod")
    problem = build(grid, modes, start, quadrature=rule)
    chunk_nodes = 4 if rule.n_nodes % 4 else 5
    problem = replace(
        problem, transform=replace(problem.transform, node_chunk=chunk_nodes)
    )
    rng = np.random.default_rng(33)
    comparisons = []
    with closing(
        NumbaCpuBackend(problem, grid=grid, threads=1, plan_flag="estimate")
    ) as backend:
        assert backend.metadata()["modal_projection"] == "stacked_packed"
        m, r = problem.transform.projection.shape
        p = problem.shape[2]
        ledger = backend.memory_ledger()
        assert ledger["packed_projection"] == 4 * m * r * 8
        assert ledger["stacked_projection"] == 2 * m * p * grid.n_to * 8
        assert ledger["embedded_projection"] == 2 * m * r * 8
        assert backend._plans.coarse_projected is not None
        # A non-unit taper tests the stage's matrix scaling as well as packing.
        scale = 1.25
        for packed, scaled in zip(
            backend._projection_chunks, backend._scaled_projection_chunks, strict=True
        ):
            np.multiply(packed, scale, out=scaled)
        backend._plans.projected.fill(np.nan)
        backend._plans.coarse_projected.fill(np.nan)
        transform = problem.transform
        for first in range(0, transform.n_nodes, chunk_nodes):
            last = min(first + chunk_nodes, transform.n_nodes)
            rows = slice(first * transform.n_pol, last * transform.n_pol)
            count = (last - first) * backend._n_pol_node
            source = rng.normal(size=(rows.stop - rows.start, p * grid.n_to))
            if zero:
                source.fill(0)
            backend._polarisation[:count] = source.reshape(count, grid.n_to)
            fine = scale * transform.projection[:, rows]
            coarse = scale * transform.projection_coarse[:, rows]
            previous = [
                np.zeros_like(source[:m])
                if first == 0
                else output.reshape(m, -1).copy()
                for output in (
                    backend._plans.projected,
                    backend._plans.coarse_projected,
                )
            ]
            # Two k-term real products and their final accumulation.
            # Derive the entrywise envelope before either output is compared.
            q = 2 * source.shape[0] + 8
            gamma = q * np.finfo(float).eps / (1 - q * np.finfo(float).eps)
            assert gamma > 0
            bounds = [
                2 * gamma * (np.abs(matrix) @ np.abs(source) + np.abs(prior))
                for matrix, prior in zip((fine, coarse), previous, strict=True)
            ]
            expected = [
                matrix @ source + prior
                for matrix, prior in zip((fine, coarse), previous, strict=True)
            ]
            backend._project(rows, count, first == 0)
            for output, want, bound in zip(
                (backend._plans.projected, backend._plans.coarse_projected),
                expected,
                bounds,
                strict=True,
            ):
                comparisons.append(
                    projection_comparison(output.reshape(m, -1), want, bound)
                )
        last_nodes = transform.n_nodes % chunk_nodes
        assert 0 < last_nodes < chunk_nodes
        assert backend._projection_chunks[-1].shape[1] == last_nodes * transform.n_pol
    record_property("projection_comparisons", comparisons)


def test_stacked_projection_is_absent_without_an_embedded_rule(grid, capillary):
    """A mode-averaged problem retains the separate path and adds no storage."""
    problem = build(grid, capillary, pulse(grid))
    with closing(
        NumbaCpuBackend(problem, grid=grid, threads=1, plan_flag="estimate")
    ) as backend:
        assert backend.metadata()["modal_projection"] == "separate"
        ledger = backend.memory_ledger()
        assert ledger["packed_projection"] == ledger["stacked_projection"] == 0
        assert ledger["embedded_projection"] == 0


@pytest.mark.parametrize(
    ("n_modes", "n_nodes", "chunk", "expected"),
    [
        (1, 384, 85, False),
        (2, 384, 170, True),
        (3, 1024, 42, True),
        (4, 384, 85, True),
        (5, 384, 85, False),
        (6, 1024, 42, False),
        (2, 255, 85, False),
        (2, 256, 85, True),
        (2, 384, 384, False),
        (4, 33, 33, False),
    ],
)
def test_stacked_projection_dispatch_keeps_unqualified_shapes_separate(
    n_modes, n_nodes, chunk, expected
):
    """Exercise measured shapes and both sides of the predeclared brackets."""
    assert numba_backend._stacked_projection_shape(n_modes, n_nodes, chunk) is expected


def test_stacked_projection_trial_keeps_the_existing_bound(
    grid, capillary, monkeypatch, record_property
):
    """Exercise packed stage scaling and the full initial trial at a tiny shape."""
    monkeypatch.setattr(numba_backend, "_STACKED_PROJECTION_MIN_NODES", 1)
    modes = modes_of(capillary, 2)
    rule = transverse_quadrature(modes, n_r=4, radial_rule="gauss_kronrod")
    problem = build(grid, modes, pulse(grid, 2), quadrature=rule)
    problem = replace(problem, transform=replace(problem.transform, node_chunk=4))
    eps = np.finfo(float).eps
    reduction = rule.n_nodes * problem.transform.n_pol * problem.shape[2]
    # A seven-stage initial-trial budget.
    gamma = reduction * eps / (1 - reduction * eps)
    bound = 7 * (
        16 * eps * math.ceil(math.log2(grid.n_to))
        + 2 * gamma
        + 64 * eps / (1 - 64 * eps)
    )
    assert bound > 0
    fields = []
    for backend_class in (NumpyRefBackend, NumbaCpuBackend):
        with closing(backend_class(problem, grid=grid)) as backend:
            backend.set_initial(problem.initial.field0, None)
            fields.append(backend._trial_step(0.0, problem.stepper.dz_init)[1].copy())
    deviation = peak_gap(fields[1], fields[0])
    assert deviation <= bound
    record_property(
        "initial_trial_comparison", {"deviation": deviation, "bound": bound}
    )


def test_a_one_mode_multimode_run_reproduces_the_mode_averaged_one(
    backend_class, grid, capillary
):
    """The mode-averaged path is the n_nodes = 1 case of this pipeline.

    The mode-averaged run converts with ``1/sqrt(eps0 c A_eff/2)`` and projects with 1;
    the multimode run synthesises the Bessel profile at the nodes and integrates the
    quartic on the rule. They are the same number by the definition of ``A_eff``, so
    what is left is the quadrature's own error.
    """
    start = pulse(grid)
    averaged, _ = propagate(backend_class, build(grid, capillary, start), grid)
    rule = transverse_quadrature(
        [capillary], n_r=CONVERGED_N_R, radial_rule="gauss_kronrod"
    )
    multimode, _ = propagate(
        backend_class, build(grid, [capillary], start, quadrature=rule), grid
    )
    # Measured 1.3e-16 at n_r = 24, against 1.1e-11 at n_r = 8: by this order the rule
    # is exact on the quartic and what is left is the round-off of two different
    # orderings of the same sum -- the machine epsilon times a 257-bin reduction -- and
    # not the quadrature at all.
    assert peak_gap(multimode.field, averaged.field) < 1e-14


@pytest.mark.parametrize("n_r", [8, 16, CONVERGED_N_R])
def test_the_quadrature_converges_towards_the_mode_averaged_answer(
    backend_class, grid, capillary, n_r
):
    """Coarse rules are wrong by the rule's error and by nothing systematic."""
    start = pulse(grid)
    averaged, _ = propagate(backend_class, build(grid, capillary, start), grid)
    rule = transverse_quadrature([capillary], n_r=n_r, radial_rule="gauss_kronrod")
    multimode, _ = propagate(
        backend_class, build(grid, [capillary], start, quadrature=rule), grid
    )
    # Measured 1.1e-11, 2.9e-18 and 1.3e-16 at n_r = 8, 16 and 24: the rule converges
    # spectrally and is already at round-off by 16. The bound is the coarsest of them
    # with a decade in hand.
    assert peak_gap(multimode.field, averaged.field) < 1e-10


# ------------------------------------------------------------------- the linear flow


def test_the_pipeline_with_a_zero_coefficient_changes_nothing(
    backend_class, grid, capillary
):
    """The zero-coefficient check through the multimode pipeline, exactly.

    A Kerr response of zero susceptibility runs every pass -- the transforms, both
    GEMMs, the cubic and the windows -- and must return exactly the state the short
    circuit returns when nothing at all is declared: its source is an exact zero, and
    adding an exact zero is an exact no-op in any order. Anything else is a term the
    pipeline invents.
    """
    start = pulse(grid)
    rule = transverse_quadrature([capillary], n_r=8, radial_rule="gauss_kronrod")
    schedule = {"dz_max": LENGTH / 4, "dz_init": LENGTH / 4}
    running, _ = propagate(
        backend_class,
        build(
            grid,
            [capillary],
            start,
            quadrature=rule,
            responses=[KerrResponse(chi3=0.0, thg=True)],
            **schedule,
        ),
        grid,
    )
    empty, _ = propagate(
        backend_class,
        build(grid, [capillary], start, quadrature=rule, responses=[], **schedule),
        grid,
    )
    assert np.array_equal(running.field, empty.field)


def test_modes_with_a_zero_coefficient_propagate_independently_and_exactly(
    backend_class, grid, capillary
):
    """Two modes, the pipeline running, and nothing coupling them: the linear flow."""
    modes = modes_of(capillary, 2)
    rule = transverse_quadrature(modes, n_r=16, radial_rule="gauss_kronrod")
    problem = build(
        grid,
        modes,
        pulse(grid, 2),
        quadrature=rule,
        responses=[KerrResponse(chi3=0.0, thg=True)],
    )
    plane, _ = propagate(backend_class, problem, grid)
    exact = np.exp(delta_phi(problem.linear, 0.0, LENGTH)) * problem.initial.field0
    # The operator reaches a large phase over this length at the top of the band, so
    # the bound is that exponent times the machine epsilon times the step count;
    # measured 8.0e-16.
    assert peak_gap(plane.field, exact) < 1e-13


# ----------------------------------------------------------------------- conservation


def total_energy(grid, field):
    """Return the energy summed over the modes of one batch member, in joules."""
    return float(fourier.energy_omega(field, grid.dt, kind=grid.kind).sum())


def test_the_transform_pair_loses_no_energy(backend_class, grid, capillary):
    """Two modes, every pass running, and a coefficient of zero: nothing is lost.

    The synthesis and projection matrices are a dual pair
    (``modes.quadrature.BASIS_NORMALISATION``), so the round trip onto the quadrature
    nodes and back is the identity and the transform pair neither creates nor destroys
    energy. With the fibre lossless the only thing left that could move the total is
    round-off.
    """
    modes = modes_of(capillary, 2)
    rule = transverse_quadrature(
        modes, n_r=CONVERGED_N_R_SET, radial_rule="gauss_kronrod"
    )
    problem = build(
        grid,
        modes,
        pulse(grid, 2),
        quadrature=rule,
        responses=[KerrResponse(chi3=0.0, thg=True)],
    )
    plane, _ = propagate(backend_class, problem, grid)
    before = total_energy(grid, problem.initial.field0[0])
    after = total_energy(grid, plane.field[0])
    # Measured 4.4e-16, which is the machine epsilon times the 257-bin reduction.
    assert abs(after / before - 1.0) < 1e-14


def test_a_kerr_run_keeps_the_energy_the_medium_does_not_hold(
    backend_class, grid, capillary
):
    """An instantaneous Kerr medium stores energy, so the field's alone is not fixed.

    A real chi3 is reactive but not inert: the medium holds a quarter of
    ``eps0 chi3 E^4``, so the field energy moves by the size of the nonlinear phase and
    not by round-off -- with the same fibre and no nonlinearity it is fixed to 4.4e-16,
    which the test above asserts. What must hold here is that the drift is *first
    order* in that phase and nothing worse: a tenth of the pulse energy must move the
    total by a tenth as much.
    """
    modes = modes_of(capillary, 2)
    rule = transverse_quadrature(
        modes, n_r=CONVERGED_N_R_SET, radial_rule="gauss_kronrod"
    )

    def drift_at(energy):
        problem = build(grid, modes, pulse(grid, 2, energy), quadrature=rule)
        plane, _ = propagate(backend_class, problem, grid)
        before = total_energy(grid, problem.initial.field0[0])
        return abs(total_energy(grid, plane.field[0]) / before - 1.0)

    strong, weak = drift_at(ENERGY), drift_at(ENERGY / 10.0)
    # Measured 1.0072e-8 and 1.0072e-9, a ratio of 10.000 over the decade. The window
    # allows for the second-order term, which is a further factor of ten down.
    assert 9.9 < strong / weak < 10.1
    assert strong < 1e-7


# ----------------------------------------------------------------------- node chunks


def test_a_chunk_of_one_node_agrees_with_all_of_them(backend_class, grid, capillary):
    """The chunk loop is an ordering of one sum, so it changes nothing but round-off."""
    modes = modes_of(capillary, 2)
    rule = transverse_quadrature(modes, n_r=16, radial_rule="gauss_kronrod")
    start = pulse(grid, 2)
    whole, backend = propagate(
        backend_class,
        build(grid, modes, start, quadrature=rule, node_budget=1 << 30),
        grid,
    )
    chunked, small = propagate(
        backend_class, build(grid, modes, start, quadrature=rule, node_budget=1), grid
    )
    assert backend._problem.transform.node_chunk == rule.n_nodes
    assert small._problem.transform.node_chunk == 1
    # Seventeen rank-one updates against one GEMM over seventeen columns: Higham's
    # n * eps is 3.8e-15. Measured 1.8e-15.
    assert peak_gap(chunked.field, whole.field) < 1e-13


def test_the_memory_ledger_sizes_the_node_buffers_by_the_chunk(
    backend_class, grid, capillary
):
    """Architecture section 13: the node buffers follow the chunk, not the rule."""
    modes = modes_of(capillary, 2)
    rule = transverse_quadrature(modes, n_r=16, radial_rule="gauss_kronrod")
    start = pulse(grid, 2)
    whole = backend_class(
        build(grid, modes, start, quadrature=rule, node_budget=1 << 30), grid=grid
    ).memory_ledger()
    chunked = backend_class(
        build(grid, modes, start, quadrature=rule, node_budget=1), grid=grid
    ).memory_ledger()
    named = ("node_field", "node_polarisation")
    for name in named:
        assert whole[name] == rule.n_nodes * chunked[name]
    assert chunked[named[0]] == grid.n_to * 8
    # The modal buffers do not follow the chunk: they are the whole state in time.
    assert whole["modal_time"] == chunked["modal_time"] == 2 * grid.n_to * 8


# --------------------------------------------------------------- the quadrature alarm


def test_the_alarm_fires_on_a_rule_too_coarse_for_the_field(
    backend_class, grid, capillary, caplog
):
    """A statistic and a logged warning, never a controller input."""
    modes = modes_of(capillary, 3)
    rule = transverse_quadrature(modes, n_r=8, radial_rule="gauss_kronrod")
    problem = build(grid, modes, pulse(grid, 3), quadrature=rule)
    with caplog.at_level(logging.WARNING, logger="solprop.backends.numpy_ref.backend"):
        _, backend = propagate(backend_class, problem, grid)
    # Measured 1.6e-1: nine radial nodes cannot integrate the quartic of three Bessel
    # profiles, and no step size can fix that.
    assert quadrature_alarm(backend) > 1e-6
    assert (
        sum("transverse quadrature" in record.message for record in caplog.records) == 1
    )


def test_the_alarm_is_silent_on_a_rule_that_resolves_the_field(
    backend_class, grid, capillary, caplog
):
    modes = modes_of(capillary, 3)
    rule = transverse_quadrature(
        modes, n_r=CONVERGED_N_R_SET, radial_rule="gauss_kronrod"
    )
    problem = build(grid, modes, pulse(grid, 3), quadrature=rule)
    with caplog.at_level(logging.WARNING, logger="solprop.backends.numpy_ref.backend"):
        _, backend = propagate(backend_class, problem, grid)
    # Measured 7.2e-9, two orders below the alarm and falling by three orders for every
    # eight nodes added.
    assert 0.0 < quadrature_alarm(backend) < 1e-6
    assert not caplog.records


# The slow ladders run on the default backend only (`docs/dev/testing.md` section 2,
# "Tiers and backends"); the two alarm tests above run on both and stand in for the
# second copy.
DEFAULT_BACKEND = NumbaCpuBackend

# The record of the ladder below, the first row of `docs/dev/accuracy.md` section 2. The
# alarm spans fourteen orders across the ladder, so what a comparison can honestly
# assert is that each rung is within a factor of two of what it was; the finest is at
# the round-off floor of the projection itself and is carried by `atol` rather than
# compared.
ALARM_LADDER = Study(
    name="quadrature_alarm",
    title="The transverse quadrature alarm against the radial order, three modes",
    case={
        "modes": "the first three HE_1m modes of a lossless helium capillary, "
        "125 um radius, 1 bar, cladding 1.45",
        "grid": "field grid, 800 nm, 300 nm to 3 um, 100 fs, 2 cm",
        "pulse": "10 fs, 2 uJ, repeated into all three modes",
        "response": "Kerr with third harmonic, chi3 = 2.2e-52 m^2/V^2",
        "rule": "Gauss-Legendre with the embedded Kronrod pair that gives the alarm",
    },
    axis={"name": "n_r", "unit": "nodes", "label": "radial quadrature order"},
    columns=(
        Column(
            name="alarm",
            unit="1 (relative, the embedded pair's disagreement)",
            why="a factor of two on a quantity that falls fourteen orders across the "
            "ladder, with atol at the projection's own round-off floor so the "
            "converged rung is recorded rather than compared",
            bound=1e-6,
            rtol=1.0,
            atol=1e-14,
        ),
    ),
)


@pytest.mark.slow
def test_the_alarm_falls_with_the_radial_order_on_a_three_mode_set(
    grid, capillary, request
):
    """The quadrature alarm's convergence ladder, run rather than quoted.

    Measured here: **2.365e-1, 3.316e-2, 1.451e-4, 1.776e-8 and 4.264e-15** at ``n_r`` =
    8, 16, 24, 32 and 48 -- between three and four orders for every eight nodes added,
    and the 1e-6 threshold falling between 24 and 32, which is why a *set* needs the
    higher order that a single HE_11 does not (`CONVERGED_N_R` is 24,
    `CONVERGED_N_R_SET` is 32). The reference backend gives the same five numbers, the
    last to 4.251e-15.

    Slow because it is a convergence ladder whose numbers are recorded
    (`docs/dev/testing.md` section 2), not because of what it costs: five runs of a
    2 cm three-mode case.
    """
    modes = modes_of(capillary, 3)
    orders = (8, 16, 24, CONVERGED_N_R_SET, 48)
    alarms = {}
    for n_r in orders:
        rule = transverse_quadrature(modes, n_r=n_r, radial_rule="gauss_kronrod")
        problem = build(grid, modes, pulse(grid, 3), quadrature=rule)
        _, backend = propagate(DEFAULT_BACKEND, problem, grid)
        alarms[n_r] = float(backend._alarm)
    # The ladder is monotone and spans more than ten orders, and the threshold lands
    # where the two constants at the top of this file say it does.
    assert sorted(alarms, key=lambda order: alarms[order], reverse=True) == list(orders)
    assert alarms[orders[0]] / alarms[orders[-1]] > 1e10
    # The threshold the backend warns at, as the two tests below assert it.
    assert alarms[24] > 1e-6 > alarms[CONVERGED_N_R_SET]
    record(
        ALARM_LADDER,
        request=request,
        backend="numba_cpu",
        rows=[
            {"series": "three HE_1m modes", "axis": n_r, "alarm": alarm}
            for n_r, alarm in alarms.items()
        ],
    )


def test_a_rule_with_no_embedded_pair_reports_nothing_rather_than_zero(
    backend_class, grid, capillary, caplog
):
    """The difference of a rule with itself is zero, which is not convergence."""
    modes = modes_of(capillary, 2)
    rule = transverse_quadrature(modes, n_r=8, radial_rule="gauss")
    problem = build(grid, modes, pulse(grid, 2), quadrature=rule)
    with caplog.at_level(logging.WARNING, logger="solprop.backends.numpy_ref.backend"):
        _, backend = propagate(backend_class, problem, grid)
    assert math.isnan(quadrature_alarm(backend))
    assert not caplog.records


def test_the_alarm_is_not_a_number_where_the_polarisation_vanishes(grid, capillary):
    """A true vacuum end has nothing to score: ``nan``, not ``inf``."""
    modes = modes_of(capillary, 2)
    rule = transverse_quadrature(
        modes, n_r=CONVERGED_N_R_SET, radial_rule="gauss_kronrod"
    )
    problem = build(grid, modes, pulse(grid, 2), quadrature=rule)
    _, backend = propagate(NumpyRefBackend, problem, grid)
    assert backend._alarm > 0.0
    backend._last_projected = np.zeros_like(backend._last_projected)
    backend._last_coarse = np.zeros_like(backend._last_coarse)
    backend._update_alarm()
    assert math.isnan(backend._alarm)


# ------------------------------------------------------------- vector polarisation


@pytest.mark.parametrize("angle", [0.0, math.pi / 4, math.pi / 2, 0.7])
def test_a_linearly_polarised_run_is_the_scalar_one_rotated(
    backend_class, grid, capillary, angle
):
    """The vector-isotropic cubic couples the components of one point and nothing else.

    ``P_p = c rho (sum_q E_q^2) E_p`` with a fixed-angle linear field is the scalar
    problem in a rotated frame, and the linear operator is the same for both components,
    so the two components must be the scalar answer times the direction cosines. This is
    the parity test of ``tests/nonlinear/test_plasma.py``, carried through the whole
    pipeline.
    """
    start = pulse(grid)
    scalar, _ = propagate(backend_class, build(grid, capillary, start), grid)
    components = np.concatenate(
        [math.cos(angle) * start, math.sin(angle) * start], axis=1
    )
    vector, _ = propagate(backend_class, build(grid, capillary, components), grid)
    assert vector.field.shape[2] == 2
    peak = np.max(np.abs(scalar.field))
    for index, cosine in enumerate((math.cos(angle), math.sin(angle))):
        gap = np.max(np.abs(vector.field[0, :, index] - cosine * scalar.field[0, :, 0]))
        # Peak-normalised: the field passes through zero at every carrier crossing, so a
        # relative comparison is meaningless there. Measured worst 5.1e-16 over the four
        # angles; the bound is the machine epsilon times the 257-bin reduction.
        assert gap / peak < 1e-14


@pytest.mark.parametrize(
    "jones",
    [(math.cos(0.7), math.sin(0.7)), (1 / math.sqrt(2), 1j / math.sqrt(2))],
    ids=["linear", "circular"],
)
def test_the_two_routes_to_a_vector_run_agree_on_a_uniformly_polarised_pair(
    grid, capillary, jones
):
    """The polarisation-resolved route and the product route, end to end.

    The components live in the transform's rows *or* on the state's polarisation axis,
    so a vector multimode run has two routes: the HE_11 pair as two modes whose
    profiles carry x and y (``components=("x", "y")``, one state component), or the one
    spatial HE_11 with one resolved component and two state components. For a set
    whose profiles are uniformly polarised they are the same equation. The pair's
    cross components are exact zeros, so the resolved route's synthesis and
    projection add only exact zeros to the product route's sums, and the two can differ
    only by the order of a reduction.

    Bound, derived before the comparison: a reduction of ``k`` terms reordered moves
    its result by at most ``gamma_k = k u / (1 - k u)`` of the sum of the magnitudes,
    ``k`` the projection's ``2 n_nodes`` rows plus the synthesis's two modes, once per
    stage of each of the ten steps (seven stages of DOPRI5), relative to the peak. The
    circular case is the one the scalar problem cannot reproduce: its cubic couples
    the two components. Measured: bitwise equal in both cases.
    """
    steps = 10
    fixed = {
        "dz_max": LENGTH / steps,
        "dz_init": LENGTH / steps,
        "schedule": FixedSchedule.uniform(
            0.0, LENGTH, LENGTH / steps, save_planes=[LENGTH]
        ),
    }
    start = pulse(grid)
    pair = ModeCollection.with_polarisations([capillary])
    resolved_rule = transverse_quadrature(
        pair, n_r=CONVERGED_N_R, radial_rule="gauss_kronrod"
    )
    resolved = build(
        grid,
        pair,
        np.concatenate([jones[0] * start, jones[1] * start], axis=0),
        quadrature=resolved_rule,
        components=("x", "y"),
        **fixed,
    )
    product = build(
        grid,
        [capillary],
        np.concatenate([jones[0] * start, jones[1] * start], axis=1),
        quadrature=transverse_quadrature(
            [capillary], n_r=CONVERGED_N_R, radial_rule="gauss_kronrod"
        ),
        **fixed,
    )
    assert resolved.shape[1:3] == (2, 1)
    assert product.shape[1:3] == (1, 2)
    # The premise: each of the pair's profiles has an exactly zero cross component.
    rows = resolved.transform.synthesis.reshape(-1, 2, 2)
    assert np.all(rows[:, 1, 0] == 0.0)
    assert np.all(rows[:, 0, 1] == 0.0)
    by_rows, _ = propagate(NumpyRefBackend, resolved, grid)
    by_state, _ = propagate(NumpyRefBackend, product, grid)
    k = 2 * resolved_rule.n_nodes + 2
    unit = np.finfo(float).eps / 2
    bound = steps * 7 * k * unit / (1 - k * unit)
    peak = np.max(np.abs(by_state.require_field()))
    for component in range(2):
        gap = np.max(
            np.abs(
                by_rows.require_field()[0, component, 0]
                - by_state.require_field()[0, 0, component]
            )
        )
        assert gap / peak <= bound


def test_a_transform_and_a_state_cannot_both_resolve_the_components(grid, capillary):
    """The components live in the transform's rows or in the state, never in both."""
    rule = transverse_quadrature([capillary], n_r=8, radial_rule="gauss_kronrod")
    start = pulse(grid)
    with pytest.raises(TranslationError, match="with_polarisations"):
        build(
            grid,
            [capillary],
            np.concatenate([start, start], axis=1),
            quadrature=rule,
            components=("x", "y"),
        )


# --------------------------------------------------------------- backend statistics


def on_axis_intensity(grid, modes, field, z=0.0):
    """The peak on-axis intensity from the modes' own profiles, by hand."""
    profiles = np.array(
        [
            mode.field(np.zeros((1, 2)), z)[0, 0] / math.sqrt(mode.normalisation(z))
            for mode in modes
        ]
    )
    analytic = analytic_field(grid, field)
    on_axis = np.tensordot(profiles, analytic[:, 0, :], axes=([0], [0]))
    weight = np.ones_like(field.real)
    for index, mode in enumerate(modes):
        weight[index, :, grid.active_mask] = np.real(
            mode.neff(grid.omega[grid.active_mask], z)
        )[:, None]
    weighted = analytic_field(grid, field * weight)
    weighted_axis = np.tensordot(profiles, weighted[:, 0, :], axes=([0], [0]))
    # The cycle-averaged Poynting flux, not the vacuum squared field.
    return 0.5 * EPS0 * C * float(np.max(np.real(on_axis * np.conj(weighted_axis))))


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_the_peak_intensity_is_taken_on_the_axis(backend_class, grid, capillary):
    """The axis is not a quadrature node, so the mode fields are evaluated there."""
    modes = modes_of(capillary, 2)
    rule = transverse_quadrature(modes, n_r=16, radial_rule="gauss_kronrod")
    problem = build(grid, modes, pulse(grid, 2), quadrature=rule)
    backend = backend_class(problem, grid=grid, modes=modes)
    backend.set_initial(problem.initial.field0, None)
    plane = backend.observe()
    got = backend.stats(StatsSpec(names=("peak_intensity",)))["peak_intensity"]
    # Both routes evaluate the same Bessel profiles at r = 0 and the same analytic
    # signal, so they agree to the round-off of a two-term sum.
    assert float(got[0]) == pytest.approx(
        on_axis_intensity(grid, modes, plane.field[0]), rel=1e-14
    )


def test_the_on_axis_intensity_needs_the_mode_set(backend_class, grid, capillary):
    """The mode fields at the axis are not in the kernel contract."""
    modes = modes_of(capillary, 2)
    rule = transverse_quadrature(modes, n_r=16, radial_rule="gauss_kronrod")
    problem = build(grid, modes, pulse(grid, 2), quadrature=rule)
    backend = backend_class(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    with pytest.raises(BackendError, match="needs the mode set"):
        backend.stats(StatsSpec(names=("peak_intensity",)))


def test_the_alarm_is_reported_as_a_statistic(backend_class, grid, capillary):
    """A statistic, and the same number the warning quotes."""
    modes = modes_of(capillary, 3)
    rule = transverse_quadrature(
        modes, n_r=CONVERGED_N_R_SET, radial_rule="gauss_kronrod"
    )
    problem = build(grid, modes, pulse(grid, 3), quadrature=rule)
    _, backend = propagate(backend_class, problem, grid)
    values = backend.stats(StatsSpec(names=("quadrature_error", "energy")))
    assert float(values["quadrature_error"][0]) == quadrature_alarm(backend)
    assert values["quadrature_error"].shape == (1,)
    assert values["energy"].shape == (1, 3, 1)
