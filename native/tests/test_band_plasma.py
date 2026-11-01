# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""solprop SEP 18's band rule for the field plasma column, against solprop's reference.

The band rule filters each of the column's two running integrals with the grid's
symmetric taps before the cumulative trapezoid, closes the window's ends by point
reflection and floors the exposure at zero (solprop ``RunningIntegral``). Every build
runs it as a team column in whole-row passes, a band filter then a chunked scan per
running integral (``include/solprop_native/plasma.hpp``); a host team splits a row
between its threads when a chunk has fewer nodes than threads, and skips each row's
quiet samples. Its statistics and run diagnostics come from the band scan. The bounds
are the equivalence ladder's (solprop's ``accelerator_bounds``): the transforms' depth
plus the plasma column's sums, which the band rule widens by its ``2K + 1`` taps per
integrand, per evaluation of an explicit seven-stage step. The tests marked ``gpu`` are
the device qualification's.
"""

# ruff: noqa: I001 -- Kokkos must load before solprop imports pyFFTW on macOS.

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import cast

import solprop_native

import numpy as np
import numpy.typing as npt
import pytest
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import Simulation
from solprop.exceptions import BackendError, MaterialRangeWarning, TranslationError
from solprop.grid import FieldGrid, FreeGrid, RadialGrid
from solprop.nonlinear.ionisation import IntegrationPolicy, RunningIntegral
from solprop.nonlinear.responses import PlasmaResponse
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import PLASMA_INTEGRATION_STATISTICS, StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.ionisation import build_rate_table
from solprop.translate.plasma_integration import design_running_integral
from solprop.translate.problem import KernelProblem, assemble

from legacy_geometry import LEGACY_GRID
from bounds import fft_bound
from conftest import PhaseACase
from plasma_bounds import assert_regression_guard, plasma_stage_bound, rate_roundoff
from solprop_native import NativeBackend, _core
from solprop_native.backend import (
    _native_problem,  # pyright: ignore[reportPrivateUsage] - corrupt native payloads
)

INFO = solprop_native.build_info()
DEVICE_BUILD = INFO.device_backend != "none"
STEPS = 2
"""Fixed steps of every propagated case."""
LENGTH = 1e-4
"""The spatial cases' length, m."""
STATISTICS = (
    "electron_density",
    "peak_ionisation_rate",
    *PLASMA_INTEGRATION_STATISTICS,
)
type ComplexArray = npt.NDArray[np.complex128]

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)


class QuadraticRate:
    """An analytic ionisation law, independent of material fit data.

    ``W = scale (E / 1e9 V/m)^2`` above a threshold field, capped. Optmat's tabulated
    rates would do as well, but loading their sources costs the first case of every
    process minutes on a contended file system.
    """

    maximum = 1e17

    def __init__(self, scale: float = 3e12, threshold: float = 1e3) -> None:
        self.scale = scale
        self.threshold = threshold
        self.citation = (
            f"Analytic test rate W={scale:g}*(E/1e9)^2 s^-1 above {threshold:g} V/m"
        )
        self.parameters = (
            ("model", "quadratic-band-oracle"),
            ("reference", 1e9),
            ("scale", scale),
            ("threshold", threshold),
        )

    def __call__(self, field_amplitude: object) -> npt.NDArray[np.float64]:
        values = np.abs(np.asarray(field_amplitude, dtype=float))
        return np.where(
            values < self.threshold,
            0.0,
            np.minimum(self.maximum, self.scale * (values / 1e9) ** 2),
        )


def evaluation_bound(problem: KernelProblem, grid: FieldGrid) -> float:
    """Bound one evaluation relative to its largest value, fixed before it runs.

    The transforms, the Kerr term and the pointwise products take the predeclared
    round-trip envelope :func:`bounds.fft_bound`; the plasma column takes its own chain
    through the three scans, the fraction, the currents, the density and the forward
    transform's ``l1 -> linf`` (:mod:`plasma_bounds`), at this machine's thread count.
    """
    return fft_bound(grid.n_to) + plasma_stage_bound(
        problem, grid, INFO.runtime_threads
    )


def march_bound(problem: KernelProblem, grid: FieldGrid, steps: int) -> float:
    """Bound `steps` fixed steps of a seven-stage explicit pair.

    Each stage's departure is at most :func:`evaluation_bound` of its value; the cases'
    nonlinear phases are far below a radian over a step and the linear flow is unitary
    to their loss, so a stage's contribution is at most that fraction of the field and
    the stages' contributions add.
    """
    return max(1, 7 * steps) * evaluation_bound(problem, grid)


def banded(problem: KernelProblem, rule: RunningIntegral) -> KernelProblem:
    """Return `problem` with its plasma integrated by `rule`."""
    responses = dataclasses.replace(problem.responses, plasma_integral=rule)
    return dataclasses.replace(problem, responses=responses)


def modal(
    kind: str, integration: IntegrationPolicy = "band", *, averaged: bool = False
) -> tuple[FieldGrid, KernelProblem]:
    """Return a cheap capillary over two fixed steps: argon, or a helium-argon mix.

    `averaged` drops the radial quadrature and the vector components: one node and one
    scalar mode, whose transforms are single products.
    """
    length = 1e-3
    # A few per cent ionised at the pulse's peak, not saturating; the threshold switches
    # the rate on abruptly, which the band filter undershoots beside (SEP 18 4.5).
    rate = QuadraticRate(2e10, 3e9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        simulation = Simulation.from_capillary(
            125e-6,
            length,
            {"He": 0.5, "Ar": 0.5} if kind == "mixture" else "Ar",
            1.0,
            plasma={"He": QuadraticRate(2e9, 3e9), "Ar": rate}
            if kind == "mixture"
            else rate,
            ionisation_field_range=(1e3, 1e11),
            # Pre-ionised, so the free current's integrand is live at the window's
            # start and the boundary level is measured.
            preionisation=1e-3 if kind == "analytic" else None,
            wavelength=800e-9,
            wavelength_limits=(200e-9, 2e-6),
            trange=100e-15,
            fwhm=10e-15,
            energy=10e-6,
            n_r=8,
            n_save=2,
            rtol=1e-6,
            modes=2 if kind == "multimode" else 1,
            # Without the harmonic the Kerr term reads the analytic signal, and the
            # plasma its real part.
            thg=kind != "analytic",
            plasma_integration=integration,
            schedule=FixedSchedule.uniform(0.0, length, length / STEPS),
            absorber_scheme="split_step",
            noise=None,
        )
        if averaged:
            simulation.settings = dataclasses.replace(
                simulation.settings, quadrature=None, components=None
            )
        problem = simulation.translate()
    grid = simulation.grid
    assert isinstance(grid, FieldGrid)
    return grid, problem


def spatial(kind: str, steps: int = STEPS) -> tuple[FieldGrid, KernelProblem]:
    """Return argon on a radial or Cartesian route, two components, the band rule."""
    grid = FieldGrid.from_limits(
        LENGTH, 800e-9, (600e-9, 1.2e-6), 200e-15, **LEGACY_GRID
    )
    space = (
        RadialGrid(6e-4, 16, window_factor=0.0)
        if kind == "radial"
        else FreeGrid(3e-4, 9, 4e-4, 12, window_factor=0)
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7),
        80e-6,
        jones=(1, 0.5j),
    )
    problem = assemble(
        grid,
        None,
        [PlasmaResponse("Ar", build_rate_table(QuadraticRate(), (1e3, 1e10)))],
        pulse.build(grid, space, medium=SpatialMedium()),
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[LENGTH],
        dz_max=LENGTH,
        dz_init=LENGTH / steps,
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / steps),
        density=1e22,
    )
    if kind == "free":
        # A remainder chunk, so the band column's rings meet a partial block.
        problem = dataclasses.replace(
            problem, transform=dataclasses.replace(problem.transform, node_chunk=7)
        )
    return grid, banded(problem, design_running_integral(grid, "band"))


CASES = ("mode_averaged", "analytic", "multimode", "mixture", "radial", "free")


def case(name: str) -> tuple[FieldGrid, KernelProblem]:
    """Return the banded ``(grid, problem)`` of one case, asking every statistic."""
    grid, problem = spatial(name) if name in {"radial", "free"} else modal(name, "band")
    return grid, dataclasses.replace(problem, stats_spec=StatsSpec(names=STATISTICS))


def march(
    implementation: type[NumpyRefBackend] | type[NativeBackend],
    problem: KernelProblem,
    grid: FieldGrid,
) -> tuple[ComplexArray, dict[str, np.ndarray]]:
    """Return the field and the statistics after the problem's fixed steps."""
    backend = implementation(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(float(problem.stepper.save_planes[-1]))
        return backend.observe().require_field().copy(), backend.stats(
            problem.stats_spec
        )
    finally:
        backend.close()


def assert_statistics_agree(
    actual: dict[str, np.ndarray], expected: dict[str, np.ndarray], bound: float
) -> None:
    """The plasma statistics to the march's bound, the diagnostics absolutely.

    The diagnostics are dimensionless ratios judged against thresholds of 1e-2 and
    1e-4, and the work defect is a difference of nearly equal sums at the level of
    their rounding, so they agree to an absolute 1e-10 rather than relatively. A rate
    statistic moves by the rate's sensitivity times the field's error, a thousand
    covering the quadratic law and the exponential of the tabulated ones.
    """
    for statistic in STATISTICS:
        want = np.asarray(expected[statistic], dtype=float)
        got = np.asarray(actual[statistic], dtype=float)
        assert got.shape == want.shape, statistic
        if statistic in PLASMA_INTEGRATION_STATISTICS:
            np.testing.assert_allclose(
                got, want, rtol=0.0, atol=1e-10, err_msg=statistic
            )
        else:
            np.testing.assert_allclose(
                got, want, rtol=1e3 * bound, atol=0.0, err_msg=statistic
            )


@pytest.mark.skipif(
    not INFO.cpu_size_policy_enabled or bool(INFO.explicit_thread_pools & 1),
    reason="requires automatic OpenMP response sizing",
)
@pytest.mark.parametrize("chunk", [19, 20, 44])
def test_response_column_boundary_preserves_full_and_tail_windows(chunk: int) -> None:
    """Check full/tail chunks across the width boundary within SEP 25 bounds."""
    grid, problem = case("free")
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=chunk)
    )
    bound = march_bound(problem, grid, STEPS)
    expected, expected_stats = march(NumpyRefBackend, problem, grid)
    backend = NativeBackend(problem, grid=grid)
    try:
        sizing = cast(dict[str, dict[str, int]], backend.metadata()["execution_sizing"])
        columns = 2 * chunk
        key = f"kernels.response.{columns * grid.n_to}.samples.{grid.n_to}"
        choice = sizing[key]
        assert choice["columns"] == columns
        # A kernel pool that already holds a full-width instance keeps its team: a
        # medium-width response takes the full pool too, except on macOS.
        full_team = sys.platform != "darwin" and any(
            entry["threads"] == INFO.thread_count
            for name, entry in sizing.items()
            if name.startswith("kernels.") and name != key
        )
        assert choice["threads"] == (
            min(INFO.thread_count, 4)
            if columns < 40 and not full_team
            else INFO.thread_count
        )
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(float(problem.stepper.save_planes[-1]))
        actual = backend.observe().require_field()
        actual_stats = backend.stats(problem.stats_spec)
        scale = float(np.max(np.abs(expected)))
        assert scale > 0.0
        assert float(np.max(np.abs(actual - expected))) <= bound * scale
        assert_statistics_agree(actual_stats, expected_stats, bound)
    finally:
        backend.close()


@pytest.mark.gpu
@pytest.mark.parametrize("name", CASES)
def test_the_band_rule_matches_the_reference_on_every_route(name: str) -> None:
    grid, problem = case(name)
    assert problem.responses.plasma_integral.policy == "band"
    assert problem.responses.plasma_integral.half_width >= 1
    expected, expected_stats = march(NumpyRefBackend, problem, grid)
    actual, actual_stats = march(NativeBackend, problem, grid)
    bound = march_bound(problem, grid, STEPS)
    scale = float(np.max(np.abs(expected)))
    assert float(np.max(np.abs(actual - expected))) <= bound * scale
    assert_regression_guard(actual, expected, f"route {name} {STEPS} steps")
    assert_statistics_agree(actual_stats, expected_stats, bound)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("trange", "ffac", "half"),
    [
        (5e-15, 2.0, 6),  # n_to = 8 < 2K: both ends reflect into one window
        (5e-15, 4.0, 8),  # n_to = 16 = 2K: the two edge regions meet
        (1e-14, 4.0, 8),  # n_to = 32 < 3K + 2K + 1: every stage on an edge
    ],
)
def test_overlapping_stencil_windows_refuse_uncertified_collar(
    trange: float, ffac: float, half: int
) -> None:
    """Refuse short historical stencil fixtures under the held-source contract."""
    grid = FieldGrid.from_limits(
        LENGTH, 800e-9, (600e-9, 1.2e-6), trange, ffac=ffac, **LEGACY_GRID
    )
    space = RadialGrid(6e-4, 16, window_factor=0.0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=2e-15, energy=1e-8),
        80e-6,
        jones=(1, 0.5j),
    )
    # These stencil stress windows cannot certify the held-source collar.
    # Their raw band closures remain tested natively in test_core.cpp.
    assert grid.n_to < 3 * half + 2 * half + 1
    with pytest.raises(TranslationError, match="held plasma current"):
        assemble(
            grid,
            None,
            [
                PlasmaResponse(
                    "Ar",
                    build_rate_table(QuadraticRate(), (1e3, 1e10)),
                    integration="trapezoid",
                )
            ],
            pulse.build(grid, space, medium=SpatialMedium()),
            spatial_grid=space,
            medium=SpatialMedium(),
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[LENGTH],
            dz_max=LENGTH,
            dz_init=LENGTH / STEPS,
            schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / STEPS),
            density=1e22,
        )


def test_the_band_cases_ionise() -> None:
    """The cases are not vacuous: their plasma moves the field and frees electrons."""
    for name in ("mode_averaged", "mixture", "radial"):
        grid, problem = case(name)
        _, stats = march(NativeBackend, problem, grid)
        assert float(stats["electron_density"][0]) > 0.0, name
        assert float(stats["plasma_work_defect"][0]) < 1e-4, name


@pytest.mark.gpu
def test_one_evaluation_matches_the_reference() -> None:
    """The first stage's nonlinear derivative, two components on the radial route."""
    grid, problem = spatial("radial")
    reference = NumpyRefBackend(problem, grid=grid)
    try:
        reference.set_initial(problem.initial.field0, None)
        reference._hoist(0.0)  # pyright: ignore[reportPrivateUsage] - oracle probe
        expected = reference._nonlinear(  # pyright: ignore[reportPrivateUsage]
            reference._require_state(),  # pyright: ignore[reportPrivateUsage]
            reference._stage_scalars(0.0, 0.0),  # pyright: ignore[reportPrivateUsage]
            np.ones(problem.shape, complex),
        )[0]
    finally:
        reference.close()
    backend = NativeBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        # A zero-length trial exposes the first stage, N(E, z = 0).
        backend._trial_step(0.0, 0.0)  # pyright: ignore[reportPrivateUsage]
        actual = np.asarray(
            backend._native_state.stage_to_host(0)  # pyright: ignore[reportPrivateUsage]
        ).reshape(problem.shape[1:])
    finally:
        backend.close()
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    assert (
        float(np.max(np.abs(actual - expected)))
        <= evaluation_bound(problem, grid) * scale
    )
    assert_regression_guard(actual, expected, "radial evaluation")


@pytest.mark.parametrize(
    ("kind", "integration", "averaged"),
    [
        ("mode_averaged", "band", True),
        ("mode_averaged", "trapezoid", True),
        ("multimode", "band", False),
    ],
)
def test_every_stage_of_a_trial_matches_the_reference(
    kind: str, integration: IntegrationPolicy, averaged: bool
) -> None:
    """Each stage of one trial, through the fused stage kernels (solprop SEP 25 6.7).

    A stage's finish runs in the next stage's entry kernel, and a single node's
    synthesis and projection in its response columns; each is the arithmetic of the
    kernels it replaces. Stage ``k`` carries its own evaluation's departure
    (:func:`evaluation_bound`) and, through its combination, those of the ``k``
    stages before it, scaled by ``dz`` times the tableau and the response's gain, which
    the cases' small nonlinear phases keep below one: ``k + 1`` evaluation bounds.
    """
    grid, problem = modal(kind, integration, averaged=averaged)
    dz = float(problem.stepper.save_planes[-1]) / STEPS
    reference = NumpyRefBackend(problem, grid=grid)
    try:
        reference.set_initial(problem.initial.field0, None)
        reference._trial_step(0.0, dz)  # pyright: ignore[reportPrivateUsage] - oracle
        expected = [
            np.asarray(stage)[0]
            for stage in reference._last_stages  # pyright: ignore[reportPrivateUsage]
        ]
    finally:
        reference.close()
    backend = NativeBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        if averaged and not DEVICE_BUILD:
            assert backend.metadata()["synthesis_provider"] == "folded"
            assert backend.metadata()["projection_provider"] == "folded"
        backend._trial_step(0.0, dz)  # pyright: ignore[reportPrivateUsage]
        state = backend._native_state  # pyright: ignore[reportPrivateUsage]
        actual = [
            np.asarray(state.stage_to_host(stage)).reshape(problem.shape[1:])
            for stage in range(len(expected))
        ]
    finally:
        backend.close()
    bound = evaluation_bound(problem, grid)
    for stage, (got, want) in enumerate(zip(actual, expected, strict=True)):
        scale = float(np.max(np.abs(want)))
        assert scale > 0.0, stage
        assert float(np.max(np.abs(got - want))) <= (stage + 1) * bound * scale, stage
        assert_regression_guard(got, want, f"{kind} {integration} stage {stage}")


@pytest.mark.parametrize("kind", ["multimode", "analytic", "mixture"])
@pytest.mark.parametrize("chunk", [1, 4])
def test_quadrature_keeps_accepted_scores_through_rejection_and_reset(
    kind: str, chunk: int
) -> None:
    """Score held field currents once, preserving accepted observations on rejection.

    The norm difference divided by the fine norm takes at most eight source
    rounding budgets when both projection norms are comparable. Accumulate the
    existing seven-stage, two-step plasma budget before either backend runs.
    """
    grid, original = modal(kind)
    problem = dataclasses.replace(
        original,
        transform=dataclasses.replace(original.transform, node_chunk=chunk),
        stepper=dataclasses.replace(original.stepper, integrator=None),
    )
    bound = march_bound(problem, grid, STEPS)
    score_bound = 8 * bound
    assert bound > 0.0
    specification = StatsSpec(names=("quadrature_error", "electron_density"))
    reference = NumpyRefBackend(problem, grid=grid)
    backend = NativeBackend(problem, grid=grid)
    state = backend._native_state  # pyright: ignore[reportPrivateUsage] - score lifetime
    dz = float(problem.stepper.save_planes[-1]) / STEPS
    try:
        for engine in (reference, backend):
            engine.set_initial(problem.initial.field0, None)
            engine.propagate_to(dz)
        accepted = backend.observe().require_field().copy()
        score = state.quadrature_error
        assert math.isfinite(score)
        assert state.quadrature_peak() == score  # One accepted step since the reset.
        assert math.isnan(state.quadrature_peak())
        expected = reference.stats(specification)["quadrature_error"]
        np.testing.assert_allclose(score, expected, rtol=0.0, atol=score_bound)

        # Leave an unaccepted trial in scratch, then observe the accepted field and
        # plasma statistics before the next trial replaces that scratch again.
        backend._trial_step(dz, 0.5 * dz)  # pyright: ignore[reportPrivateUsage] - rejected trial
        observed = backend.stats(specification)
        assert state.quadrature_error == score  # Stored accepted value, no arithmetic.
        assert math.isnan(state.quadrature_peak())
        np.testing.assert_array_equal(backend.observe().require_field(), accepted)
        np.testing.assert_allclose(
            observed["quadrature_error"], expected, rtol=0.0, atol=score_bound
        )
        for engine in (reference, backend):
            engine.propagate_to(2 * dz)
        want = reference.observe().require_field()
        got = backend.observe().require_field()
        assert np.isfinite(got).all()
        np.testing.assert_allclose(
            got, want, rtol=0.0, atol=bound * float(np.max(np.abs(want)))
        )
        np.testing.assert_allclose(
            state.quadrature_error,
            reference.stats(specification)["quadrature_error"],
            rtol=0.0,
            atol=score_bound,
        )
        backend.set_initial(problem.initial.field0, None)
        assert math.isnan(state.quadrature_error)
        assert math.isnan(state.quadrature_peak())
        backend.propagate_to(dz)
        np.testing.assert_allclose(
            backend.observe().require_field(),
            accepted,
            rtol=0.0,
            atol=bound * float(np.max(np.abs(accepted))),
        )
    finally:
        backend.close()
        reference.close()


@pytest.mark.gpu
def test_a_trapezoid_problem_reports_the_run_diagnostics() -> None:
    """The trapezoid answers the diagnostics from the band scan with its one tap."""
    grid, problem = modal("mixture", "trapezoid")
    problem = dataclasses.replace(problem, stats_spec=StatsSpec(names=STATISTICS))
    assert problem.responses.plasma_integral.policy == "trapezoid"
    expected, expected_stats = march(NumpyRefBackend, problem, grid)
    actual, actual_stats = march(NativeBackend, problem, grid)
    bound = march_bound(problem, grid, STEPS)
    assert float(np.max(np.abs(actual - expected))) <= bound * float(
        np.max(np.abs(expected))
    )
    assert_statistics_agree(actual_stats, expected_stats, bound)
    # The trapezoid's own work defect is its finite-step defect, never zero here.
    assert float(actual_stats["plasma_work_defect"][0]) > 0.0


@pytest.mark.gpu
def test_a_band_rule_of_one_tap_is_the_trapezoid() -> None:
    """``K = 0``: the band column and the trapezoid's column agree to rounding.

    The identity tap filters exactly and a trapezoid exposure is never negative, so the
    two columns differ only in the band column's unfused products and, for the team
    column, its scans' association.
    """
    grid, problem = spatial("radial")
    single = RunningIntegral(
        "band", np.ones(1), dt=grid.dt_fine, band_edge=float(np.max(grid.omega))
    )
    fields: list[ComplexArray] = []
    for rule in (RunningIntegral.trapezoid(), single):
        field, _ = march(NativeBackend, banded(problem, rule), grid)
        fields.append(field)
    scale = float(np.max(np.abs(fields[0])))
    assert (
        float(np.max(np.abs(fields[1] - fields[0])))
        <= march_bound(problem, grid, STEPS) * scale
    )


TILED = INFO.band_column == "tiled"
"""Whether the band rule runs the tiled column (a device's, or set at start)."""


def band_slots(
    problem: KernelProblem, backend: NativeBackend, band: bool = True
) -> int:
    """Return the band column's scratch slots: a device's chunk, the host's teams.

    The tiled column keeps a device's rows on the host too, for a band problem.
    """
    chunk = problem.transform.node_chunk
    keeps_chunk = DEVICE_BUILD or (TILED and band)
    if keeps_chunk:
        return chunk
    # Scratch follows the full chunk's effective response width, not the process cap.
    sizing = cast(dict[str, dict[str, int]], backend.metadata()["execution_sizing"])
    width = max(
        choice["threads"]
        for operation, choice in sizing.items()
        if operation.startswith("kernels.response.")
        and choice.get("fine_samples") == problem.grids.n_to
    )
    return min(chunk, width)


def band_tile_count(n_to: int) -> int:
    """Return the tiled column's tiles per row (`band_tile_samples`, plasma.hpp)."""
    tile = 256
    while 2 * tile <= 1024 and 2 * tile * 32 <= n_to:
        tile *= 2
    return -(-n_to // tile)


def test_the_band_rows_are_in_the_ledger() -> None:
    """The column's rows, by slot: rate, exposure, filtered, guard, and per component
    the integrand. Total current has its own ledger row. A device keeps its
    trapezoid column's rows, and the tiled column also keeps its tiles' totals."""
    grid, problem = spatial("free")
    backend = NativeBackend(problem, grid=grid)
    try:
        ledger = backend.memory_ledger()
        slots = band_slots(problem, backend)
    finally:
        backend.close()
    span = 2 * problem.responses.plasma_integral.half_width + 1
    rows = (3 + 3 * 2) if DEVICE_BUILD or TILED else (3 + 1 + 2)
    tiles = 2 * slots * band_tile_count(grid.n_to) if TILED else 0
    assert "plasma_band_rings" not in ledger
    assert ledger["plasma_device_scan"] == (rows * slots * grid.n_to + tiles) * 8
    assert (
        ledger["plasma_integration_diagnostics"]
        == (span // 2 + 1 + 6 * problem.transform.n_nodes) * 8
    )


@pytest.mark.skipif(DEVICE_BUILD, reason="a device keeps every row of its column")
def test_a_trapezoid_problem_keeps_only_the_statistics_rows() -> None:
    """The trapezoid's run diagnostics need the band statistics' three rows alone."""
    grid, problem = modal("mixture", "trapezoid")
    backend = NativeBackend(problem, grid=grid)
    try:
        ledger = backend.memory_ledger()
        slots = band_slots(problem, backend, False)
    finally:
        backend.close()
    assert ledger["plasma_device_scan"] == 3 * slots * grid.n_to * 8


@pytest.mark.parametrize("damage", ["too_many", "nonfinite", "empty", "identity"])
def test_native_taps_are_validated(damage: str) -> None:
    grid, problem = modal("mode_averaged", "band")
    backend = NativeBackend(problem, grid=grid)
    try:
        payload = _native_problem(
            problem,
            backend._terms,  # pyright: ignore[reportPrivateUsage] - fixed payload
        )
    finally:
        backend.close()
    taps = np.array(payload["plasma_taps"], dtype=float, copy=True)
    if damage == "too_many":
        payload["plasma_taps"] = np.ones(10)
    elif damage == "nonfinite":
        taps[1] = np.nan
        payload["plasma_taps"] = taps
    elif damage == "empty":
        payload["plasma_taps"] = np.empty(0)
    else:
        # The trapezoid is the one identity tap and nothing else.
        payload["plasma_band"] = False
    with pytest.raises(ValueError, match=r"taps|trapezoid|band"):
        _core.State(payload)


def test_the_diagnostics_need_a_field_plasma_column(phase_a_case: PhaseACase) -> None:
    problem, grid = phase_a_case
    backend = NativeBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        with pytest.raises(BackendError, match="field-grid plasma column"):
            backend.stats(StatsSpec(names=("plasma_undershoot",)))
    finally:
        backend.close()


@pytest.mark.skipif(DEVICE_BUILD, reason="a device build's host space is serial")
def test_the_band_column_agrees_across_thread_counts(tmp_path: Path) -> None:
    """One and four threads march the same fixed steps to within their budget.

    Four threads give each single-node window of 4096 samples a team of two, which
    scans each row in chunks; no bitwise agreement is promised across thread counts
    (solprop SEP 25):
    both sides perform the same
    discrete computation and are held to the reference comparison's own bounds, the
    march's for the fields and :func:`assert_statistics_agree`'s for the statistics.
    """
    script = Path(__file__).with_name("band_thread_probe.py")
    runs: list[dict[str, np.ndarray]] = []
    threads_seen: list[object] = []
    for threads in (1, 4):
        out = tmp_path / f"band-{threads}.npz"
        completed = subprocess.run(
            [sys.executable, str(script), str(out)],
            capture_output=True,
            text=True,
            check=False,
            env={
                **os.environ,
                "OMP_PROC_BIND": "false",
                "SOLPROP_NATIVE_THREADS": str(threads),
                "SOLPROP_OVERSUBSCRIBE": "1",
                "SOLPROP_THREADS": "",
                "SOLPROP_NATIVE_KERNEL_THREADS": str(threads),
            },
        )
        assert completed.returncode == 0, completed.stderr
        threads_seen.append(json.loads(completed.stdout.splitlines()[-1])["threads"])
        with np.load(out) as stored:
            runs.append({name: stored[name] for name in stored.files})
    assert threads_seen == [1, 4]
    one, four = runs
    from test_band_stress import case

    for label, (grid, problem), steps in (
        ("radial", spatial("radial"), STEPS),
        ("mixture", modal("mixture", "band"), STEPS),
        ("long_quiet", case("preionised_quiet", 10), 10),
        ("long_mixture", case("mixture_edge", 10), 10),
    ):
        bound = march_bound(problem, grid, steps)
        expected = one[f"{label}_field"]
        scale = float(np.max(np.abs(expected)))
        assert scale > 0.0
        assert float(np.max(np.abs(four[f"{label}_field"] - expected))) <= bound * scale
        assert_statistics_agree(
            {name: four[f"{label}_{name}"] for name in STATISTICS},
            {name: one[f"{label}_{name}"] for name in STATISTICS},
            bound,
        )


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["radial", "mixture"])
def test_the_band_column_repeats_to_rounding(name: str) -> None:
    """Two marches in one process, on one device, agree to the march's budget.

    Both are the same discrete computation, so they differ by rounding at most
    (solprop SEP 25): the reference comparison's own bounds hold them.
    """
    grid, problem = case(name)
    first, first_stats = march(NativeBackend, problem, grid)
    second, second_stats = march(NativeBackend, problem, grid)
    bound = march_bound(problem, grid, STEPS)
    scale = float(np.max(np.abs(first)))
    assert scale > 0.0
    assert float(np.max(np.abs(second - first))) <= bound * scale
    assert_statistics_agree(second_stats, first_stats, bound)


def test_the_rate_bound_preserves_the_exact_lower_clip() -> None:
    """A zero rate must not acquire subnormal loss work at zero field."""
    _, problem = case("mode_averaged")
    coefficients = problem.responses.plasma_coefficients
    assert coefficients is not None
    table = coefficients.rate
    field = np.asarray([[0.0, np.finfo(float).tiny, table.field_min / 2]])
    assert np.count_nonzero(rate_roundoff(table, field, np.zeros_like(field))) == 0
