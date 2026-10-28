# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The band column's stress inputs (solprop SEP 25 section 4.6), against the reference.

The host column runs whole-row passes: each running integral is a band filter and a
chunked scan, a team may split a row between its threads, and each pass is limited to
its row's live window (``include/solprop_native/plasma.hpp``). These cases aim at what
that can get wrong: a preionised field whose rate is identically zero (the current
lives with no exposure), a field just above threshold (a tiny exposure from a few
cycle peaks), pulses cut by either end of the window (the point reflections and the
windows reaching an end), a mixture there (several channels), and a two-component
field on the radial route. Each is compared at one evaluation, one fixed step and ten,
under a fixed-order method, with :mod:`test_band_plasma`'s bounds, decided before these
cases were measured.

The statistics at the launch plane hold the exposure scan to its own absolute bound
(solprop SEP 25 section 4.6, :mod:`plasma_bounds`): the final exposure is a prefix of
``n_to`` filtered rates, within ``2 gamma_D M[W] + (1 + gamma_D) M[dW]`` with ``M`` the
prefix of the filtered trapezoid's magnitudes (a reflected sample ``2 W_0 - W_m``
entering as ``2 |W_0| + |W_m|``), ``dW`` the rate's spread over the two sides' field
syntheses and ``D = n_to + T + 2K + 6``; the fraction moves by ``(1 - f_0) e^{-X}``
times that plus each side's roundings on the actual survival and fraction.
"""

# ruff: noqa: I001 -- Kokkos must load before solprop imports pyFFTW on macOS.

from __future__ import annotations

import dataclasses
import warnings

import solprop_native

import numpy as np
import numpy.typing as npt
import pytest
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import Simulation
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import FieldGrid, RadialGrid
from solprop.nonlinear.responses import PlasmaResponse
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController, RungeKutta
from solprop.translate.ionisation import build_rate_table
from solprop.translate.plasma_integration import design_running_integral
from solprop.translate.problem import KernelProblem, assemble

from solprop_native import NativeBackend
from legacy_geometry import LEGACY_GRID
from bounds import UNIT_ROUNDOFF
from plasma_bounds import (
    assert_regression_guard,
    node_field_error,
    rate_spread,
    scan_bound,
)
from test_band_plasma import (
    STATISTICS,
    QuadraticRate,
    assert_statistics_agree,
    banded,
    evaluation_bound,
    march_bound,
)

type ComplexArray = npt.NDArray[np.complex128]

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)

INFO = solprop_native.build_info()
THRESHOLD = 3e9
"""The quadratic rate's threshold field, V/m."""
LENGTH = 1e-3
"""The capillary cases' length, m."""
TRANGE = 400e-15
"""The capillary cases' window, s."""
CASES = (
    "preionised_quiet",
    "near_threshold",
    "leading_edge",
    "trailing_edge",
    "mixture_edge",
    "vector_edge",
)


def capillary(
    kind: str, steps: int, energy: float = 10e-6
) -> tuple[FieldGrid, KernelProblem]:
    """Return a mode-averaged argon (or helium-argon) capillary under the band rule.

    One node and ``n_to = 4096``: four threads give it a team of two, which splits
    each row's passes and scans into chunks.
    """
    # The fine window spans about +-228 fs: these pulses peak 13 fs from an end.
    t0 = {
        "leading_edge": -215e-15,
        "trailing_edge": 215e-15,
        "mixture_edge": 215e-15,
    }.get(kind, 0.0)
    # A long flat-topped pulse for the quiet case: many carrier cycles of a current
    # whose prefix cancels, and no exposure at all.
    pulse = GaussianPulse(
        wavelength=800e-9,
        fwhm=40e-15 if kind == "preionised_quiet" else 10e-15,
        m=6 if kind == "preionised_quiet" else 1,
        energy=energy,
        t0=t0,
    )
    rate = QuadraticRate(2e10, THRESHOLD)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        simulation = Simulation.from_capillary(
            125e-6,
            LENGTH,
            {"He": 0.5, "Ar": 0.5} if kind == "mixture_edge" else "Ar",
            1.0,
            plasma={"He": QuadraticRate(2e9, THRESHOLD), "Ar": rate}
            if kind == "mixture_edge"
            else rate,
            ionisation_field_range=(1e3, 1e11),
            preionisation=1e-2 if kind == "preionised_quiet" else None,
            lambda0=800e-9,
            grid=FieldGrid.from_limits(
                LENGTH, 800e-9, (200e-9, 2e-6), TRANGE, **LEGACY_GRID
            ),
            pulses=[pulse],
            n_r=8,
            n_save=2,
            plasma_integration="band",
            integrator=RungeKutta(),
            schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / steps),
            absorber_scheme="split_step",
            noise=None,
        )
        # Mode-averaged: one node, so a host team of several threads splits each row.
        simulation.settings = dataclasses.replace(
            simulation.settings, quadrature=None, components=None
        )
        problem = simulation.translate()
    grid = simulation.grid
    assert isinstance(grid, FieldGrid)
    return grid, dataclasses.replace(problem, stats_spec=StatsSpec(names=STATISTICS))


def vector(steps: int) -> tuple[FieldGrid, KernelProblem]:
    """Return two-component argon on the radial route, its pulse at the window's end."""
    length = 1e-4
    # Resolve the analytic collar derivative without moving the pulse or the
    # window: the leading-edge pulse still occupies the taper.
    grid = FieldGrid.from_limits(
        length,
        800e-9,
        (600e-9, 1.2e-6),
        150e-15,
        **{**LEGACY_GRID, "collar_fraction": 0.2},
    )
    space = RadialGrid(6e-4, 16, window_factor=0.0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7, t0=-70e-15),
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
        save_planes=[length],
        dz_max=length,
        dz_init=length / steps,
        schedule=FixedSchedule.uniform(0.0, length, length / steps),
        density=1e22,
    )
    problem = banded(problem, design_running_integral(grid, "band"))
    return grid, dataclasses.replace(problem, stats_spec=StatsSpec(names=STATISTICS))


def node_fields(problem: KernelProblem, grid: FieldGrid) -> np.ndarray:
    """Return the reference's node fields at the launch plane, (nodes, n_pol, n_to)."""
    reference = NumpyRefBackend(problem, grid=grid)
    try:
        reference.set_initial(problem.initial.field0, None)
        state = reference._require_state()  # pyright: ignore[reportPrivateUsage]
        return np.concatenate(
            list(reference._plasma_nodes(state[0]))  # pyright: ignore[reportPrivateUsage]
        )
    finally:
        reference.close()


def case(kind: str, steps: int) -> tuple[FieldGrid, KernelProblem]:
    """Return the stress case `kind` marching `steps` fixed steps."""
    if kind == "vector_edge":
        return vector(steps)
    if kind == "preionised_quiet":
        # Below threshold everywhere: about 1.7e9 V/m at the peak.
        return capillary(kind, steps, energy=2e-6)
    if kind != "near_threshold":
        return capillary(kind, steps)
    # Scale the energy so the peak field is 5 % above threshold: a few cycle peaks
    # ionise, and the exposure is tiny.
    grid, problem = capillary(kind, steps)
    peak = float(np.max(np.abs(node_fields(problem, grid))))
    return capillary(kind, steps, energy=10e-6 * (1.05 * THRESHOLD / peak) ** 2)


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


def launch_statistics(
    implementation: type[NumpyRefBackend] | type[NativeBackend],
    problem: KernelProblem,
    grid: FieldGrid,
) -> dict[str, np.ndarray]:
    """Return the statistics of the launch field, before any step."""
    backend = implementation(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        return backend.stats(problem.stats_spec)
    finally:
        backend.close()


def assert_is_stressing(kind: str, problem: KernelProblem, grid: FieldGrid) -> None:
    """The case exercises what it is named for, so a pass is not vacuous."""
    fields = node_fields(problem, grid)
    peak = float(np.max(np.abs(fields)))
    half = problem.responses.plasma_integral.half_width
    if kind == "preionised_quiet":
        assert peak < THRESHOLD  # the rate is identically zero
    elif kind == "near_threshold":
        assert THRESHOLD < peak < 1.1 * THRESHOLD
    elif kind in {"leading_edge", "trailing_edge", "mixture_edge", "vector_edge"}:
        # Within K samples of an end the field is a sizeable part of its peak, so the
        # live windows reach the end and the point reflections carry the pulse.
        ends = np.abs(fields[..., : half + 1]), np.abs(fields[..., -half - 1 :])
        assert max(float(np.max(end)) for end in ends) > 1e-3 * peak


@pytest.mark.parametrize("kind", CASES)
def test_one_evaluation_matches_the_reference(kind: str) -> None:
    grid, problem = case(kind, 1)
    assert_is_stressing(kind, problem, grid)
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
    error = float(np.max(np.abs(actual - expected)))
    assert error <= evaluation_bound(problem, grid) * scale
    assert_regression_guard(actual, expected, f"stress {kind} evaluation")


@pytest.mark.parametrize("steps", [1, 10])
@pytest.mark.parametrize("kind", CASES)
def test_fixed_steps_match_the_reference(kind: str, steps: int) -> None:
    grid, problem = case(kind, steps)
    expected, expected_stats = march(NumpyRefBackend, problem, grid)
    actual, actual_stats = march(NativeBackend, problem, grid)
    bound = march_bound(problem, grid, steps)
    scale = float(np.max(np.abs(expected)))
    assert float(np.max(np.abs(actual - expected))) <= bound * scale
    assert_regression_guard(actual, expected, f"stress {kind} {steps} steps")
    assert_statistics_agree(actual_stats, expected_stats, bound)


@pytest.mark.parametrize("kind", CASES)
def test_the_launch_exposure_is_within_its_scan_bound(kind: str) -> None:
    """The final fraction from one field: the exposure scan's absolute bound.

    :mod:`plasma_bounds` items 1 to 4: the rate's spread over each side's synthesised
    field, the exposure scan's ``2 gamma_D M[W] + (1 + gamma_D) M[dW]``, the fraction's
    ``(1 - p) e^{-X} dX`` and each side's roundings on the actual survival and fraction,
    ``2 u (4 s + f)``. The electron density is the neutral density times the largest
    final fraction (a mixture's partial densities sum to the total), and a maximum moves
    by at most the largest of its arguments' moves.
    """
    grid, problem = case(kind, 1)
    responses = problem.responses
    rule = responses.plasma_integral
    assert rule.policy == "band"
    dt = grid.dt_fine
    reference = NumpyRefBackend(problem, grid=grid)
    try:
        reference.set_initial(problem.initial.field0, None)
        reference._hoist(0.0)  # pyright: ignore[reportPrivateUsage] - oracle probe
        state = reference._require_state()  # pyright: ignore[reportPrivateUsage]
        bundle = reference._stage_scalars(0.0, 0.0)  # pyright: ignore[reportPrivateUsage]
        fields = np.real(
            np.concatenate(
                list(reference._plasma_nodes(state[0]))  # pyright: ignore[reportPrivateUsage]
            )
        )
        field_error = node_field_error(reference, problem, fields)
    finally:
        reference.close()
    channels = responses.plasma_channels
    if len(channels) > 1:
        groups = [
            (channel.rate, initial)
            for channel, initial in zip(
                channels, bundle.plasma_preionisation, strict=True
            )
        ]
    else:
        assert responses.plasma_coefficients is not None
        groups = [(responses.plasma_coefficients.rate, float(bundle.preionisation))]
    magnitude = np.sqrt(np.sum(fields**2, axis=1))
    delta = np.sum(field_error, axis=1)
    worst = 0.0
    for table, initial in groups:
        rate, spread = rate_spread(table, magnitude, delta)
        exposure = rule.floor(rule.integrate(rate, dt))[:, -1]
        exposure_error = scan_bound(
            rule.integrate(rate, dt), rate, spread, rule, dt, INFO.runtime_threads
        )[:, -1]
        survival = (1.0 - initial) * np.exp(-exposure)
        fraction = 1.0 - survival
        fraction_error = (1.0 - initial) * np.exp(
            -np.maximum(exposure - exposure_error, 0.0)
        ) * exposure_error + 2 * UNIT_ROUNDOFF * (4 * survival + fraction)
        worst = max(worst, float(np.max(fraction_error)))
    expected = launch_statistics(NumpyRefBackend, problem, grid)
    actual = launch_statistics(NativeBackend, problem, grid)
    neutral = float(responses.density(0.0))
    density = float(expected["electron_density"][0])
    error = abs(float(actual["electron_density"][0]) - density)
    assert error <= neutral * worst, (error, neutral * worst)
    if kind != "preionised_quiet":
        assert density > 0.0
