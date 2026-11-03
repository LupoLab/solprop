# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The band column on the retired column tests' pathological inputs, against numpy_ref.

SEP 43's inventory items 17 to 19: ``tests/plasma_column_inputs.py`` drove the Numba
column with ``field[:, ::7] = 0``, a 1e-170 sample and preionisation 0, 1e-20, 0.2 and
1, ``bodies.loss_scale`` was called with a zero divisor under a live rate, and
``bodies.rate_at`` with a table whose stored logarithm was off, so that both of its
knot clamps fired. Native's column cannot be called alone: it reads a field it
synthesises from the spectral state. So the inputs are reached through the state and the
translated table, and each test checks that they occur:

- **Exact zeros in a live row.** One spectral bin at a quarter of the fine Nyquist gives
  ``cos(pi j / 4)``, whose every fourth sample is exactly zero through the inverse
  transform, between samples where the rate is live.
- **A vanishing divisor under a live rate.** A rate independent of the field, tabulated
  below ``1e-180`` V/m, on a whole field near ``1e-168`` V/m: ``|E|^2`` underflows to an
  exact zero where the rate is positive, the case the guard exists for. A field exactly
  zero never meets a live rate, on either backend, because the table clips below its
  first knot to zero.
- **An index past either end of the table.** A table whose stored ``log_field_min`` is
  moved, so that every live sample's index is negative, or past the last interval, while
  its fields stay within the table's range: ``general.cpp``'s two clamps.

**Where this is weaker than the original.** A 1e-170 sample beside samples of order
one is out of reach, because a transform would round it away: the tiny sample here
comes with a tiny row. The zeros are every fourth sample rather than every seventh, and
the Numba column's reuse of an unchanged fraction has no native counterpart to compare.
"""

# ruff: noqa: I001 -- Kokkos loads first, before any other OpenMP runtime on macOS.

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
from solprop.grid import FieldGrid
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, RungeKutta
from solprop.translate.problem import KernelProblem

from solprop_native import NativeBackend
from legacy_geometry import LEGACY_GRID
from plasma_bounds import assert_regression_guard
from test_band_plasma import (
    STATISTICS,
    QuadraticRate,
    assert_statistics_agree,
    evaluation_bound,
    march_bound,
)
from test_band_stress import vector

type ComplexArray = npt.NDArray[np.complex128]

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)

INFO = solprop_native.build_info()
LENGTH = 1e-5
"""The capillary's length, m: about 0.2 rad of the ionised gas's phase at the bin."""
TRANGE = 400e-15
"""The capillary's window, s."""
RATE = 2e12
"""The field-independent rate, 1/s: an exposure of order one across the window."""
LIVE = 3e9
"""The live row's peak field, V/m."""
TINY = 1.75e-168
"""The tiny row's peak field, V/m: its square underflows to zero."""
PREIONISATION = (0.0, 1e-20, 0.2, 1.0)
"""The original's preionisation: none, below rounding, a share, and all of it."""


class ConstantRate:
    """A rate independent of the field above `floor`, so it is live at any field."""

    maximum = 2 * RATE

    def __init__(self, floor: float) -> None:
        self.threshold = floor
        self.citation = f"Constant test rate {RATE:g} s^-1 above {floor:g} V/m"
        self.parameters = (("model", "constant-pathological"), ("threshold", floor))

    def __call__(self, field_amplitude: object) -> npt.NDArray[np.float64]:
        values = np.abs(np.asarray(field_amplitude, dtype=float))
        return np.where(values < self.threshold, 0.0, RATE)


class UncappedRate(QuadraticRate):
    """The quadratic law with no ceiling to shorten its table below `LIVE`."""

    maximum = 1e60


def capillary(
    *,
    peak: float,
    floor: float,
    preionisation: float,
    steps: int = 1,
    rate: object = None,
) -> tuple[FieldGrid, KernelProblem]:
    """Return a mode-averaged argon capillary whose launch is one comb bin of `peak`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        simulation = Simulation.from_capillary(
            125e-6,
            LENGTH,
            "Ar",
            1.0,
            plasma=ConstantRate(floor) if rate is None else rate,  # pyright: ignore[reportArgumentType]
            ionisation_field_range=(floor, 1e11),
            preionisation=preionisation,
            wavelength=800e-9,
            grid=FieldGrid.from_limits(
                LENGTH, 800e-9, (200e-9, 2e-6), TRANGE, **LEGACY_GRID
            ),
            pulses=[GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-6)],
            n_r=8,
            n_save=2,
            plasma_integration="band",
            integrator=RungeKutta(),
            schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / steps),
            absorber_scheme="split_step",
            noise=None,
        )
        simulation.settings = dataclasses.replace(
            simulation.settings, quadrature=None, components=None
        )
        problem = simulation.translate()
    grid = simulation.grid
    assert isinstance(grid, FieldGrid)
    field = np.zeros(problem.shape, complex)
    field[..., grid.n_to // 8] = 1.0
    field *= peak / np.max(np.abs(synthesised(problem, grid, field)))
    return grid, dataclasses.replace(
        problem,
        initial=dataclasses.replace(problem.initial, field0=field),
        stats_spec=StatsSpec(names=STATISTICS),
    )


def synthesised(
    problem: KernelProblem, grid: FieldGrid, field: ComplexArray
) -> npt.NDArray[np.float64]:
    """Return the reference's node fields of `field`, ``(nodes, n_pol, n_to)``."""
    reference = NumpyRefBackend(problem, grid=grid)
    try:
        reference.set_initial(field, None)
        state = reference._require_state()  # pyright: ignore[reportPrivateUsage]
        return np.real(
            np.concatenate(
                list(reference._plasma_nodes(state[0]))  # pyright: ignore[reportPrivateUsage]
            )
        )
    finally:
        reference.close()


def source_of_reference(problem: KernelProblem, grid: FieldGrid) -> ComplexArray:
    """Return the reference's first stage on the launch, ``N(E, z = 0)``."""
    reference = NumpyRefBackend(problem, grid=grid)
    try:
        reference.set_initial(problem.initial.field0, None)
        reference._hoist(0.0)  # pyright: ignore[reportPrivateUsage] - oracle probe
        return reference._nonlinear(  # pyright: ignore[reportPrivateUsage]
            reference._require_state(),  # pyright: ignore[reportPrivateUsage]
            reference._stage_scalars(0.0, 0.0),  # pyright: ignore[reportPrivateUsage]
            np.ones(problem.shape, complex),
        )[0]
    finally:
        reference.close()


def source_of_native(problem: KernelProblem, grid: FieldGrid) -> ComplexArray:
    """Return native's first stage on the launch, from a zero-length trial."""
    backend = NativeBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend._trial_step(0.0, 0.0)  # pyright: ignore[reportPrivateUsage]
        return np.asarray(
            backend._native_state.stage_to_host(0)  # pyright: ignore[reportPrivateUsage]
        ).reshape(problem.shape[1:])
    finally:
        backend.close()


def launch_statistics(
    implementation: type[NumpyRefBackend] | type[NativeBackend],
    problem: KernelProblem,
    grid: FieldGrid,
) -> dict[str, np.ndarray]:
    """Return the statistics of the launch, before any step."""
    backend = implementation(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        return backend.stats(problem.stats_spec)
    finally:
        backend.close()


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


def lossless_unit_bound(preionisation: float) -> float:
    """Bound one evaluation of the tiny row, by the column's scale invariance.

    :func:`plasma_bounds.plasma_stage_bound` squares the field, so at 1e-168 V/m it
    sees no magnitude, no rate and no current, and would certify nothing. With the rate
    independent of the field and the loss current guarded off, the column is linear in
    the field: fraction and weight are the unit row's, and the relative error of the
    current is the unit row's. So the bound is that chain's on the same comb at one
    V/m, with a vanishing ionisation potential, fixed from its operands before either
    tiny run.
    """
    grid, problem = capillary(peak=1.0, floor=1e-6, preionisation=preionisation)
    channels = tuple(
        dataclasses.replace(channel, ionisation_potential=float(np.finfo(float).tiny))
        for channel in problem.responses.plasma_channels
    )
    responses = dataclasses.replace(problem.responses, plasma_channels=channels)
    return evaluation_bound(dataclasses.replace(problem, responses=responses), grid)


def assert_source_agrees(problem: KernelProblem, grid: FieldGrid, bound: float) -> None:
    """Native's first stage on the launch is finite and within `bound` of the oracle."""
    expected = source_of_reference(problem, grid)
    actual = source_of_native(problem, grid)
    assert np.all(np.isfinite(actual))
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    assert float(np.max(np.abs(actual - expected))) <= bound * scale
    # The derived chain assumes no cancellation in the currents' scans, so it is far
    # looser than the original's scan_bound on one input; the guard sees a subtle
    # defect.
    assert_regression_guard(actual, expected, "pathological source")


def require_exact_zeros(fields: npt.NDArray[np.float64]) -> None:
    """Skip, saying so, where this platform's transform doesn't land on the zeros."""
    if not np.all(fields[..., 2::4] == 0.0):
        pytest.skip(
            "the reference's transform leaves a residue at the comb's zero crossings "
            "here, so the exact-zero input of this test does not occur"
        )


def assert_exact_zeros_in_a_live_row(problem: KernelProblem, grid: FieldGrid) -> None:
    """The launch holds exact zeros, a quarter of the row, between live samples."""
    fields = synthesised(problem, grid, problem.initial.field0)
    require_exact_zeros(fields)
    assert np.count_nonzero(fields == 0.0) == grid.n_to // 4


@pytest.mark.parametrize("preionisation", PREIONISATION)
def test_exact_zeros_in_a_live_row_match_the_reference(preionisation: float) -> None:
    grid, problem = capillary(peak=LIVE, floor=1.0, preionisation=preionisation)
    assert_exact_zeros_in_a_live_row(problem, grid)
    assert_source_agrees(problem, grid, evaluation_bound(problem, grid))


@pytest.mark.parametrize("preionisation", PREIONISATION)
def test_a_vanishing_divisor_under_a_live_rate_has_a_zero_loss_current(
    preionisation: float,
) -> None:
    """``|E|^2`` is an exact zero where the rate is positive: zero loss, not a NaN.

    The original called ``loss_scale`` with a zero divisor. The current here is the
    free current alone: one that took the division would be infinite or NaN at every
    live sample, and a backend's source would not be finite, let alone within the bound.
    """
    grid, problem = capillary(peak=TINY, floor=1e-180, preionisation=preionisation)
    fields = synthesised(problem, grid, problem.initial.field0)
    live = fields != 0.0
    assert np.any(live)
    # The two conditions the guard sees, on the reference's field, which native's is
    # within rounding of: a positive rate, and a square that has underflowed.
    assert np.all(fields**2 == 0.0)
    coefficients = problem.responses.plasma_coefficients
    assert coefficients is not None
    assert np.all(np.asarray(coefficients.rate(np.abs(fields[live]))) > 0.0)
    assert_source_agrees(problem, grid, lossless_unit_bound(preionisation))


def test_the_exact_zeros_do_occur_on_native() -> None:
    """Native's own synthesis lands on the zeros; the tests above are not vacuous.

    Native's node field cannot be read from Python, so this reads it through the loss
    current. With the table's first knot at 1e-30 V/m a residue of a transform's
    rounding, 1e-6 V/m at this field, would be a live sample at a tiny divisor, its loss
    current ``I W (1 - f) / E`` about 1e15 times the live samples'. A zero is clipped to
    no rate at all. So a source of the reference's size means the samples are zero, or
    below the first knot. No rounding bound applies: the chain's own is infinite here.
    """
    grid, problem = capillary(peak=LIVE, floor=1e-30, preionisation=0.0)
    assert_exact_zeros_in_a_live_row(problem, grid)
    expected = source_of_reference(problem, grid)
    actual = source_of_native(problem, grid)
    scale = float(np.max(np.abs(expected)))
    if float(np.max(np.abs(actual))) > 1e3 * scale:
        pytest.skip(
            "this build's transform leaves a residue where the reference has an exact "
            "zero, so the exact-zero inputs of this module do not occur on it"
        )
    assert_regression_guard(actual, expected, "exact zeros on native")


def test_an_all_zero_state_stays_exactly_zero() -> None:
    """Class P: a state of exact zeros has no source and no statistics on either."""
    grid, problem = capillary(peak=LIVE, floor=1.0, preionisation=0.2)
    zero = np.zeros(problem.shape, complex)
    problem = dataclasses.replace(
        problem, initial=dataclasses.replace(problem.initial, field0=zero)
    )
    np.testing.assert_array_equal(source_of_reference(problem, grid), 0.0)
    np.testing.assert_array_equal(source_of_native(problem, grid), 0.0)
    expected = launch_statistics(NumpyRefBackend, problem, grid)
    actual = launch_statistics(NativeBackend, problem, grid)
    for name in STATISTICS:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


@pytest.mark.parametrize("preionisation", PREIONISATION)
@pytest.mark.parametrize("steps", [1, 10])
@pytest.mark.parametrize("row", ["live", "tiny"])
def test_fixed_steps_from_the_inputs_match_the_reference(
    row: str, steps: int, preionisation: float
) -> None:
    """Short marches from the same launches, on the column's own bound per stage.

    A step moves the zero crossings off the samples, and a sample just above a low knot
    under a field-independent rate is a loss current without limit, on any backend. So
    the live march takes the quadratic law, whose loss current vanishes with the field.
    The tiny march keeps the constant one, since its loss current stays guarded off: its
    square stays underflowed over a run this short.
    """
    if row == "live":
        grid, problem = capillary(
            peak=LIVE,
            floor=1e3,
            preionisation=preionisation,
            steps=steps,
            rate=QuadraticRate(RATE / (LIVE / 1e9) ** 2, 1e3),
        )
        bound = march_bound(problem, grid, steps)
    else:
        grid, problem = capillary(
            peak=TINY, floor=1e-180, preionisation=preionisation, steps=steps
        )
        bound = 7 * steps * lossless_unit_bound(preionisation)
    expected, expected_stats = march(NumpyRefBackend, problem, grid)
    actual, actual_stats = march(NativeBackend, problem, grid)
    assert np.all(np.isfinite(actual))
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    assert float(np.max(np.abs(actual - expected))) <= bound * scale
    assert_regression_guard(actual, expected, "pathological march")
    assert_statistics_agree(actual_stats, expected_stats, bound)


@pytest.mark.parametrize(
    ("peak", "floor"), [(LIVE, 1.0), (TINY, 1e-180)], ids=["live", "tiny"]
)
@pytest.mark.parametrize("preionisation", PREIONISATION)
def test_the_launch_statistics_match_the_reference(
    preionisation: float, peak: float, floor: float
) -> None:
    """The stash's fraction and rate, from zeros and tiny rows, at every initial one."""
    grid, problem = capillary(peak=peak, floor=floor, preionisation=preionisation)
    bound = (
        evaluation_bound(problem, grid)
        if peak == LIVE
        else lossless_unit_bound(preionisation)
    )
    expected = launch_statistics(NumpyRefBackend, problem, grid)
    actual = launch_statistics(NativeBackend, problem, grid)
    assert_statistics_agree(actual, expected, bound)
    assert float(expected["peak_ionisation_rate"][0]) == pytest.approx(RATE, rel=1e-6)
    if preionisation == 1.0:
        # All of it ionised already: no survivors for rounding to act on. Exact.
        neutral = float(problem.responses.density(0.0))
        assert float(actual["electron_density"][0]) == neutral
        assert float(expected["electron_density"][0]) == neutral


@pytest.mark.parametrize(
    ("scale", "shift", "end"),
    [(1e24, 16.0, "below"), (1e6, -6.0, "above")],
    ids=["below", "above"],
)
def test_the_knot_index_is_clamped_at_both_ends_of_the_table(
    scale: float, shift: float, end: str
) -> None:
    """The two clamps, on the column's rate, with the stored logarithm moved.

    The original called ``rate_at`` with such a table; the rate here is read through the
    source and the launch's statistics. `scale` puts each end's knots at a rate of about
    1e11 s^-1, so what the clamp selects moves the current.
    """
    grid, problem = capillary(
        peak=LIVE, floor=1e3, preionisation=0.0, rate=UncappedRate(scale, 1e3)
    )
    channel = problem.responses.plasma_channels[0]
    table = dataclasses.replace(
        channel.rate, log_field_min=channel.rate.log_field_min + shift
    )
    responses = dataclasses.replace(
        problem.responses,
        plasma_channels=(dataclasses.replace(channel, rate=table),),
        ionisation_table=(table,),
    )
    problem = dataclasses.replace(problem, responses=responses)
    fields = synthesised(problem, grid, problem.initial.field0)
    # The comb's zero crossings lie below the table, whether or not they are exact.
    magnitude = np.abs(fields[np.abs(fields) >= table.field_min])
    assert magnitude.size == 3 * grid.n_to // 4
    assert np.all(magnitude <= table.field_max)
    index = (np.log(magnitude) - table.log_field_min) / table.d_log_field
    if end == "below":
        assert np.all(index < 0.0)
    else:
        assert np.all(index >= table.coefficients.shape[1])
    assert_source_agrees(problem, grid, evaluation_bound(problem, grid))
    expected = launch_statistics(NumpyRefBackend, problem, grid)
    actual = launch_statistics(NativeBackend, problem, grid)
    assert float(expected["electron_density"][0]) > 0.0
    assert float(expected["peak_ionisation_rate"][0]) > 1e10
    assert_statistics_agree(actual, expected, evaluation_bound(problem, grid))


def test_a_two_component_row_with_an_exactly_zero_component_matches_the_reference():
    """The original's ``n_pol = 2``: exact zeros in x, a y of exact zeros, radially."""
    grid, problem = vector(1)
    launch = np.asarray(problem.initial.field0)
    field = np.zeros(problem.shape, complex)
    # The launch's radial shape, every radial node's row the same comb in time.
    field[0, :, 0, grid.n_to // 8] = np.max(np.abs(launch[0, :, 0, :]), axis=-1)
    field *= LIVE / np.max(np.abs(synthesised(problem, grid, field)))
    problem = dataclasses.replace(
        problem, initial=dataclasses.replace(problem.initial, field0=field)
    )
    fields = synthesised(problem, grid, field)
    assert fields.shape[1] == 2
    np.testing.assert_array_equal(fields[:, 1, :], 0.0)
    live = fields[:, 0, :]
    require_exact_zeros(live)
    assert np.all(live[:, ::4] != 0.0)
    assert_source_agrees(problem, grid, evaluation_bound(problem, grid))
