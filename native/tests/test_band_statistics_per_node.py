# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The band statistics of one node and of each channel, against numpy_ref (SEP 43).

Inventory item 25. The retired Numba tests held the statistics column's per-node
fraction, peak rate and diagnostics, and the column's per-channel fractions, to the
band column's budget ``64 (2K + 1) n_to eps``, about 1e-10 relative. What survived was
plane-level, at ``1e3`` times that for the rates and absolutely for the diagnostics.

**How it is reached.** Native reports no node's statistics, only the largest over a
plane, so each node of the original (a pulse of 6e10, 9e10 or 1.2e10 V/m, 4 fs, on the
same grid and argon rate) is its own one-node mode-averaged problem, where the largest
is that node. **What is added to the original's budget.** The Numba column took the
same real array as the reference. Native synthesises its own, so each bound is the
budget plus what one synthesis's rounding moves that statistic by, from
:mod:`plasma_bounds`'s operands, fixed before either side runs. **Weaker than the
original:** only one component, since a mode-averaged node has one, and each of the
diagnostics is the ratio of two parts, since native returns no parts.
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
from solprop.nonlinear.ionisation import RunningIntegral
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, RungeKutta
from solprop.translate.plasma_integration import design_running_integral
from solprop.translate.ionisation import RateTable
from solprop.translate.problem import KernelProblem

from solprop_native import NativeBackend
from bounds import UNIT_ROUNDOFF
from legacy_geometry import LEGACY_GRID
from plasma_bounds import node_field_error, rate_spread, scan_bound
from test_band_plasma import STATISTICS
from test_band_stress import capillary

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)

INFO = solprop_native.build_info()
LENGTH = 1e-5
"""The capillary's length, m; nothing here steps."""
GRID = FieldGrid.from_limits(
    LENGTH, 800e-9, (200e-9, 2e-6), 60e-15, ffac=6.0, **LEGACY_GRID
)
"""The original's grid: a short default-oversampled window, K = 4."""
AMPLITUDES = (6e10, 9e10, 1.2e10)
"""The original's three nodes, V/m: two ionising and one weak."""
PREIONISATION = 0.2
"""The original's initial fraction."""


def rules() -> dict[str, RunningIntegral]:
    """The band rule of the grid, and the band rule at K = 0."""
    return {
        "K=4": design_running_integral(GRID, "band"),
        "K=0": RunningIntegral(
            "band", np.ones(1), dt=GRID.dt_fine, band_edge=GRID.omega_max_window
        ),
    }


def band_budget(rule: RunningIntegral) -> float:
    """The original's bound: 64 eps per fine sample per tap of the band filter."""
    width = 2 * rule.half_width + 1
    return 64 * width * GRID.n_to * float(np.finfo(float).eps)


def launch(amplitude: float) -> npt.NDArray[np.float64]:
    """The original's fine field: a 4 fs Gaussian of the carrier 2.35e15 rad/s."""
    t = (np.arange(GRID.n_to) - GRID.n_to // 2) * GRID.dt_fine
    return amplitude * np.exp(-((t / 4e-15) ** 2)) * np.cos(2.35e15 * t)


def problem_for(amplitude: float, rule: RunningIntegral) -> KernelProblem:
    """Return a one-node argon problem whose launch synthesises to :func:`launch`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        simulation = Simulation.from_capillary(
            125e-6,
            LENGTH,
            "Ar",
            1.0,
            plasma=OptmatRate("Ar", 800e-9),
            ionisation_field_range=(1e8, 2e11),
            preionisation=PREIONISATION,
            wavelength=800e-9,
            grid=GRID,
            pulses=[GaussianPulse(wavelength=800e-9, fwhm=5e-15, energy=1e-9)],
            n_r=8,
            n_save=2,
            plasma_integration="band",
            integrator=RungeKutta(),
            schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH),
            absorber_scheme="split_step",
            noise=None,
        )
        simulation.settings = dataclasses.replace(
            simulation.settings, quadrature=None, components=None
        )
        problem = simulation.translate()
    spectrum = np.fft.rfft(launch(amplitude))[np.asarray(GRID.crop_index)]
    field = spectrum.reshape(problem.shape)
    unit = fields_of(problem, field)
    field = field * (amplitude / np.max(np.abs(unit)))
    return dataclasses.replace(
        problem,
        initial=dataclasses.replace(problem.initial, field0=field),
        responses=dataclasses.replace(problem.responses, plasma_integral=rule),
        stats_spec=StatsSpec(names=STATISTICS),
    )


def fields_of(problem: KernelProblem, field: npt.NDArray[np.complex128]) -> np.ndarray:
    """Return the reference's node fields for the state `field`."""
    reference = NumpyRefBackend(problem, grid=GRID)
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


def statistics(
    implementation: type[NumpyRefBackend] | type[NativeBackend],
    problem: KernelProblem,
) -> dict[str, np.ndarray]:
    """Return the launch's statistics."""
    backend = implementation(problem, grid=GRID)
    try:
        backend.set_initial(problem.initial.field0, None)
        return backend.stats(problem.stats_spec)
    finally:
        backend.close()


def node_fields_and_error(
    problem: KernelProblem, grid: FieldGrid
) -> tuple[np.ndarray, np.ndarray]:
    """Return the reference's launch node fields and their error between syntheses."""
    reference = NumpyRefBackend(problem, grid=grid)
    try:
        reference.set_initial(problem.initial.field0, None)
        state = reference._require_state()  # pyright: ignore[reportPrivateUsage]
        reference._hoist(0.0)  # pyright: ignore[reportPrivateUsage] - oracle probe
        fields = np.real(
            np.concatenate(
                list(reference._plasma_nodes(state[0]))  # pyright: ignore[reportPrivateUsage]
            )
        )
        return fields, node_field_error(reference, problem, fields)
    finally:
        reference.close()


def fraction_terms(
    table: RateTable,
    initial: float,
    rule: RunningIntegral,
    dt: float,
    magnitude: np.ndarray,
    delta: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return what the synthesis's error moves one channel's rate and fraction by.

    The rate's spread over ``|E| +- delta`` (:func:`plasma_bounds.rate_spread`), and the
    final fraction's bound of ``test_band_stress`` through the exposure's scan. Returns
    the largest rate, its spread, the largest final fraction and its error.
    """
    rate, spread = rate_spread(table, magnitude, delta)
    integral = rule.integrate(rate, dt)
    exposure = rule.floor(integral)[:, -1]
    exposure_error = scan_bound(integral, rate, spread, rule, dt, INFO.runtime_threads)[
        :, -1
    ]
    survival = (1.0 - initial) * np.exp(-exposure)
    fraction = 1.0 - survival
    fraction_error = (1.0 - initial) * np.exp(
        -np.maximum(exposure - exposure_error, 0.0)
    ) * exposure_error + 2 * UNIT_ROUNDOFF * (4 * survival + fraction)
    return (
        float(np.max(rate)),
        float(np.max(spread)),
        float(np.max(fraction)),
        float(np.max(fraction_error)),
    )


@pytest.mark.parametrize("name", ["K=4", "K=0"])
@pytest.mark.parametrize("amplitude", AMPLITUDES)
def test_the_node_s_statistics_are_the_reference_s(name: str, amplitude: float) -> None:
    """Fraction, largest rate and diagnostics of one node, at the original's budget."""
    rule = rules()[name]
    problem = problem_for(amplitude, rule)
    budget = band_budget(rule)
    fields, field_error = node_fields_and_error(problem, GRID)
    magnitude = np.sqrt(np.sum(fields**2, axis=1))
    delta = np.sum(field_error, axis=1)
    coefficients = problem.responses.plasma_coefficients
    assert coefficients is not None
    peak_rate, spread, _, final_error = fraction_terms(
        coefficients.rate, PREIONISATION, rule, GRID.dt_fine, magnitude, delta
    )
    expected = statistics(NumpyRefBackend, problem)
    actual = statistics(NativeBackend, problem)
    # The fraction and the largest rate: the original's budget, plus the synthesis.
    neutral = float(problem.responses.density(0.0))
    fraction = float(expected["electron_density"]) / neutral
    assert fraction > PREIONISATION
    assert (
        abs(float(actual["electron_density"]) / neutral - fraction)
        <= budget * fraction + final_error
    )
    rate = float(expected["peak_ionisation_rate"])
    assert rate == pytest.approx(peak_rate, rel=budget)
    assert abs(float(actual["peak_ionisation_rate"]) - rate) <= (budget * rate + spread)
    # The diagnostics are ratios of two parts. The undershoot's parts are differences of
    # a cumulative sum on one side, so 1e-9 each in the original; the exposure's steps
    # also move by the rate's spread, relative to the rise. At K = 0 the steps of a
    # trapezoid of a positive rate never dip, on either side: an exact zero.
    undershoot = float(expected["plasma_undershoot"])
    if name == "K=0":
        assert undershoot == 0.0
        assert float(actual["plasma_undershoot"]) == 0.0
    else:
        assert undershoot > 0.0
        assert abs(float(actual["plasma_undershoot"]) - undershoot) <= (
            2e-9 * undershoot + 2 * spread / peak_rate
        )
    # The work defect's parts sum in another order: 1e-3 relative or 1e-15 absolute of
    # the defect in the original, over the freed share, which moves by the fraction's
    # error on top.
    freed = fraction - PREIONISATION
    defect = float(expected["plasma_work_defect"])
    assert defect > 0.0
    assert abs(float(actual["plasma_work_defect"]) - defect) <= (
        (1e-3 + 2 * budget) * defect + (1e-15 + 2 * final_error) / freed
    )
    # The boundary level is the fraction times the field at the window's first samples,
    # against the largest of that, formed by one product each. Those first samples are
    # the pulse's tail, so the field's own absolute error is what moves it: ``delta``
    # over the peak. At K = 0 it reads no samples at all, an exact zero.
    boundary = float(expected["plasma_boundary_level"])
    if name == "K=0":
        assert boundary == 0.0
        assert float(actual["plasma_boundary_level"]) == 0.0
    else:
        assert boundary > 0.0
        assert abs(float(actual["plasma_boundary_level"]) - boundary) <= (
            2 * budget * boundary + 2 * float(np.max(delta) / np.max(magnitude))
        )


def test_each_channel_s_fraction_is_the_reference_s() -> None:
    """``species_ionisation``, channel by channel, at the original's budget.

    The original read each channel's final fraction from the column's stash, with one
    channel starting at exactly no ionisation (0 and 0.05), on three nodes of two
    components. This is the one node of the stress set's helium-argon mixture, with a
    pulse cut by the window, a rate the quadratic law's and starts of 0 and 0.01: one
    component and a different rate, which is weaker than the original on those counts.
    """
    grid, problem = capillary("mixture_edge", 1)
    # The helium channel starts from exactly none; the argon from the stress case's 1 %.
    _, preionised = capillary("preionised_quiet", 1)
    assert preionised.responses.plasma_coefficients is not None
    plain, argon = problem.responses.plasma_channels
    assert float(plain.preionisation(0.0)) == 0.0
    initial_argon = preionised.responses.plasma_coefficients.preionisation
    assert float(initial_argon(0.0)) == 0.01
    channels = (plain, dataclasses.replace(argon, preionisation=initial_argon))
    rule = problem.responses.plasma_integral
    problem = dataclasses.replace(
        problem,
        responses=dataclasses.replace(problem.responses, plasma_channels=channels),
        stats_spec=StatsSpec(names=("species_ionisation",)),
    )
    budget = 64 * (2 * rule.half_width + 1) * grid.n_to * float(np.finfo(float).eps)
    fields, field_error = node_fields_and_error(problem, grid)
    magnitude = np.sqrt(np.sum(fields**2, axis=1))
    delta = np.sum(field_error, axis=1)

    def launch_ionisation(implementation: type) -> np.ndarray:
        backend = implementation(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            return np.asarray(backend.stats(problem.stats_spec)["species_ionisation"])
        finally:
            backend.close()

    expected = launch_ionisation(NumpyRefBackend)
    actual = launch_ionisation(NativeBackend)
    # One row per channel, a largest over this one node.
    assert expected.shape == actual.shape == (2, 1)
    assert expected[0, 0] > 0.0
    assert expected[1, 0] > 0.01
    for channel, initial, want, got in zip(
        channels, (0.0, 0.01), expected.ravel(), actual.ravel(), strict=True
    ):
        _, _, final, final_error = fraction_terms(
            channel.rate, initial, rule, grid.dt_fine, magnitude, delta
        )
        assert want == pytest.approx(final, rel=budget)
        assert abs(got - want) <= budget * want + final_error
