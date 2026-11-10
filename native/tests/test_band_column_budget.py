# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The plasma column on SEP 25's stress fields, against the reference.

SEP 25 section 4.6: the column's integrands are signed and cancel over carrier cycles.
``test_numba_band_bounds.py`` held every sample of the time-domain current to an
absolute budget on seven stress fields, two component counts and two rules, with a
real argon rate table. Its fields, its table, its two rules and its budget's structure
are kept here, with what could not be kept stated plainly:

**Weaker than the original.** Native exposes no time-domain current: the column runs
inside a stage, and the only readings are the stage (a spectrum) and the launch's
statistics. So:

* The current is held at the stage, through :mod:`plasma_bounds`'s chain, ``l1 -> linf``
  over the forward transform: one number relative to the stage's largest value, not one
  per sample. Inverting the transform does not recover the per-sample claim, as the
  transform's own normwise rounding then exceeds the per-sample budget. A cancellation
  error confined to a few samples passes here if it is small against the stage. The
  chain allows 2e-8 to 5e-6 on these inputs and native measures 2e-16 to 2e-10; the
  original's per-sample claim needs a test-only binding of the column.
* The two sides synthesise their own node fields from one state, so the chain also
  carries the fields' difference through the rate's slope; the original gave both
  sides one real array. The launch cannot carry frequencies outside the grid's band,
  so ``leading_edge`` and ``trailing_edge`` reach their ends band-limited.
* The final fraction is held absolutely, as in :mod:`test_band_stress`: the exposure
  scan's own bound, through the electron density at the launch. Fractions round to
  ``2^-53`` beside one, so the two backends usually agree there exactly (they did on all
  28 of these): it is a coarse witness of the exposure.

The stress inputs that exist nowhere else are ``preionised_oscillation`` (flat 1e9 V/m
with tiny nonzero rates), ``zero_crossings``, ``tiny_exposure``,
``long_cancelling_prefix`` and pre-ionised ``trailing_edge``; the rate is the real
``OptmatRate`` table. Three radial nodes make the node fields exactly the original's
three scales: the synthesis is square, and is inverted for the launch. The collar is
replaced by one, since the stress fields reach the window's ends and the original had
no collar in the column.

The regression guard of :mod:`plasma_bounds` is not asserted here: relative to the stage
of a nearly unionised case it measures a tiny current's own conditioning (2e-10 on
``near_threshold`` with the steep real rate), not a defect.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
from bounds import UNIT_ROUNDOFF, fft_bound
from legacy_geometry import LEGACY_GRID
from plasma_bounds import node_field_error, plasma_stage_bound, rate_spread, scan_bound
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.constants import C
from solprop.grid import FieldGrid, RadialGrid
from solprop.nonlinear.ionisation import RunningIntegral
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import PlasmaResponse
from solprop.physdata import (
    Component,
    IonisationData,
    Mixture,
    Species,
    TabulatedSusceptibility,
)
from solprop.profiles import Constant
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.ionisation import build_rate_table
from solprop.translate.plasma_integration import design_running_integral
from solprop.translate.problem import PropagationProblem, assemble
from test_band_stress import launch_statistics

import solprop_native
from solprop_native import NativeBackend

type Real = npt.NDArray[np.float64]
type Complex = npt.NDArray[np.complex128]

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)

INFO = solprop_native.build_info()
GRID = FieldGrid.from_limits(
    1.0, 800e-9, (200e-9, 2e-6), 120e-15, ffac=6.0, **LEGACY_GRID
)
SPACE = RadialGrid(6e-4, 3, collar_fraction=0.0)
DENSITY = 2e25
LENGTH = 1e-6
CASES = (
    ("preionised_oscillation", 0.2),
    ("zero_crossings", 0.0),
    ("tiny_exposure", 0.0),
    ("near_threshold", 0.0),
    ("long_cancelling_prefix", 0.05),
    ("leading_edge", 0.0),
    ("trailing_edge", 0.1),
)
STATISTICS = ("electron_density", "peak_ionisation_rate")


def rule(name: str) -> RunningIntegral:
    """Return the grid's band rule, or the band machinery with one tap."""
    if name == "K=0":
        return RunningIntegral(
            "band", np.ones(1), dt=GRID.dt_fine, band_edge=GRID.omega_max_window
        )
    return design_running_integral(GRID, "band")


def stress(name: str, n_pol: int) -> Real:
    """Return one of SEP 25's stress fields, three nodes, ``(3, n_pol, n_to)``."""
    n_to = GRID.n_to
    t = (np.arange(n_to) - n_to // 2) * GRID.dt_fine
    carrier = 2.35e15
    out = np.zeros((3, n_pol, n_to))
    for node, scale in enumerate((0.6, 1.0, 1.3)):
        if name == "preionised_oscillation":
            # Keep the original input: tiny but nonzero rates at some samples.
            envelope = 1e9 * scale * np.ones(n_to)
        elif name == "zero_crossings":
            envelope = 8e10 * scale * np.exp(-((t / 8e-15) ** 2))
        elif name == "tiny_exposure":
            envelope = 2.2e10 * scale * np.exp(-((t / 5e-15) ** 2))
        elif name == "near_threshold":
            envelope = 1.0e10 * scale * np.exp(-((t / 20e-15) ** 2))
        elif name == "long_cancelling_prefix":
            envelope = 7e10 * scale * np.exp(-((t / 40e-15) ** 4))
        elif name == "leading_edge":
            envelope = 9e10 * scale * np.exp(-(((t - t[0]) / 6e-15) ** 2))
        elif name == "trailing_edge":
            envelope = 9e10 * scale * np.exp(-(((t - t[-1]) / 6e-15) ** 2))
        else:
            raise ValueError(name)
        # Floored far below anything physical, so no product reaches gradual
        # underflow, which needs an absolute term of its own (SEP 25 section 4.1).
        envelope = np.maximum(envelope, 1e-20 * envelope.max())
        out[node, 0] = envelope * np.cos(carrier * t + 0.3 * node)
        if n_pol == 2:
            out[node, 1] = 0.5 * envelope * np.sin(carrier * t + 0.3 * node)
    return out


def launched(
    problem: PropagationProblem, real: Real, integral: RunningIntegral
) -> PropagationProblem:
    """Return `problem` launching `real`, band-limited, under `integral`, no collar."""
    grids = problem.grids
    # The state's spectrum is doubled on its way to the real field.
    spectrum = 2.0 * np.fft.fft(real, axis=-1)[..., grids.crop_index] / grids.pad_scale
    inverse = np.linalg.inv(problem.transform.synthesis)
    state = np.einsum("mn,npw->mpw", inverse, spectrum)
    state[..., ~grids.active_mask] = 0.0
    n_to = grids.n_to
    return dataclasses.replace(
        problem,
        grids=dataclasses.replace(
            grids,
            t_window_fine=np.ones(n_to),
            t_window_fine_derivative=np.zeros(n_to),
        ),
        initial=dataclasses.replace(
            problem.initial, field0=np.ascontiguousarray(state)
        ),
        responses=dataclasses.replace(problem.responses, plasma_integral=integral),
        stats_spec=StatsSpec(names=STATISTICS),
    )


def assembled(
    responses: list[PlasmaResponse],
    n_pol: int,
    medium: SpatialMedium,
    density: float | None = None,
) -> PropagationProblem:
    """Return a one-step radial problem of three nodes, launching nothing yet."""
    return assemble(
        GRID,
        None,
        responses,
        np.zeros((3, n_pol, GRID.n_omega), dtype=complex),
        spatial_grid=SPACE,
        medium=medium,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[LENGTH],
        dz_max=LENGTH,
        dz_init=LENGTH,
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH),
        density=density,
    )


def argon(
    name: str, initial: float, n_pol: int, integral: RunningIntegral
) -> PropagationProblem:
    """Return a stress case: argon's real rate, its own pre-ionisation."""
    table = build_rate_table(OptmatRate("Ar", 800e-9), (1e8, 3e11))
    problem = assembled(
        [PlasmaResponse("Ar", table, preionisation=initial)],
        n_pol,
        SpatialMedium(),
        density=DENSITY,
    )
    return launched(problem, stress(name, n_pol), integral)


def stages(problem: PropagationProblem) -> tuple[Complex, Complex]:
    """Return native's and the reference's first stage on the launch."""
    reference = NumpyRefBackend(problem, grid=GRID)
    try:
        reference.set_initial(problem.initial.field0, None)
        reference._hoist(0.0)  # pyright: ignore[reportPrivateUsage] - oracle probe
        expected = reference._nonlinear(  # pyright: ignore[reportPrivateUsage]
            reference._require_state(),  # pyright: ignore[reportPrivateUsage]
            reference._stage_scalars(0.0, 0.0),  # pyright: ignore[reportPrivateUsage]
            np.ones((1, *problem.shape), complex),
        )[0]
    finally:
        reference.close()
    backend = NativeBackend(problem, grid=GRID)
    try:
        backend.set_initial(problem.initial.field0, None)
        # A zero-length trial exposes the first stage, N(E, z = 0).
        backend._trial_step(0.0, 0.0)  # pyright: ignore[reportPrivateUsage]
        actual = np.asarray(
            backend._native_state.stage_to_host(0)  # pyright: ignore[reportPrivateUsage]
        ).reshape(problem.shape)
    finally:
        backend.close()
    return actual, expected


def assert_stage_within_bound(
    problem: PropagationProblem, record: Callable[[str, float], None]
) -> None:
    bound = fft_bound(GRID.n_to) + plasma_stage_bound(
        problem, GRID, INFO.runtime_threads
    )
    assert np.isfinite(bound)
    assert 0.0 < bound < 1e-3
    actual, expected = stages(problem)
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    measured = float(np.max(np.abs(actual - expected)))
    record("measured", measured / scale)
    record("bound", bound)
    assert measured <= bound * scale


def launch_fraction_bound(problem: PropagationProblem) -> float:
    """Return the final fraction's absolute bound at the launch (``plasma_bounds``)."""
    responses = problem.responses
    integral = responses.plasma_integral
    dt = GRID.dt_fine
    reference = NumpyRefBackend(problem, grid=GRID)
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
    assert responses.plasma_coefficients is not None
    table = responses.plasma_coefficients.rate
    initial = float(bundle.preionisation)
    magnitude = np.sqrt(np.sum(fields**2, axis=1))
    delta = np.sum(field_error, axis=1)
    rate, spread = rate_spread(table, magnitude, delta)
    exposure = integral.floor(integral.integrate(rate, dt))[:, -1]
    exposure_error = scan_bound(
        integral.integrate(rate, dt), rate, spread, integral, dt, INFO.runtime_threads
    )[:, -1]
    survival = (1.0 - initial) * np.exp(-exposure)
    fraction = 1.0 - survival
    fraction_error = (1.0 - initial) * np.exp(
        -np.maximum(exposure - exposure_error, 0.0)
    ) * exposure_error + 2 * UNIT_ROUNDOFF * (4 * survival + fraction)
    return float(np.max(fraction_error))


@pytest.mark.parametrize(("name", "initial"), CASES)
@pytest.mark.parametrize("n_pol", [1, 2])
@pytest.mark.parametrize("name_of_rule", ["K=4", "K=0"])
def test_the_column_is_within_its_stage_budget(
    name: str,
    initial: float,
    n_pol: int,
    name_of_rule: str,
    record_property: Callable[[str, float], None],
) -> None:
    """Weaker than per sample, as the module says: at the stage, one relative number."""
    problem = argon(name, initial, n_pol, rule(name_of_rule))
    assert_stage_within_bound(problem, record_property)


@pytest.mark.parametrize(("name", "initial"), CASES)
@pytest.mark.parametrize("n_pol", [1, 2])
@pytest.mark.parametrize("name_of_rule", ["K=4", "K=0"])
def test_the_launch_fraction_is_within_its_scan_bound(
    name: str,
    initial: float,
    n_pol: int,
    name_of_rule: str,
    record_property: Callable[[str, float], None],
) -> None:
    """The final fraction from one field: the exposure scan's absolute bound."""
    problem = argon(name, initial, n_pol, rule(name_of_rule))
    bound = launch_fraction_bound(problem)
    expected = launch_statistics(NumpyRefBackend, problem, GRID)
    actual = launch_statistics(NativeBackend, problem, GRID)
    error = abs(float(actual["electron_density"] - expected["electron_density"]))
    record_property("measured", error / DENSITY)
    record_property("bound", bound)
    assert error <= DENSITY * bound, (error, DENSITY * bound)
    # Every case but one has electrons; the pre-ionised one starts with them too.
    assert float(expected["electron_density"]) > 0.0


@dataclass(frozen=True)
class PowerRate:
    """The clipped power law of solprop's ``tests/mixture_cases.py``."""

    scale: float = 1e13
    exponent: int = 2
    reference: float = 3e10
    threshold: float = 1e7
    maximum: float = 1e16
    citation: str = "Synthetic rate for analytic channel tests"

    @property
    def parameters(self) -> tuple[tuple[str, float], ...]:
        return tuple(
            sorted(
                (
                    ("scale", self.scale),
                    ("exponent", self.exponent),
                    ("reference", self.reference),
                    ("threshold", self.threshold),
                    ("maximum", self.maximum),
                )
            )
        )

    def __call__(self, field_amplitude: npt.ArrayLike) -> Real:
        magnitude = np.abs(np.asarray(field_amplitude, dtype=float))
        return np.where(
            magnitude >= self.threshold,
            np.minimum(
                self.maximum, self.scale * (magnitude / self.reference) ** self.exponent
            ),
            0.0,
        )


def two_channels(
    initial: tuple[float, float],
) -> tuple[SpatialMedium, list[PlasmaResponse]]:
    """Return the mixture of ``mixture_cases`` at the original's two densities."""
    densities = (1.5e25, 0.5e25)
    omega = np.array([1e12, 1e18])
    optical = TabulatedSusceptibility(
        omega,
        np.full(2, 1e-30),
        np.zeros(2),
        (2 * np.pi * C / omega[-1], 2 * np.pi * C / omega[0]),
        citation="Synthetic constant refractivity with no absorption",
    )
    components = tuple(
        Component(
            Species(
                f"synthetic {i}",
                optical,
                gamma3=(i + 1) * 1e-52,
                ionisation=IonisationData((i + 1) * 1e-18),
            ),
            Constant(density),
            f"population-{i}",
            preionisation=fraction,
        )
        for i, (density, fraction) in enumerate(zip(densities, initial, strict=True))
    )
    responses = [
        PlasmaResponse(
            component.species,
            build_rate_table(PowerRate(scale=(i + 1) * 1e13), (1e7, 1e11)),
            preionisation=component.preionisation,
            species=component.label,
        )
        for i, component in enumerate(components)
    ]
    return SpatialMedium(Mixture.of(*components)), responses


def test_two_channels_are_within_their_budget(
    record_property: Callable[[str, float], None],
) -> None:
    """A mixture, ``zero_crossings``: density-weighted budgets, at the stage."""
    initial = (0.0, 0.05)
    medium, responses = two_channels(initial)
    problem = assembled(responses, 2, medium)
    problem = launched(problem, stress("zero_crossings", 2), rule("K=4"))
    assert len(problem.responses.plasma_channels) == 2
    assert_stage_within_bound(problem, record_property)
