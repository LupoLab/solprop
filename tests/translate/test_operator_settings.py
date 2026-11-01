# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The separable-operator settings and the exponent budget (SEP 22 section 5.3).

Every setting at its default is today's operator and record exactly (structural);
the budget refuses only a step the run cannot shrink whose factor overflows, warns
within a tenth of the limit, and leaves adaptive runs alone (section 4.5).
"""

import dataclasses
import warnings

import numpy as np
import pytest

from legacy_geometry import bare_absorber
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.boundaries import TransverseCollar, absorber, transverse_collar_rate
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import (
    ExponentBudgetWarning,
    TranslationError,
)
from solprop.grid import EnvelopeGrid, LineGrid, RadialGrid
from solprop.linear import Frame
from solprop.modes import PlanarGrid, SlabMode
from solprop.nonlinear.responses import KerrResponse
from solprop.output.schema import serialise
from solprop.profiles import Linear
from solprop.pulses import GaussianPulse
from solprop.spatial import (
    EVANESCENT_RATE,
    SpatialDispersion,
    SpatialFlux,
    SpatialMedium,
    spatial_energy,
)
from solprop.stepper import FixedSchedule, Multistep, PIController, RungeKutta
from solprop.translate.chebyshev import build
from solprop.translate.linop import LinearOperator, phi_rate
from solprop.translate.problem import EXPONENT_BUDGET, LOG_DBL_MAX, assemble
from solprop.translate.radial import radial_operators, radial_rate_bounds
from solprop.translate.spatial import rate_bounds, separable_values

WAVELENGTH = 800e-9


def evanescent_beam(grid, space, medium):
    """A 1 um Gaussian beam of 1 pJ as an array, occupying evanescent columns.

    ``SpatialPulse`` refuses a beam with more than rounding-level content in columns
    that do not propagate (SEP 34 section 4.3); these tests need that content, so the
    beam is built here and scaled to 1 pJ of the propagating flux.
    """
    profile = np.exp(-((space.r / 1e-6) ** 2))
    if isinstance(space, RadialGrid):
        columns = space.hankel.forward(profile)
    else:
        columns = np.fft.fft2(profile.reshape(space.n_y, space.n_x)).ravel()
    temporal = GaussianPulse(wavelength=WAVELENGTH, fwhm=100e-15, energy=1.0).build(
        grid
    )
    field = columns[:, None, None] * temporal
    flux = SpatialFlux.resolve(grid, space, medium)
    return field * np.sqrt(1e-12 / spatial_energy(field, flux))


def line(
    *,
    step: float | None = 1e-4,
    length: float = 4e-4,
    integrator=None,
    medium: SpatialMedium | None = None,
    **settings,
) -> Simulation:
    """A Kerr line whose finest columns are evanescent, replayed at `step`."""
    grid = EnvelopeGrid.from_limits(length, WAVELENGTH, (700e-9, 950e-9), 400e-15)
    # The budgets below are the evanescent and transverse rates alone: no collar.
    settings.setdefault("absorber", bare_absorber(grid))
    # A 0.2 um spacing: the grid's largest k_x, 1.6e7 rad/m, is past k0 = 7.9e6.
    space = LineGrid(2e-6, 22, window_factor=0.1)
    pulse = evanescent_beam(grid, space, SpatialMedium() if medium is None else medium)
    planes = [length]
    return Simulation.from_free(
        grid,
        space,
        SpatialMedium() if medium is None else medium,
        [KerrResponse(1e-23, thg=False)],
        pulse,
        settings=SimulationSettings(
            controller=PIController(rtol=1e-6, atol=1e-14),
            save_planes=planes,
            dz_max=length if step is None else step,
            dz_init=length / 8 if step is None else step,
            schedule=None
            if step is None
            else FixedSchedule.uniform(0.0, length, step, save_planes=planes),
            integrator=RungeKutta() if integrator is None else integrator,
            backend="numpy_ref",
            **settings,
        ),
        noise=None,
    )


def test_every_setting_at_its_default_is_todays_operator_and_record():
    plain = line()
    zero = line(transverse_absorber=TransverseCollar(0.0))
    operator = plain.translate().linear.separable
    assert operator is not None
    assert operator.evanescent_rate == EVANESCENT_RATE
    assert operator.kperp_rate is not None
    assert np.array_equal(operator.kperp_rate, np.zeros(operator.kperp_sq.size))
    collared = zero.translate().linear.separable
    assert collared is not None
    # A zero rate leaves every element of the operator as it was (s + 0 = s).
    for one, other in zip(
        separable_values(operator), separable_values(collared), strict=True
    ):
        assert np.array_equal(one, other)
    configuration = plain.metadata().configuration
    assert "evanescent_rate" not in configuration
    assert "transverse_absorber" not in configuration
    # An explicit default rate is the same operator, recorded because it was asked.
    explicit = line(evanescent_rate=EVANESCENT_RATE)
    stated = explicit.translate().linear.separable
    assert stated is not None
    assert np.array_equal(separable_values(stated)[0], separable_values(operator)[0])
    assert explicit.metadata().configuration["evanescent_rate"] == EVANESCENT_RATE


def test_the_settings_reach_the_operator_and_the_record():
    collar = TransverseCollar(5e3, fraction=0.2)
    simulation = line(evanescent_rate=3e5, transverse_absorber=collar)
    problem = simulation.translate()
    operator = problem.linear.separable
    assert operator is not None
    assert operator.evanescent_rate == 3e5
    space = simulation.spatial_grid
    assert isinstance(space, LineGrid) and operator.kperp_rate is not None
    assert np.array_equal(operator.kperp_rate, transverse_collar_rate(space, collar))
    configuration = simulation.metadata().configuration
    assert configuration["evanescent_rate"] == 3e5
    # The collar is recorded as its two numbers, never as its column array.
    recorded = serialise(configuration["transverse_absorber"])
    assert recorded == {
        "__class__": "solprop.boundaries.TransverseCollar",
        "strength": 5e3,
        "fraction": 0.2,
    }
    assert (
        simulation.metadata().configuration_hash()
        != line().metadata().configuration_hash()
    )


def test_field_retention_leaves_the_configuration_hash_unchanged():
    digest = line().metadata().configuration_hash()
    assert line(fields="none").metadata().configuration_hash() == digest
    assert (
        line(fields="planes", field_planes=(-1,)).metadata().configuration_hash()
        == digest
    )


def test_the_settings_are_refused_where_they_do_not_apply():
    with pytest.raises(TranslationError, match="positive and finite"):
        line(evanescent_rate=0.0).translate()
    modal = Simulation.from_gnlse(
        0.11,
        0.01,
        [0.0, 0.0, -1.183e-26],
        wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=9.8,
        shape="sech",
        noise=None,
    )
    for name, value in (
        ("evanescent_rate", 1e3),
        ("transverse_absorber", TransverseCollar(1.0)),
    ):
        modal.settings = dataclasses.replace(modal.settings, **{name: value})
        modal._problem = None
        with pytest.raises(TranslationError, match="no spatial grid"):
            modal.translate()
        modal.settings = dataclasses.replace(modal.settings, **{name: None})
    grid = EnvelopeGrid.from_limits(1e-3, WAVELENGTH, (700e-9, 950e-9), 400e-15)
    controls = {
        "controller": PIController(rtol=1e-6, atol=1e-14),
        "save_planes": [1e-3],
        "dz_max": 1e-4,
        "dz_init": 1e-4,
        "transverse_absorber": TransverseCollar(1.0),
    }
    radial = RadialGrid(1e-4, 16)
    with pytest.raises(TranslationError, match="not on RadialGrid"):
        assemble(
            grid,
            None,
            [],
            np.zeros((16, 1, grid.n_omega), complex),
            spatial_grid=radial,
            medium=SpatialMedium(),
            **controls,
        )
    planar = PlanarGrid([SlabMode(50e-6, "Ar", 1.0)], 1e-4, 8)
    with pytest.raises(TranslationError, match="not on PlanarGrid"):
        assemble(
            grid,
            None,
            [],
            np.zeros((planar.n_columns, 1, grid.n_omega), complex),
            spatial_grid=planar,
            medium=SpatialMedium(),
            **controls,
        )


# --- the exponent budget (section 4.5) -------------------------------------------

RATE = 1e6
"""An evanescent cap below the grid's physical rate, so the cap is the loss bound."""


def test_the_loss_bound_is_the_evanescent_cap_here():
    operator = line(evanescent_rate=RATE).translate().linear.separable
    assert operator is not None
    bounds = rate_bounds(operator)
    assert bounds.loss == RATE and bounds.evanescent == RATE and bounds.gain == 0.0


def test_a_replayed_step_just_inside_the_budget_completes_finite():
    step = 0.85 * LOG_DBL_MAX / RATE
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExponentBudgetWarning)
        simulation = line(evanescent_rate=RATE, step=step, length=4 * step)
    # The exponent budget is the subject; the beam reaches the window's edge.
    simulation.settings = dataclasses.replace(simulation.settings, boundary_check=None)
    output = simulation.run(progress="none")
    assert np.all(np.isfinite(output["field"]))


def test_a_step_in_the_warning_band_warns_naming_the_admissible_step():
    step = 0.95 * LOG_DBL_MAX / RATE
    with pytest.warns(ExponentBudgetWarning, match="within a tenth") as caught:
        line(evanescent_rate=RATE, step=step, length=4 * step).translate()
    admissible = EXPONENT_BUDGET * LOG_DBL_MAX / RATE
    assert f"take steps of at most {admissible:.3g} m" in str(caught[0].message)


def test_a_replayed_step_past_the_budget_is_refused_naming_step_and_rate():
    step = 1.02 * LOG_DBL_MAX / RATE
    with pytest.raises(TranslationError, match="overflows") as caught:
        line(evanescent_rate=RATE, step=step, length=4 * step).translate()
    message = str(caught.value)
    assert f"replayed step {step:.3g} m with evanescent_rate {RATE:.3g} 1/m" in message
    assert f"exp({step * RATE:.0f})" in message
    assert f"take steps of at most {LOG_DBL_MAX / RATE:.3g} m" in message
    assert f"evanescent_rate at most {LOG_DBL_MAX / step:.3g} 1/m" in message


def test_the_step_floor_is_held_to_the_budget_and_an_adaptive_run_is_not():
    floor = 1.02 * LOG_DBL_MAX / RATE
    with pytest.raises(TranslationError, match="step floor dz_min"):
        line(
            evanescent_rate=RATE, step=None, length=8 * floor, dz_min=floor
        ).translate()
    # Adaptive, with dz_max * lambda = 785: its controller rejects a non-finite trial
    # and shrinks, so translation refuses nothing.
    adaptive = line(evanescent_rate=RATE, step=None, length=785 / RATE)
    adaptive.translate()


def test_a_transverse_collar_past_the_budget_names_its_strength():
    step = 1e-4
    strength = 2.2 * LOG_DBL_MAX / step
    with pytest.raises(
        TranslationError, match="transverse_absorber strength"
    ) as caught:
        line(transverse_absorber=TransverseCollar(strength), step=step).translate()
    rest = EVANESCENT_RATE
    assert f"at most {2 * (LOG_DBL_MAX / step - rest):.3g} 1/m" in str(caught.value)


def test_gain_spans_the_adams_methods_steps():
    # A gain medium, Im n < 0: the propagating columns grow, fastest near cutoff.
    gain = SpatialMedium(1.0 - 1e-4j)
    operator = line(medium=gain).translate().linear.separable
    assert operator is not None
    bounds = rate_bounds(operator)
    assert bounds.gain >= float(np.max(np.real(separable_values(operator)[0]))) > 0
    # An exponent of 300 a step: inside the budget for the Runge-Kutta march, five
    # steps' worth for a fourth-order Adams method, whose carry spans q + 1 steps.
    step = 300 / bounds.gain
    line(medium=gain, step=step, length=2 * step).translate()
    with pytest.raises(TranslationError, match=r"gain over 5 step\(s\)"):
        line(
            medium=gain, step=step, length=2 * step, integrator=Multistep(order=4)
        ).translate()


def test_a_backend_handed_the_refused_step_forms_a_non_finite_stage():
    allowed = 0.85 * LOG_DBL_MAX / RATE
    past = 1.1 * LOG_DBL_MAX / RATE
    simulation = line(evanescent_rate=RATE, step=allowed, length=4 * allowed)
    problem = simulation.translate()
    backend = NumpyRefBackend(problem, grid=simulation.grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        # A trial of the refused length, as only a direct caller can take one: the
        # inverse factor exp(lambda h) is infinite and so is the stage it scales.
        backend._hoist(0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            norm, state = backend._trial_step(0.0, past)
        assert not (np.isfinite(norm) and np.all(np.isfinite(state)))
        norm, state = backend._trial_step(0.0, allowed)
        assert np.isfinite(norm) and np.all(np.isfinite(state))
    finally:
        backend.close()


def test_a_lossy_medium_past_the_budget_names_only_the_step():
    # Im n = 1e-2: the propagating columns lose k0 Im n = 7.9e4 1/m, more than the
    # evanescent cap; no setting moves that term, so only the step is named.
    step = 1e-2
    with pytest.raises(TranslationError, match="the medium's attenuation") as caught:
        line(medium=SpatialMedium(1.0 + 1e-2j), step=step, length=2 * step).translate()
    assert (
        " at most " in str(caught.value)
        and "1/m" not in str(caught.value).split("take steps")[1]
    )


def test_two_settings_that_overflow_alone_name_only_the_step():
    step = 1e-4
    rate = 1.2 * LOG_DBL_MAX / step
    with pytest.raises(TranslationError, match="overflows") as caught:
        line(
            evanescent_rate=rate,
            transverse_absorber=TransverseCollar(2.4 * LOG_DBL_MAX / step),
            step=step,
        ).translate()
    # Either term alone overflows, so no admissible rate for the binding one exists.
    assert "1/m" not in str(caught.value).split("take steps")[1]


# --- the radial kind's budget (SEP 22, Q16) --------------------------------------


def radial(
    *,
    step: float | None,
    length: float,
    integrator=None,
    **settings,
) -> Simulation:
    """A radial Kerr run whose outer columns are evanescent, capped at RATE."""
    grid = EnvelopeGrid.from_limits(length, WAVELENGTH, (700e-9, 950e-9), 400e-15)
    # The budgets below are the evanescent and transverse rates alone: no collar.
    settings.setdefault("absorber", bare_absorber(grid))
    # Sixteen columns over 4 um: the largest k_r, about 1.2e7 rad/m, is past k0.
    space = RadialGrid(4e-6, 16)
    pulse = evanescent_beam(grid, space, SpatialMedium())
    planes = [length]
    return Simulation.from_radial(
        grid,
        space,
        SpatialMedium(),
        [KerrResponse(1e-23, thg=False)],
        pulse,
        settings=SimulationSettings(
            controller=PIController(rtol=1e-6, atol=1e-14),
            save_planes=planes,
            dz_max=length if step is None else step,
            dz_init=length / 8 if step is None else step,
            schedule=None
            if step is None
            else FixedSchedule.uniform(0.0, length, step, save_planes=planes),
            integrator=RungeKutta() if integrator is None else integrator,
            backend="numpy_ref",
            evanescent_rate=RATE,
            **settings,
        ),
        noise=None,
    )


def test_the_radial_bounds_are_the_constant_operators_extremes():
    linear = radial(step=None, length=1e-3).translate().linear
    assert linear.separable is None and linear.phi_constant is not None
    bounds = radial_rate_bounds(linear)
    # Structural: the bound is the table's own extreme, dPhi = D h exactly; the
    # capped evanescent columns bind it.
    assert bounds.loss == float(np.max(-np.real(linear.phi_constant))) == RATE
    assert bounds.gain == 0.0


def test_the_radial_budget_refuses_warns_and_admits():
    past = 1.02 * LOG_DBL_MAX / RATE
    with pytest.raises(TranslationError, match="overflows") as caught:
        radial(step=past, length=4 * past).translate()
    message = str(caught.value)
    assert f"replayed step {past:.3g} m with the radial operator's attenuation" in (
        message
    )
    assert f"take steps of at most {LOG_DBL_MAX / RATE:.3g} m" in message
    band = 0.95 * LOG_DBL_MAX / RATE
    with pytest.warns(ExponentBudgetWarning, match="within a tenth"):
        radial(step=band, length=4 * band).translate()
    # Adaptive runs are never refused above their floor.
    radial(step=None, length=4 * past).translate()


def test_a_radial_step_inside_the_budget_completes_finite():
    step = 0.85 * LOG_DBL_MAX / RATE
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExponentBudgetWarning)
        simulation = radial(step=step, length=4 * step)
        # The exponent budget is the subject; the beam reaches the window's edge.
        simulation.settings = dataclasses.replace(
            simulation.settings, boundary_check=None
        )
        output = simulation.run(progress="none")
    assert np.all(np.isfinite(output["field"]))


def test_a_radial_backend_handed_the_refused_step_forms_a_non_finite_stage():
    allowed = 0.85 * LOG_DBL_MAX / RATE
    simulation = radial(step=allowed, length=4 * allowed)
    problem = simulation.translate()
    backend = NumpyRefBackend(problem, grid=simulation.grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend._hoist(0.0)
        # The radial inverse factor exp(-dPhi) overflows as the separable one does.
        with np.errstate(over="ignore", invalid="ignore"):
            norm, state = backend._trial_step(0.0, 1.1 * LOG_DBL_MAX / RATE)
        assert not (np.isfinite(norm) and np.all(np.isfinite(state)))
    finally:
        backend.close()


def test_the_radial_bounds_cover_a_varying_medium():
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    space = RadialGrid(6e-4, 16, window_factor=0.0)
    medium = SpatialMedium("Ar", pressure=Linear(0.5, 1.5, grid.length))
    model = SpatialDispersion(medium, grid, Frame.comoving(grid))
    # The spectral collar gives the operator a real part for the bound to cover.
    expansion, _ = radial_operators(
        space, model, n_pol=1, z_max=grid.length, absorber=absorber(grid)
    )
    assert expansion.phi_expansion is not None
    # The same rate as a z table, the representation of many-parameter media.
    lo, hi = expansion.z_span
    table = build(
        lambda positions: np.stack([phi_rate(expansion, float(z)) for z in positions]),
        lo,
        hi,
    )
    phi = table.integral()
    tabulated = LinearOperator(
        phi_representation="table",
        phi_breakpoints=phi.breakpoints,
        phi_coefficients=phi.coefficients,
        phi_degrees=phi.degrees,
        hard_kinks=phi.hard_kinks,
        phi_constant=None,
    )
    positions = np.linspace(lo, hi, 41)
    for operator in (expansion, tabulated):
        bounds = radial_rate_bounds(operator)
        sampled = np.stack([np.real(phi_rate(operator, float(z))) for z in positions])
        # Sufficient: never below the rate the backends evaluate anywhere sampled.
        assert bounds.loss >= float(np.max(-sampled)) > 0
        assert bounds.gain >= float(np.max(sampled, initial=0.0))
        # And close: the series converge fast on a smooth density, so the bound
        # sits within a hundredth of the sampled extreme.
        assert bounds.loss <= 1.01 * float(np.max(-sampled))
