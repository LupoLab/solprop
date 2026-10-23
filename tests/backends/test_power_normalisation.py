# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Weighted energy and analytic-signal Poynting flux on every backend."""

import math
from dataclasses import replace

import numpy as np
import pytest

from backends.matched import fixed_order
from bounds import EPS, assert_within, gamma, step_evaluations
from power_identity import constant_fill_m2
from solprop.constants import EPS0, C
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.linear import Frame
from solprop.modes import MarcatiliMode
from solprop.modes.quadrature import mode_matrix, on_axis_rule
from solprop.nonlinear.gnlse import problem_from
from solprop.profiles import PressureGradient
from solprop.pulses import GaussianPulse, SechPulse
from solprop.stats import StatsSpec, _poynting_statistics, analytic_field, energy
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.linop import _dispersion_phi, power_weight_at
from solprop.translate.problem import assemble


# One backend of the three is enough to hold the medium tier's budget: the identity is
# a property of the translation, and `numpy_ref` is the oracle every other backend is
# validated against. The two compiled arms replay the same ten steps and are worth their
# seconds only in the slow lane. One tier marker each, as `test_tiers.py` requires.
@pytest.mark.parametrize(
    "backend_name",
    [
        pytest.param("numpy_ref", marks=pytest.mark.medium),
        pytest.param("numba_cpu", marks=pytest.mark.slow),
        pytest.param("native", marks=pytest.mark.slow),
    ],
    indirect=True,
)
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_constant_fill_internal_trajectory_is_unchanged(backend_class):
    # Ten steps, 1025 spectral bins, two modes: a short functional replay. The two
    # contracts hold the same launch and operator to rounding (a division, a fit of
    # one function), and replay one fixed-order method on one schedule, so their
    # trajectories agree within a rounding budget fixed before measurement and checked
    # at every edge: ten steps of the Adams weights' evaluations, each a transform's
    # 32 eps a level and the plasma column's gamma(12 n_to) (the band-rule ladder's
    # count with one tap).
    simulation = constant_fill_m2()
    problem = fixed_order(simulation.translate())
    grid = simulation.grid
    root_weight = np.ones(problem.shape)
    for mode_index, mode in enumerate(simulation.modes):
        root_weight[0, mode_index, 0, grid.active_mask] = np.sqrt(
            np.real(mode.neff(grid.omega[grid.active_mask], 0.0))
        )
    assert simulation.noise is not None
    legacy_initial = replace(
        problem.initial,
        field0=simulation._field0[None] / root_weight,
        noise0=simulation.noise.spectrum[None] / root_weight,
    )
    absorber = simulation.settings.absorber
    assert absorber is not None
    legacy_phi = _dispersion_phi(
        simulation.modes,
        grid,
        simulation.frame
        or Frame.comoving(
            grid,
            thg=any(bool(getattr(one, "thg", False)) for one in simulation.responses),
        ),
        z_max=grid.length,
        absorber_rate=absorber.spectral_rate,
    )
    # One division by the root weight on either side, and one fit of one function.
    rounding = 4 * EPS
    np.testing.assert_allclose(
        problem.initial.field0, legacy_initial.field0, rtol=rounding
    )
    assert problem.initial.noise0 is not None
    assert legacy_initial.noise0 is not None
    np.testing.assert_allclose(
        problem.initial.noise0, legacy_initial.noise0, rtol=rounding
    )
    assert_within(
        problem.linear.phi_coefficients, legacy_phi.phi_coefficients, rounding
    )
    # Replay the old dispersion-only contract with the independently pre-divided
    # launch. Keep response/rate tables identical: this tests propagation, not
    # rebuilding a field-range-dependent table after changing its input units.
    reference = replace(
        problem,
        initial=legacy_initial,
        linear=replace(
            problem.linear,
            phi_constant=legacy_phi.phi_constant,
            phi_coefficients=legacy_phi.phi_coefficients,
        ),
    )
    histories = []
    assert problem.stepper.schedule is not None
    for source in (problem, reference):
        owner = backend_class(source, grid=grid, modes=simulation.modes)
        try:
            owner.set_initial(source.initial.field0, None)
            history = [owner.observe().require_field().copy()]
            for position in problem.stepper.schedule.edges[1:11]:
                summary = owner.propagate_to(float(position))
                assert summary.n_steps == 1
                assert summary.n_rejected == 0
                history.append(owner.observe().require_field().copy())
            histories.append(np.stack(history))
        finally:
            owner.close()
    n_to = grid.n_to
    per_evaluation = 32 * EPS * math.ceil(math.log2(n_to)) + gamma(12 * n_to)
    bound = 10 * step_evaluations(problem) * per_evaluation
    for index, (ours, theirs) in enumerate(zip(*histories, strict=True)):
        assert_within(ours, theirs, bound, what=f"edge {index}")


@pytest.mark.medium
def test_gnlse_route_is_unchanged(backend_class):
    # Ten steps on 256 bins exercise the M1 soliton equation and its statistics;
    # this is a short functional identity, not the slow recurrence convergence study.
    # The two contracts hold the same launch and operator to rounding and replay one
    # Runge-Kutta schedule: the ladder's 16 eps a step for the field, and for the
    # statistics twice that, a reduction, and a factor four for the FWHM's
    # interpolation.
    grid = EnvelopeGrid.from_limits(0.001, 835e-9, (600e-9, 1.2e-6), 400e-15)
    gnlse = problem_from(
        [0.0, 0.0, -1.2e-26], omega_ref=grid.omega0, gamma=0.11, shock=False
    )
    field = SechPulse(wavelength=835e-9, width=30e-15, peak_power=1000.0).build(grid)
    problem = assemble(
        grid,
        gnlse.mode,
        [gnlse.kerr],
        field,
        controller=PIController(rtol=1e-6, atol=1e-12),
        save_planes=[grid.length],
        dz_max=grid.length / 10,
        dz_init=grid.length / 10,
        schedule=FixedSchedule.uniform(0.0, grid.length, grid.length / 10),
        prefactor_index=1.0,
        density=1.0,
        shock=False,
    )
    legacy = _dispersion_phi(gnlse.mode, grid, Frame.comoving(grid), z_max=grid.length)
    np.testing.assert_allclose(problem.initial.field0[0], field, rtol=4 * EPS)
    assert_within(problem.linear.phi_coefficients, legacy.phi_coefficients, 4 * EPS)
    # Structural: a GNLSE's power weight is exactly one.
    np.testing.assert_array_equal(power_weight_at(problem, 0.0), 1.0)
    reference = replace(
        problem,
        linear=replace(
            problem.linear,
            phi_constant=legacy.phi_constant,
            phi_coefficients=legacy.phi_coefficients,
        ),
        initial=replace(problem.initial, field0=field[None]),
    )
    spec = StatsSpec(names=("energy", "peak_power", "fwhm_t"))
    results = []
    for source in (problem, reference):
        backend = backend_class(source, grid=grid, modes=(gnlse.mode,))
        try:
            backend.set_initial(source.initial.field0, None)
            backend.propagate_to(grid.length)
            results.append(
                (backend.observe().require_field().copy(), backend.stats(spec))
            )
        finally:
            backend.close()
    bound = 10 * 16 * EPS
    assert_within(results[0][0], results[1][0], bound, what="field")
    statistics = 4 * (2 * bound + gamma(grid.n_to))
    for name in spec.names:
        np.testing.assert_allclose(
            results[0][1][name], results[1][1][name], rtol=statistics, err_msg=name
        )


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.parametrize("grid_class", [EnvelopeGrid, FieldGrid])
def test_backend_temporal_statistics_use_the_hermitian_flux(backend_class, grid_class):
    grid = grid_class.from_limits(0.01, 800e-9, (600e-9, 1e-6), 100e-15)
    mode = MarcatiliMode(125e-6, "Ar", 20.0, cladding=1.45, loss=False)
    field = GaussianPulse(wavelength=800e-9, fwhm=12e-15, energy=1e-9).build(grid)
    spec = StatsSpec(names=("peak_power", "peak_power_total", "peak_intensity"))
    problem = assemble(
        grid,
        mode,
        [],
        field,
        controller=PIController(rtol=1e-6, atol=1e-12),
        save_planes=[0.01],
        dz_max=0.01,
        dz_init=0.001,
    )
    backend = backend_class(problem, grid=grid, modes=(mode,))
    try:
        backend.set_initial(problem.initial.field0, None)
        values = backend.stats(spec)
        a = analytic_field(grid, problem.initial.field0[0])
        aw = analytic_field(
            grid, problem.initial.field0[0] * power_weight_at(problem, 0.0)
        )
        power = np.real(a * np.conj(aw)).sum(axis=1)
        axis = mode_matrix(
            (mode,), on_axis_rule(mode.dimlimits(0.0)[0]), 0.0, components=("x", "y")
        ).synthesis
        electric = np.tensordot(axis, a, axes=([1], [0]))
        magnetic = np.tensordot(axis, aw, axes=([1], [0]))
        intensity = (
            0.5 * EPS0 * C * np.real(electric * np.conj(magnetic)).sum(axis=(0, 1))
        )
        # Independent FFT implementations and two Hermitian products; the FFT
        # error grows logarithmically with the coarse transform length.
        bound = 512 * np.finfo(float).eps * np.ceil(np.log2(grid.n_t))
        np.testing.assert_allclose(
            values["peak_power"][0], power.max(axis=-1), rtol=bound, atol=0
        )
        np.testing.assert_allclose(
            values["peak_power_total"][0], power.sum(axis=0).max(), rtol=bound, atol=0
        )
        np.testing.assert_allclose(
            values["peak_intensity"][0], intensity.max(), rtol=bound, atol=0
        )
    finally:
        backend.close()


@pytest.mark.medium
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.parametrize("grid_class", [EnvelopeGrid, FieldGrid])
def test_true_energy_is_conserved_along_a_lossless_gradient(backend_class, grid_class):
    # Ten fixed, linear-only steps on a 40 fs window: a few seconds for all CPU
    # backends including translation; no nonlinear compilation or long timing run.
    grid = grid_class.from_limits(0.001, 800e-9, (600e-9, 1e-6), 40e-15)
    mode = MarcatiliMode(
        125e-6,
        "Ar",
        PressureGradient.two_point(grid.length, 20.0, 1.0),
        cladding=1.45,
        loss=False,
    )
    field = np.zeros((1, 1, grid.n_omega), complex)
    frequency = np.flatnonzero(grid.active_mask)[3]
    field[0, 0, frequency] = 1.0
    problem = assemble(
        grid,
        mode,
        [],
        field,
        controller=PIController(rtol=1e-6, atol=1e-12),
        save_planes=[grid.length],
        dz_max=grid.length / 10,
        dz_init=grid.length / 10,
    )
    backend = backend_class(problem, grid=grid, modes=(mode,))
    spec = StatsSpec(names=("energy",))
    try:
        backend.set_initial(problem.initial.field0, None)
        entrance = backend.stats(spec)["energy"]
        backend.propagate_to(grid.length)
        final = backend.observe().require_field()
        # The conservation bound, set before measurement.
        bound = 10 * grid.n_omega * np.finfo(float).eps
        np.testing.assert_allclose(
            backend.stats(spec)["energy"], entrance, rtol=bound, atol=0
        )
        ratio = (
            power_weight_at(problem, 0.0)[0, 0, frequency]
            / power_weight_at(problem, grid.length)[0, 0, frequency]
        )
        np.testing.assert_allclose(
            np.sum(np.abs(final) ** 2) / np.sum(np.abs(problem.initial.field0) ** 2),
            ratio,
            rtol=bound,
            atol=0,
        )
    finally:
        backend.close()


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.parametrize("grid_class", [EnvelopeGrid, FieldGrid])
def test_energy_statistics_equal_the_weighted_sum(backend_class, grid_class):
    grid = grid_class.from_limits(0.01, 800e-9, (600e-9, 1e-6), 40e-15)
    mode = MarcatiliMode(125e-6, "Ar", 20.0, cladding=1.45, loss=False)
    rng = np.random.default_rng(42)
    field = rng.normal(size=(1, 1, grid.n_omega)).astype(complex)
    field += 1j * rng.normal(size=field.shape)
    field *= grid.active_mask
    spec = StatsSpec(
        names=("energy", "energy_window"),
        energy_windows=np.array(
            [[float(grid.omega.min()) - 1.0, float(grid.omega.max()) + 1e10]]
        ),
    )
    problem = assemble(
        grid,
        mode,
        [],
        field,
        controller=PIController(rtol=1e-6, atol=1e-12),
        save_planes=[0.01],
        dz_max=0.01,
        dz_init=0.001,
        stats_spec=spec,
    )
    backend = backend_class(problem, grid=grid, modes=(mode,))
    try:
        backend.set_initial(problem.initial.field0, None)
        actual = backend.stats(spec)
        expected = energy(
            grid, problem.initial.field0[0] * np.sqrt(power_weight_at(problem, 0.0))
        )
        # The n_omega-epsilon spectral reduction bound, fixed in advance.
        bound = grid.n_omega * np.finfo(float).eps
        np.testing.assert_allclose(actual["energy"][0], expected, rtol=bound, atol=0)
        np.testing.assert_allclose(
            actual["energy_window"][0, 0], expected, rtol=bound, atol=0
        )
    finally:
        backend.close()


@pytest.mark.parametrize("grid_class", [EnvelopeGrid, FieldGrid])
def test_poynting_monochromatic_reduction_and_integrated_true_power(grid_class):
    grid = grid_class.from_limits(0.01, 800e-9, (600e-9, 1e-6), 40e-15)
    field = np.zeros((1, 1, grid.n_omega), complex)
    field[0, 0, np.flatnonzero(grid.active_mask)[3]] = 2 + 3j
    weight = np.full(field.shape, 1.006)
    axis = np.array([[2.0], [0.0]])
    spec = StatsSpec(names=("peak_intensity", "peak_power", "energy"))
    values = _poynting_statistics(spec, grid, field, weight, axis)
    electric = 2 * analytic_field(grid, field)
    expected = 0.5 * EPS0 * C * 1.006 * np.max(np.abs(electric) ** 2)
    # The inverse transform and Hermitian product each round; 32 eps bounds
    # this single occupied frequency without a propagation/discretisation budget.
    np.testing.assert_allclose(
        values["peak_intensity"], expected, rtol=32 * np.finfo(float).eps, atol=0
    )
    rng = np.random.default_rng(42)
    field = rng.normal(size=field.shape) + 1j * rng.normal(size=field.shape)
    field *= grid.active_mask
    weight = rng.uniform(1, 1.01, size=field.shape)
    flux = np.real(
        analytic_field(grid, field) * np.conj(analytic_field(grid, field * weight))
    )
    amplitude_power = np.abs(analytic_field(grid, field * np.sqrt(weight))) ** 2
    # Parseval cancels all unequal-frequency terms in either definition.
    np.testing.assert_allclose(
        np.sum(flux),
        np.sum(amplitude_power),
        rtol=10 * grid.n_t * np.finfo(float).eps,
        atol=0,
    )
