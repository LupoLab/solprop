# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Separate carrier averaging discrepancy from production-source numerical error."""

import math

import numpy as np
import pytest

from envelope_oracle_angular import BUDGETS, order_record
from envelope_oracle_propagation import mixture_propagation, propagate_pair
from envelope_oracle_qualification import (
    carrier_comparison,
    controlled_cases,
    mixture_work_convergence,
    window_cases,
    work_convergence,
)
from rate_oracles import adk_oracle
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.constants import CHARGE_MASS_RATIO, EPS0, C
from solprop.grid import EnvelopeGrid
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.pulses import GaussianPulse
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import assemble


@pytest.mark.slow
def test_complete_mixture_carrier_envelope_run_resolves_each_population():
    for resolution in (4, 8, 16):
        result = mixture_propagation(resolution)
        assert (
            max(result["species_fraction_relative_discrepancies"])
            < BUDGETS["controlled_envelope_relative_target"]
        )
        for model in result["models"].values():
            assert model["steps"] == 8
            assert all(0.01 < value < 1 for value in model["fractions"])
            # The growing plasma produces a blue shift in this chosen slab;
            # resolve its sign above the centroid reduction's rounding bound.
            assert (
                model["fundamental_blueshift_over_carrier"]
                > model["centroid_rounding_relative_bound"]
            )
            assert model["energy"] < model["initial_energy"]


@pytest.mark.slow
def test_mixture_work_equals_new_species_inventory_with_second_order_convergence():
    record = mixture_work_convergence()
    assert record["order"]["status"] == "pass"
    assert all(
        all(value > 0 for value in one["species_inventory_change_J_per_m3"])
        for one in record["levels"]
    )


@pytest.mark.slow
@pytest.mark.parametrize("ratio", [0.0, 0.35, 1.0])
def test_complete_carrier_envelope_run_in_controlled_regime(ratio):
    for resolution in (4, 8, 16):
        record = propagate_pair(8, 1e-3, ratio, oversampling=resolution)
        for quantity in (
            "fraction_relative_discrepancy",
            "energy_relative_discrepancy",
        ):
            # The model budget is independent of sampling; bandwidth, depletion,
            # oversampling and RK steps are varied separately.
            assert record[quantity] < BUDGETS["controlled_envelope_relative_target"]
        assert all(one["accepted_steps"] == 8 for one in record["models"].values())


@pytest.mark.slow
@pytest.mark.parametrize("ratio", [0.0, 0.35, 1.0])
def test_elliptical_current_carrier_limit(ratio):
    rows = [carrier_comparison(8, 1e-3, ratio, count) for count in (32, 64, 128)]
    for row in rows:
        # The budget is 1% for the controlled narrowband, weak-per-cycle case.
        # Both directional loss components must satisfy it independently.
        for name in (
            "loss_component_relative_errors",
            "drude_component_relative_errors",
        ):
            assert max(row[name]) < BUDGETS["controlled_envelope_relative_target"]
        assert (
            row["causal_current_relative_error"]
            < BUDGETS["controlled_envelope_relative_target"]
        )
        # The degree-four oracle's logarithmic rate is linear. Floating-point
        # axis recovery, table arithmetic and moment sums dominate its error.
        rounding = 64 * 128 * np.finfo(float).eps
        assert row["production_loss_relative_error"] < rounding
        assert row["production_population_error"] < rounding
    # Carrier sampling has its own convergence witness below. It cannot be
    # traded against the finite model discrepancy in these Fourier comparisons.
    convergence = work_convergence(ratio)
    assert convergence["envelope"]["status"] == "pass"
    assert convergence["carrier"]["status"] == "pass"


@pytest.mark.slow
def test_envelope_model_regime_varies_bandwidth_and_depletion_separately():
    rows = controlled_cases()
    assert len(rows) == 27
    outside = [
        row
        for row in rows
        if row["cycles_sigma"] == 0.25 and row["peak_rate_over_carrier"] == 0.5
    ]
    assert len(outside) == 3
    for row in outside:
        # Deliberately outside the accepted 1% envelope regime; this assertion
        # diagnoses the breakdown instead of relaxing a numerical error budget.
        assert (
            max(row["loss_component_relative_errors"])
            > BUDGETS["controlled_envelope_relative_target"]
        )
        assert row["production_loss_relative_error"] < 64 * 128 * np.finfo(float).eps


@pytest.mark.slow
def test_causal_current_windows_and_padding_are_independent_controls():
    rows = window_cases()
    assert len(rows) == 36
    for row in rows:
        # Padding and collar never alter the local cycle moments. Their effect
        # on the projected causal current is measured separately in the receipt.
        assert row["production_loss_relative_error"] < 64 * 128 * np.finfo(float).eps
        if row["cycles_sigma"] == 8 and row["window_sigmas"] >= 12:
            assert (
                row["causal_current_relative_error"]
                < BUDGETS["controlled_envelope_relative_target"]
            )
        elif row["cycles_sigma"] == 0.25:
            assert (
                row["causal_current_relative_error"]
                > BUDGETS["controlled_envelope_relative_target"]
            )


@pytest.mark.slow
def test_prescribed_density_drude_conserves_energy_with_fifth_order_steps():
    grid = EnvelopeGrid.from_limits(8e-6, 800e-9, (500e-9, 1.2e-6), 500e-15)
    mode = SpecifiedMode.from_taylor([grid.omega0 / C, 1 / C], grid.omega0, aeff=1e-8)
    initial = GaussianPulse(wavelength=800e-9, fwhm=20e-15, energy=1e-12).build(grid)
    density = 2e26
    phase = np.zeros(grid.n_omega)
    active = grid.active_mask & (grid.omega > 0)
    phase[active] = (
        CHARGE_MASS_RATIO * density * grid.length / (2 * EPS0 * C * grid.omega[active])
    )
    exact = initial * np.exp(1j * phase)
    norm = np.linalg.norm(initial)
    errors = []
    for steps in (8, 16, 32):
        problem = assemble(
            grid,
            mode,
            [EnvelopePlasmaResponse("Ar", adk_oracle("Ar"), preionisation=1.0)],
            initial,
            density=density,
            controller=PIController(rtol=1e-6, atol=1e-14),
            save_planes=[grid.length],
            dz_init=grid.length / steps,
            dz_max=grid.length / steps,
            schedule=FixedSchedule.uniform(0.0, grid.length, grid.length / steps),
            ionisation_field_range=(1e7, 1e11),
        )
        owner = NumbaCpuBackend(
            problem,
            grid=grid,
            threads=1,
            fftw_threads=1,
            blas_threads=1,
            plan_flag="estimate",
        )
        try:
            owner.set_initial(problem.initial.field0, None)
            owner.propagate_to(grid.length)
            actual = owner.observe().require_field()[0]
        finally:
            owner.close()
        error = np.linalg.norm(actual - exact) / norm
        errors.append(error)
        # DOPRI5 R(z) = sum(z**j/j!, j=0..5) + z**6/600.
        # Bound its difference from exp(z) and telescope N unitary exact steps.
        size = np.max(phase) / steps
        local = size**6 / 3600 + size**7 * np.exp(size) / math.factorial(7)
        bound = steps * local * (1 + local) ** (steps - 1)
        rounding = 128 * grid.n_to * steps * np.finfo(float).eps
        assert error < bound + rounding
        energy_ratio = np.linalg.norm(actual) ** 2 / norm**2
        assert abs(energy_ratio - 1) < 2 * bound + bound**2 + rounding
    assert order_record(errors, 5, 64 * np.finfo(float).eps)["status"] == "pass"
