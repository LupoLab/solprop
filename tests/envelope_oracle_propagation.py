# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run independent field and envelope solvers with a controlled magnitude law."""

import warnings

import numpy as np

from envelope_oracle_angular import order_record
from envelope_oracle_qualification import InstantaneousPolynomial
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.boundaries import absorber
from solprop.constants import EPS0, C
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import MarcatiliMode
from solprop.nonlinear import EnvelopePlasmaResponse, responses_for
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.nonlinear.responses import PlasmaResponse
from solprop.physdata import Mixture
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.ionisation import build_rate_table
from solprop.translate.problem import assemble


def propagate_pair(cycles, depletion, ratio, *, steps=8, oversampling=8):
    wavelength, length, area = 800e-9, 8e-6, 1e-8
    omega = 2 * np.pi * C / wavelength
    sigma = cycles * wavelength / C
    # Equal major-axis physical amplitude in both formulations; total energy
    # includes both axes. The uniform-disc mode has n_eff=1 exactly.
    energy = 0.5 * EPS0 * C * area * 1e20 * (1 + ratio**2) * np.sqrt(np.pi) * sigma
    pulse = GaussianPulse(
        wavelength=wavelength, fwhm=2 * np.sqrt(np.log(2)) * sigma, energy=energy
    )
    rate = build_rate_table(
        InstantaneousPolynomial(
            scale=depletion * omega,
            reference=1e10,
            exponent=4,
            threshold=1.0,
            maximum=1e30,
        ),
        (1.0, 1e11),
        rtol=1e-7,
    )
    names = ("energy", "species_ionisation", "peak_ionisation_rate")
    results = {}
    for label, grid_type, response in (
        ("field", FieldGrid, PlasmaResponse("Ar", rate)),
        ("envelope", EnvelopeGrid, EnvelopePlasmaResponse("Ar", rate)),
    ):
        grid = grid_type.from_limits(
            length,
            wavelength,
            (500e-9, 1.2e-6),
            12 * sigma,
            ffac=oversampling,
        )
        mode = SpecifiedMode.from_taylor([omega / C, 1 / C], omega, aeff=area)
        jones = np.asarray([1.0, 1j * ratio]) / np.sqrt(1 + ratio**2)
        initial = pulse.build(grid) * jones[None, :, None]
        problem = assemble(
            grid,
            mode,
            [response],
            initial,
            density=1e24,
            controller=PIController(rtol=1e-7, atol=1e-14),
            save_planes=[length],
            dz_init=length / steps,
            dz_max=length / steps,
            schedule=FixedSchedule.uniform(0.0, length, length / steps),
            absorber=absorber(grid, boundary_length=1e-3),
            stats_spec=StatsSpec(names=names),
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
            before = owner.stats(problem.stats_spec)
            summary = owner.propagate_to(length)
            after = owner.stats(problem.stats_spec)
            results[label] = {
                "n_t": grid.n_t,
                "n_to": grid.n_to,
                "dt": grid.dt,
                "dt_fine": grid.dt / (grid.n_to / grid.n_t),
                "accepted_steps": summary.n_steps,
                "initial_energy": float(np.sum(before["energy"])),
                "final_energy": float(np.sum(after["energy"])),
                "initial_fraction": before["species_ionisation"].item(),
                "final_fraction": after["species_ionisation"].item(),
                "memory_ledger": owner.memory_ledger(),
            }
        finally:
            owner.close()
    return {
        "cycles_sigma": cycles,
        "rate_over_carrier": depletion,
        "ratio": ratio,
        "oversampling": oversampling,
        "steps": steps,
        "models": results,
        "fraction_relative_discrepancy": abs(
            results["field"]["final_fraction"] - results["envelope"]["final_fraction"]
        )
        / results["field"]["final_fraction"],
        "energy_relative_discrepancy": abs(
            results["field"]["final_energy"] - results["envelope"]["final_energy"]
        )
        / results["field"]["final_energy"],
    }


def refinement_record(rows, reference, expected_order):
    """Keep an independent numerical floor beside every propagation order study."""
    floor = (
        128
        * max(one["n_to"] for one in reference["models"].values())
        * np.finfo(float).eps
    )
    return {
        "levels": rows,
        "reference": reference,
        "fraction_orders": {
            kind: order_record(
                [
                    abs(
                        one["models"][kind]["final_fraction"]
                        - reference["models"][kind]["final_fraction"]
                    )
                    / reference["models"][kind]["final_fraction"]
                    for one in rows
                ],
                expected_order,
                floor,
            )
            for kind in ("field", "envelope")
        },
    }


def propagation_studies():
    return {
        "mixture_temporal_levels": [mixture_propagation(value) for value in (4, 8, 16)],
        "model_comparisons": [
            propagate_pair(cycles, depletion, ratio)
            for cycles in (2, 8)
            for depletion in (1e-3, 0.05, 0.5)
            for ratio in (0.0, 0.35, 1.0)
        ],
        "temporal_refinement": [
            refinement_record(
                [
                    propagate_pair(2, 0.05, ratio, oversampling=value)
                    for value in (4, 8, 16)
                ],
                propagate_pair(2, 0.05, ratio, oversampling=32),
                2,
            )
            for ratio in (0.0, 0.35, 1.0)
        ],
        "step_refinement": [
            refinement_record(
                [propagate_pair(2, 0.05, ratio, steps=value) for value in (2, 4, 8)],
                propagate_pair(2, 0.05, ratio, steps=16),
                5,
            )
            for ratio in (0.0, 0.35, 1.0)
        ],
    }


def mixture_propagation(oversampling):
    """Compare two independently depleting species through complete propagation."""
    fill = Mixture.fill({"Ar": 0.3, "He": 0.7}, 1.0)
    mode = MarcatiliMode(125e-6, mixture=fill, cladding=1.45, loss=False)
    pulse = GaussianPulse(wavelength=800e-9, fwhm=60e-15, energy=1e-3)
    rates = {
        species: InstantaneousPolynomial(
            scale=scale * 2 * np.pi * C / 800e-9,
            reference=1e10,
            exponent=4,
            threshold=1.0,
            maximum=1e30,
        )
        for species, scale in (("Ar", 0.001), ("He", 0.002))
    }
    results = {}
    for label, grid_type in (("field", FieldGrid), ("envelope", EnvelopeGrid)):
        grid = grid_type.from_limits(
            8e-6, 800e-9, (500e-9, 1.2e-6), 320e-15, ffac=oversampling
        )
        initial = (
            pulse.build(grid)
            * np.asarray([1, 0.35j])[None, :, None]
            / np.sqrt(1 + 0.35**2)
        )
        # The neutral-index fit does not cover the FFT collar. The active optical
        # band and the same material extrapolation are retained on both grids.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MaterialRangeWarning)
            problem = assemble(
                grid,
                mode,
                responses_for(
                    fill, envelope=label == "envelope", kerr=False, plasma=rates
                ),
                initial,
                controller=PIController(rtol=1e-7, atol=1e-14),
                save_planes=[grid.length],
                dz_init=1e-6,
                dz_max=1e-6,
                schedule=FixedSchedule.uniform(0.0, grid.length, 1e-6),
                absorber=absorber(grid, boundary_length=1e-3),
                stats_spec=StatsSpec(names=("energy", "species_ionisation")),
                ionisation_field_range=(1.0, 1e11),
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
            before = owner.observe().require_field()
            initial_energy = float(
                np.sum(owner.stats(StatsSpec(names=("energy",)))["energy"])
            )
            summary = owner.propagate_to(grid.length)
            values = owner.stats(problem.stats_spec)
            results[label] = {
                "fractions": values["species_ionisation"].ravel().tolist(),
                "energy": float(np.sum(values["energy"])),
                "steps": summary.n_steps,
                "n_to": grid.n_to,
                "initial_energy": initial_energy,
                "fundamental_blueshift_over_carrier": (
                    fundamental_centroid(owner.observe().require_field(), grid)
                    - fundamental_centroid(before, grid)
                )
                / grid.omega0,
                "centroid_rounding_relative_bound": 128
                * grid.n_omega
                * np.finfo(float).eps,
            }
        finally:
            owner.close()
    fractions = np.asarray(results["field"]["fractions"])
    return {
        "oversampling": oversampling,
        "models": results,
        "species_fraction_relative_discrepancies": (
            np.abs(fractions - results["envelope"]["fractions"]) / fractions
        ).tolist(),
    }


def fundamental_centroid(field, grid):
    band = (grid.omega > 0.5 * grid.omega0) & (grid.omega < 1.5 * grid.omega0)
    power = np.sum(np.abs(field.reshape(-1, grid.n_omega)) ** 2, axis=0)[band]
    return float(np.sum(power * grid.omega[band]) / np.sum(power))
