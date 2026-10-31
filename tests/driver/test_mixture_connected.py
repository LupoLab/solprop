# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise the connected chemistry interface on its optical geometry and pulse.

All absorption data added here are synthetic. The nitrogen/oxygen rates are the
fitted effective-charge ADK model of Talebpour, Chien and Chin, given as the test-only
ADK oracle with the inputs solprop's former registry held (at dc4ca0d, before SEP 3
Phase 4b deleted its rates). This is a numerical interface acceptance case, not a
qualification of the air data in the ultraviolet.
"""

import json
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from rate_oracles import AdkOracle
from selected_backend import selected_backend
from solprop import fourier, processing
from solprop.constants import EPS0, C
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import FieldGrid
from solprop.modes import MarcatiliMode
from solprop.modes.quadrature import transverse_quadrature
from solprop.nonlinear import responses_for
from solprop.physdata import Component, Mixture, Species, TabulatedSusceptibility
from solprop.profiles import Constant
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.linop import delta_phi, power_weight_at

# Talebpour, Chien and Chin's effective charges (N2 0.9, O2 0.53) on a sigma (l = 0)
# shell, with the potentials of solprop's former registry (at dc4ca0d): the former
# ADK("N2") and ADK("O2") bit for bit.
NITROGEN_RATE = AdkOracle("N2", 2.4963898279351553e-18, 0, 0.9)
OXYGEN_RATE = AdkOracle("O2", 1.9337778507686377e-18, 0, 0.53)


def connected_simulation(ffac=6.0, *, passive=False):
    air = Mixture.air(6.0, temperature=300.0)
    omega = np.array([1e12, 1e18])
    optical = TabulatedSusceptibility(
        omega,
        np.full(2, 1e-30),
        np.full(2, 1e-24),
        (2 * np.pi * C / omega[-1], 2 * np.pi * C / omega[0]),
        citation="Synthetic constant refractivity and absorption; no ozone data",
    )
    mixture = Mixture.of(
        *(
            replace(component, preionisation=(index + 1) * 1e-3 if index < 2 else 0.0)
            for index, component in enumerate(air.components)
        ),
        Component(
            Species("synthetic absorber", optical, gamma3=0.0),
            Constant(1e20),
            "absorber",
        ),
        temperature=300.0,
    )
    grid = FieldGrid.from_limits(0.225, 800e-9, (300e-9, 2e-6), 200e-15, ffac=ffac)
    mode = MarcatiliMode(15e-6, mixture=mixture, cladding=1.45, loss=True)
    responses = (
        ()
        if passive
        else responses_for(
            mixture,
            plasma={"N2": NITROGEN_RATE, "O2": OXYGEN_RATE},
            # No Raman: optmat broadens a mixture's lines only by gases it resolves,
            # and refuses the synthetic absorber (SEP 3, D16). The accounting checked
            # here is the ionisation and the absorption; the former nitrogen Raman
            # term was the deleted registry's.
            raman=False,
            thg=False,
            wavelength=800e-9,
        )
    )
    return Simulation(
        grid,
        mode,
        responses,
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=2.2e-6),
        settings=SimulationSettings(
            controller=PIController(rtol=1e-7, atol=1e-12),
            save_planes=[grid.length / 2, grid.length],
            dz_init=1e-5,
            dz_max=2e-3,
            quadrature=transverse_quadrature([mode], n_r=16),
            ionisation_field_range=None if passive else (1e7, 1e12),
            backend=selected_backend(),
            schedule=FixedSchedule(np.full(10, grid.length / 10)) if passive else None,
            stats=StatsSpec(
                names=(
                    "energy",
                    "species_ionisation",
                    "species_density",
                    "species_electron_density",
                    "species_ionisation_energy",
                    "species_peak_ionisation_rate",
                )
            ),
        ),
    )


def assert_saved_accounting(simulation, output):
    problem = simulation.translate()
    grid = simulation.grid
    # Each independent exposure is a non-negative trapezoid sum. The backend
    # recurrence and this cumulative sum differ by O(n_to eps), followed by exp.
    bound = (64 * grid.n_to + 128 * problem.transform.n_nodes) * np.finfo(float).eps
    assert bound > 0
    weights_at_planes = np.stack(
        [power_weight_at(problem, float(z)) for z in output["z"]]
    )
    spectra = output.fields("fft") / np.sqrt(weights_at_planes)
    modal = fourier.inverse(
        fourier.oversample(spectra, grid.n_to, kind=grid.kind), kind=grid.kind
    )
    physical = np.einsum(
        "rm,zmpt->zrpt",
        problem.transform.synthesis / grid.pad_scale,
        modal,
    )[:, :, 0]
    weighted_modal = fourier.inverse(
        fourier.oversample(spectra * weights_at_planes, grid.n_to, kind=grid.kind),
        kind=grid.kind,
    )
    magnetic = np.einsum(
        "rm,zmpt->zrpt",
        problem.transform.synthesis / grid.pad_scale,
        weighted_modal,
    )[:, :, 0]
    fractions = processing.species_ionisation(output)
    density = output["stats/species_density"]
    inventory = processing.species_ionisation_energy(output)
    potentials = np.zeros(4)
    initial = np.zeros(4)
    for index, channel in enumerate(problem.responses.plasma_channels):
        potentials[index] = channel.ionisation_potential
        initial[index] = float(channel.preionisation(0.0))
        rates = channel.rate(np.abs(physical))
        exposure = cumulative_trapezoid(rates, dx=grid.dt_fine, axis=-1, initial=0)
        expected = initial[index] - (1 - initial[index]) * np.expm1(-exposure[..., -1])
        np.testing.assert_allclose(
            fractions[:, index], expected, rtol=bound, atol=bound
        )
        np.testing.assert_allclose(
            output["stats/species_peak_ionisation_rate"][:, index],
            rates.max(axis=(1, 2)),
            rtol=bound,
            atol=0,
        )
    weights = problem.transform.node_weights
    assert weights is not None
    expected_inventory = np.sum(fractions * weights, axis=-1) * density * potentials
    np.testing.assert_allclose(inventory, expected_inventory, rtol=bound, atol=0)
    initial_inventory = density * initial * potentials * np.sum(weights)
    new_source = density[:, :, None] * (fractions - initial[None, :, None])
    np.testing.assert_allclose(
        inventory - initial_inventory,
        np.sum(new_source * weights, axis=-1) * potentials,
        rtol=bound,
        atol=bound * np.max(inventory),
    )
    electrons = density[:, :, None] * fractions
    densest = np.argmax(electrons.sum(axis=1), axis=-1)
    expected_electrons = electrons[np.arange(output.n_saved), :, densest]
    np.testing.assert_allclose(
        output["stats/species_electron_density"], expected_electrons, rtol=bound, atol=0
    )
    np.testing.assert_array_equal(fractions[:, 2:], 0.0)
    omega, absorbed = processing.absorbed_spectral_energy(output, species="absorber")
    _, spectral = processing.get_spectral_density(output, axis="omega", band="active")
    np.testing.assert_allclose(absorbed[:, 0], 1e-4 * spectral, rtol=bound, atol=0)
    fluence = processing.node_fluence(output)
    expected_fluence = EPS0 * C * grid.dt_fine * np.sum(physical * magnetic, axis=-1)
    np.testing.assert_allclose(
        fluence, expected_fluence, rtol=bound, atol=bound * np.max(fluence)
    )
    return {
        "rounding_bound": bound,
        "omega_points": omega.size,
        "inventory_J_per_m": inventory.tolist(),
        "new_ions_per_m": np.sum(new_source * weights, axis=-1).tolist(),
    }


@pytest.mark.slow
def test_connected_air_saved_source_and_spectrum_converge(tmp_path):
    # Refining the fine temporal spacing halves the exposure trapezoid's h^2
    # defect. Compare successive differences on the unchanged coarse spectral
    # grid: the asymptotic ratio is 1/4; 1/2 allows the next-order remainder.
    # The controller's 1e-7 target gets 100 local budgets for the functional
    # comparison floor. This is a convergence test, not a physical UV error bar.
    ratio_bound = 0.5
    floor = 100 * 1e-7
    budgets: dict[str, object] = {
        "difference_ratio_bound": ratio_bound,
        "controller_floor": floor,
    }
    (tmp_path / "bounds.json").write_text(json.dumps(budgets, indent=2))
    outputs = []
    records = []
    for ffac in (6.0, 12.0, 24.0):
        simulation = connected_simulation(ffac)
        output = simulation.run()
        records.append(assert_saved_accounting(simulation, output))
        outputs.append((simulation, output))
    spectra = [
        processing.get_spectral_density(output, axis="omega", band="active")[1]
        for _, output in outputs
    ]
    sources = [np.array(record["new_ions_per_m"])[:, :2] for record in records]
    for name, values in (("spectrum", spectra), ("source", sources)):
        scale = np.linalg.norm(values[-1])
        errors = [
            float(np.linalg.norm(right - left) / scale)
            for left, right in pairwise(values)
        ]
        budgets[name] = errors
        (tmp_path / "measurements.json").write_text(
            json.dumps(budgets | {"records": records}, indent=2)
        )
        # Each refinement must at least halve the error, to a floor of 1e-5.
        assert errors[1] <= ratio_bound * errors[0] + floor, (name, errors, budgets)


@pytest.mark.slow
def test_connected_passive_phase_and_loss_match_the_operator():
    # Ten steps of a constant diagonal operator have no nonlinear truncation
    # error. Allocate 128 eps per complex product/phase reduction per step.
    bound = 128 * 10 * np.finfo(float).eps
    assert bound > 0
    simulation = connected_simulation(passive=True)
    problem = simulation.translate()
    output = simulation.run()
    # Propagation starts from the saved active-band input; Gaussian construction
    # also has tiny tails beyond that band, removed when the backend adopts it.
    expected = output.fields("fft")[0] * np.exp(
        delta_phi(problem.linear, 0.0, simulation.grid.length)
    )
    error = np.linalg.norm(output.fields("fft")[-1] - expected) / np.linalg.norm(
        expected
    )
    assert error < bound
    assert output["stats/energy"][-1].sum() < output["stats/energy"][0].sum()
