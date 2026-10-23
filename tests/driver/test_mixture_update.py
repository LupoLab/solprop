# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check immutable sharing and fresh-construction agreement after material updates."""

import logging
from dataclasses import replace

import numpy as np
import pytest

from former_raman import FORMER_SILICA_LINES
from mixture_cases import mixture_case
from solprop.constants import C
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import DriverError
from solprop.grid import RadialGrid
from solprop.nonlinear.raman import (
    DampedOscillator,
    IntermediateBroadening,
    RamanResponse,
    undamped_dephasing,
)
from solprop.nonlinear.responses import PlasmaResponse
from solprop.physdata import Mixture
from solprop.profiles import Constant, Linear
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.translate.linop import delta_phi
from solprop.translate.responses import prefactor_at


@pytest.fixture
def source():
    mixture, grid, mode, responses, problem = mixture_case()
    mixture = mixture.with_densities(
        {
            label: Linear((i + 1) * 1e24, (i + 2) * 1e24, grid.length)
            for i, label in enumerate(mixture.labels)
        }
    )
    raman = RamanResponse(
        models=(DampedOscillator(1e-37, 2e14, undamped_dephasing()),),
        thg=False,
        species=mixture.labels[0],
    )
    return Simulation(
        grid,
        [replace(mode, mixture=mixture)],
        [*responses, raman],
        problem.initial.field0[0],
        settings=SimulationSettings(
            controller=problem.stepper.controller,
            save_planes=[grid.length],
            dz_max=1e-4,
            dz_init=1e-5,
        ),
    )


def fresh(source, mixture):
    return Simulation(
        source.grid,
        [replace(source.modes[0], mixture=mixture)],
        source.responses,
        source.initial,
        settings=source.settings,
        noise=source.noise,
    )


def test_update_shares_independent_data_and_matches_fresh(source):
    original = source.translate()
    material = source.modes[0].mixture
    mixture = material.with_densities(
        {
            label: Linear((i + 1.2) * 1e24, (i + 1.8) * 1e24, source.grid.length)
            for i, label in enumerate(material.labels)
        }
    )
    # Both interpolants target 1e-12; reference subtraction and basis summation
    # receive 512 eps per coefficient at the original optical wave-number
    # scale, bounded by omega/c for this dilute synthetic medium. Multiply by
    # length for a phase bound; the prefactor needs no reference subtraction.
    phi_terms = original.linear.phi_expansion.n_terms
    bound = (
        2
        * source.grid.length
        * np.max(source.grid.omega)
        / C
        * (1e-12 + 512 * phi_terms * np.finfo(float).eps)
    )
    prefactor_bound = 2e-12 + 512 * phi_terms * np.finfo(float).eps
    assert bound > 0 and prefactor_bound > 0
    updated = source.with_mixture(mixture)
    problem = updated.translate()
    reference = fresh(source, mixture).translate()
    assert updated.grid is source.grid
    assert updated.settings is source.settings
    assert updated.initial is source.initial
    assert updated.noise is source.noise
    assert problem.transform is original.transform
    # The user launch is preserved; a changed entrance index rescales a.
    assert problem.initial.field0 is not original.initial.field0
    assert problem.linear.phi_expansion is not None
    assert original.linear.phi_expansion is not None
    assert problem.linear.phi_expansion.terms is original.linear.phi_expansion.terms
    assert problem.responses.prefactor_expansion is not None
    assert original.responses.prefactor_expansion is not None
    assert (
        problem.responses.prefactor_expansion.terms
        is original.responses.prefactor_expansion.terms
    )
    assert (
        problem.responses.raman_terms[0].undamped
        is original.responses.raman_terms[0].undamped
    )
    for old, new in zip(
        original.responses.plasma_channels,
        problem.responses.plasma_channels,
        strict=True,
    ):
        assert new.rate is old.rate
        assert new.density is not old.density
    for z in np.linspace(0, source.grid.length, 13)[1:]:
        left = delta_phi(problem.linear, 0.0, float(z))
        right = delta_phi(reference.linear, 0.0, float(z))
        np.testing.assert_allclose(left, right, rtol=0, atol=bound)
        left = prefactor_at(problem.responses, float(z))
        right = prefactor_at(reference.responses, float(z))
        np.testing.assert_allclose(
            left, right, rtol=0, atol=prefactor_bound * np.max(np.abs(right))
        )


@pytest.mark.parametrize("change", ["labels", "species", "order"])
def test_update_refuses_material_identity_changes_by_name(source, change):
    material = source.modes[0].mixture
    first, second = material.components
    if change == "labels":
        first = replace(first, label="changed")
    elif change == "species":
        first = replace(first, species=replace(first.species, gamma3=2e-52))
    components = (second, first) if change == "order" else (first, second)
    with pytest.raises(DriverError, match=r"labels|species data"):
        source.with_mixture(Mixture.of(*components))


def test_update_outside_sampled_range_rebuilds_with_reason(source, caplog):
    original = source.translate()
    material = source.modes[0].mixture
    mixture = material.with_densities({material.labels[0]: Constant(4e24)})
    with caplog.at_level(logging.INFO, logger="solprop"):
        problem = source.with_mixture(mixture).translate()
    assert problem.linear.phi_expansion is not None
    assert original.linear.phi_expansion is not None
    assert problem.linear.phi_expansion.terms is not original.linear.phi_expansion.terms
    assert "range" in caplog.text


def test_fill_fraction_change_invalidates_the_pressure_function(source, caplog):
    material = source.modes[0].mixture
    fill = material.with_fill(
        dict(zip(material.labels, (0.5, 0.5), strict=True)),
        Linear(1.0, 3.0, source.grid.length),
    )
    simulation = fresh(source, fill)
    original = simulation.translate()
    changed = fill.with_fill(dict(zip(fill.labels, (0.4, 0.6), strict=True)))
    with caplog.at_level(logging.INFO, logger="solprop"):
        problem = simulation.with_mixture(changed).translate()
    assert problem.linear.phi_expansion is not None
    assert original.linear.phi_expansion is not None
    assert problem.linear.phi_expansion.terms is not original.linear.phi_expansion.terms
    assert "fill fractions" in caplog.text


@pytest.mark.parametrize("constant", [False, True])
def test_radial_update_shares_covered_operator_terms_and_input(source, constant):
    material = source.modes[0].mixture
    space = RadialGrid(6e-4, 8, window_factor=0.0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-9), 80e-6
    )
    simulation = Simulation.from_radial(
        source.grid,
        space,
        SpatialMedium(material),
        source.responses,
        pulse,
        settings=source.settings,
    )
    original = simulation.translate()
    changed = material.with_densities(
        {
            label: Constant((i + 1.5) * 1e24)
            if constant
            else Linear((i + 1.2) * 1e24, (i + 1.8) * 1e24, source.grid.length)
            for i, label in enumerate(material.labels)
        }
    )
    updated = simulation.with_mixture(changed)
    problem = updated.translate()
    assert updated.spatial_grid is space
    assert updated.initial is pulse
    assert problem.initial is original.initial
    assert problem.transform is original.transform
    assert problem.linear.phi_expansion is not None
    assert original.linear.phi_expansion is not None
    assert problem.linear.phi_expansion.terms is original.linear.phi_expansion.terms
    assert problem.responses.prefactor_expansion is not None
    assert original.responses.prefactor_expansion is not None
    assert (
        problem.responses.prefactor_expansion.terms
        is original.responses.prefactor_expansion.terms
    )


@pytest.mark.parametrize("geometry", ["modal", "radial"])
def test_uncovered_constant_update_falls_back_to_constant_operator(source, geometry):
    material = source.modes[0].mixture
    if geometry == "radial":
        source = Simulation.from_radial(
            source.grid,
            RadialGrid(6e-4, 8, window_factor=0.0),
            SpatialMedium(material),
            source.responses,
            SpatialPulse(
                GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-9), 80e-6
            ),
            settings=source.settings,
        )
    source.translate()
    changed = material.with_densities(
        {label: Constant(4e24) for label in material.labels}
    )
    result = source.with_mixture(changed).translate()
    assert result.linear.phi_representation == "constant"
    assert result.responses.prefactor_expansion is None


def test_fixed_density_change_invalidates_radius_expansion(source, caplog):
    material = source.modes[0].mixture.with_densities(
        {label: Constant(1e24) for label in source.modes[0].mixture.labels}
    )
    source.modes = (
        replace(
            source.modes[0],
            mixture=material,
            radius=Linear(30e-6, 35e-6, source.grid.length),
        ),
    )
    original = source.translate()
    with caplog.at_level(logging.INFO, logger="solprop"):
        result = source.with_mixture(
            material.with_densities({material.labels[0]: Constant(2e24)})
        ).translate()
    assert result.linear.phi_expansion.terms is not original.linear.phi_expansion.terms
    assert "fixed parameter" in caplog.text


def test_update_requires_a_shared_material_declaration(source):
    material = source.modes[0].mixture
    source.modes = (replace(source.modes[0], mixture=None),)
    with pytest.raises(DriverError, match="needs a Mixture"):
        source.with_mixture(material)
    source.modes = (
        replace(source.modes[0], mixture=material),
        replace(
            source.modes[0],
            m=2,
            mixture=material.with_densities({material.labels[0]: Constant(4e24)}),
        ),
    )
    with pytest.raises(DriverError, match="mode 1 carries a different mixture"):
        source.with_mixture(material)


@pytest.mark.parametrize("populations", [1, 2])
def test_anonymous_rates_resolve_unique_species_during_update(source, populations):
    material = Mixture.of(*source.modes[0].mixture.components[:populations])
    source.modes = (replace(source.modes[0], mixture=material),)
    source.responses = tuple(
        replace(response, species=None)
        for response in source.responses
        if isinstance(response, PlasmaResponse) and response.species in material.labels
    )
    original = source.translate()
    changed = material.with_densities(
        {label: Constant(1.5e24) for label in material.labels}
    )
    updated = source.with_mixture(changed).translate()
    assert (
        tuple(channel.label for channel in updated.responses.plasma_channels)
        == material.labels
    )
    assert all(
        a.rate is b.rate
        for a, b in zip(
            original.responses.plasma_channels,
            updated.responses.plasma_channels,
            strict=True,
        )
    )


def test_custom_broadened_kernel_is_shared_when_its_model_is_unchanged(source):
    material = source.modes[0].mixture
    # A caller's line set: solprop's former silica lines, supplied (former_raman).
    response = RamanResponse(
        models=(IntermediateBroadening(FORMER_SILICA_LINES, amplitude=1e-37),),
        thg=False,
        species=material.labels[0],
    )
    source.responses = (
        *(one for one in source.responses if not isinstance(one, RamanResponse)),
        response,
    )
    original = source.translate()
    updated = source.with_mixture(
        material.with_densities({material.labels[0]: Constant(1.5e24)})
    ).translate()
    assert (
        updated.responses.raman_terms[0].undamped
        is original.responses.raman_terms[0].undamped
    )


def test_input_and_mixture_updates_compose_and_preserve_rate_tables(source):
    mixture = source.modes[0].mixture
    changed = mixture.with_densities(dict.fromkeys(mixture.labels, 1.2e24))
    original = source.translate()
    field = source._field0 * 1.5
    input_first = source.with_input(field).with_mixture(changed)
    mixture_first = source.with_mixture(changed).with_input(field)
    np.testing.assert_array_equal(
        input_first.translate().initial.field0, mixture_first.translate().initial.field0
    )
    assert (
        input_first.translate().linear.phi_representation
        == mixture_first.translate().linear.phi_representation
    )
    assert source.with_input(field).translate().responses is original.responses
    assert (
        input_first.metadata().configuration_hash()
        == mixture_first.metadata().configuration_hash()
    )


def test_input_update_replaces_or_removes_constructor_pulse_recipe(source):
    previous = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-9)
    source.configuration = {"pulse": previous, "label": "kept"}
    replacement = GaussianPulse(wavelength=800e-9, fwhm=12e-15, energy=2e-9)
    updated = source.with_input(replacement)
    assert updated.configuration["pulse"] is replacement
    array_update = source.with_input(source._field0 * 2)
    assert "pulse" not in array_update.configuration
    assert array_update.configuration["label"] == "kept"
    assert source.configuration["pulse"] is previous
    assert array_update.metadata().configuration["label"] == "kept"


def test_phi_reuse_refuses_a_changed_index_by_name(source, caplog):
    from solprop.driver.simulation import _same_operator_function  # noqa: PLC0415

    original = source.translate()
    material = source.modes[0].mixture
    updated = Simulation(
        source.grid,
        source.modes,
        source.responses,
        source.initial,
        settings=replace(source.settings, prefactor_index=1.0),
    )
    with caplog.at_level(logging.INFO, logger="solprop.driver.simulation"):
        assert (
            _same_operator_function(
                source,
                updated,
                material,
                material,
                original.linear.phi_expansion,
            )
            is None
        )
    assert "prefactor_index changed" in caplog.text
