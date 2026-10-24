# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify the labelled material vocabulary at the public driver boundary."""

from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pytest

from bounds import step_evaluations
from former_raman import FORMER_N2
from mixture_cases import AnalyticRate, mixture_case
from selected_backend import selected_backend
from solprop.constants import K_B
from solprop.driver import Simulation, prop_capillary
from solprop.driver.mixtures import is_air
from solprop.exceptions import DriverError, MaterialError
from solprop.grid import EnvelopeGrid
from solprop.modes import MarcatiliMode
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.nonlinear.responses import PlasmaResponse
from solprop.physdata import Gas, Mixture
from solprop.physdata.raman_types import Dephasing, Linewidth
from solprop.physdata.sellmeier import gamma
from solprop.processing import species_ionisation, species_ionisation_energy
from solprop.profiles import Constant
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule


@pytest.mark.parametrize("material", [Gas.Air, "air", " AIR ", Gas.He, "He"])
def test_air_selection_accepts_the_enum_and_normalised_name(material):
    assert is_air(material) == (material in (Gas.Air, "air", " AIR "))


COLLAR_STEPS = 1600
"""Replayed steps under the in-stage collar's limit for the default integrator."""


@pytest.fixture(scope="module")
def material_grid():
    # No initial fraction: most tests switch one population's plasma off, and a
    # declared preionisation without a plasma channel is refused.
    mixture, grid, _, _, _ = mixture_case(preionisation=(0.0, 0.0))
    return mixture, grid


def construct(mixture, grid, **options):
    return Simulation.from_capillary(
        30e-6,
        grid.length,
        mixture,
        **(
            {
                "lambda0": 800e-9,
                "grid": grid,
                "thg": False,
                "fwhm": 10e-15,
                "energy": 1e-9,
                "n_r": 8,
                "n_save": 2,
                "plasma": {mixture.labels[0]: AnalyticRate()},
                "loss": False,
                "cladding": 1.45,
                "ionisation_field_range": (1e7, 1e11),
            }
            | options
        ),
    )


@pytest.mark.parametrize("pressure", [1.0, (1.0, 0.0), Constant(1.0)])
def test_mixture_refuses_every_explicit_pressure(material_grid, pressure):
    mixture, grid = material_grid
    with pytest.raises(DriverError, match=r"Mixture.*pressure"):
        construct(mixture, grid, pressure=pressure)


@pytest.mark.parametrize("fractions", [False, True])
def test_mixture_optics_conflict_is_explicit(material_grid, fractions):
    mixture, grid = material_grid
    material = {"He": 0.5, "Ar": 0.5} if fractions else mixture
    with pytest.raises(DriverError, match="susceptibility"):
        Simulation.from_capillary(
            30e-6,
            grid.length,
            material,
            lambda0=800e-9,
            grid=grid,
            fwhm=10e-15,
            energy=1e-9,
            plasma=False,
            susceptibility=gamma("He"),
        )


def test_unsupported_supplied_mode_fails_by_geometry_name(material_grid):
    mixture, grid = material_grid
    mode = SpecifiedMode.from_taylor([0.0], omega_ref=grid.omega0)
    with pytest.raises(DriverError, match="SpecifiedMode"):
        construct(mixture, grid, modes=[mode])


def test_shared_to_independent_update_preserves_pressure_statistics(material_grid):
    material, grid = material_grid
    fill = material.with_fill(
        dict(zip(material.labels, (0.5, 0.5), strict=True)), pressure=2.0
    )
    simulation = construct(fill, grid)
    updated = simulation.with_mixture(material)
    assert updated.settings is simulation.settings
    z = np.array([0.0, grid.length / 2, grid.length])
    # Both custom species have ideal EOS. Summing two independently converted
    # partials costs less than eight eps relative to converting their density sum.
    bound = 8 * np.finfo(float).eps
    expected = material.total_density(z) * K_B * material.temperature / 1e5
    np.testing.assert_allclose(
        updated.profiles["pressure"](z), expected, rtol=bound, atol=0
    )


def test_changed_linewidth_shares_undamped_raman_samples(material_grid):
    material, grid = material_grid
    first, second = material.components
    # solprop's former N2 constants, supplied (former_raman: dc4ca0d's registry).
    first = replace(first, species=replace(first.species, raman=FORMER_N2))
    material = Mixture.of(first, second)
    source = construct(material, grid, raman=True)
    original = source.translate().responses.raman_terms[0]
    changed = replace(
        material,
        components=(
            replace(first, dephasing=Dephasing(Linewidth(1e12))),
            second,
        ),
    )
    result = source.with_mixture(changed).translate().responses.raman_terms[0]
    assert result.undamped is original.undamped
    assert result.decay_rates is not original.decay_rates
    assert any(
        float(left(0.0)) != float(right(0.0))
        for left, right in zip(original.decay_rates, result.decay_rates, strict=True)
    )


def test_analytic_plasma_with_temporal_collar_matches_the_reference(material_grid):
    material, grid = material_grid
    # The in-stage collar is stepped by the default Adams history only while
    # h alpha / 2 stays below 0.0877: on this collar, steps under 7.0e-7 m.
    steps = COLLAR_STEPS
    outputs = []
    for backend in ("numpy_ref", selected_backend()):
        simulation = construct(
            material,
            grid,
            backend=backend,
            schedule=FixedSchedule(np.full(steps, grid.length / steps)),
            stats=StatsSpec(names=("energy", "species_ionisation")),
        )
        # One exposure/current pair and the temporal absorber enter each step
        # through its evaluations' weights. Allocate 64 eps per fine sample and 128
        # per radial node, following the channel ladder.
        evaluations = step_evaluations(simulation.translate())
        bound = (64 * grid.n_to + 128 * 8) * evaluations * steps * np.finfo(float).eps
        outputs.append(simulation.run())
    fields = [output.fields("fft") for output in outputs]
    relative = np.linalg.norm(fields[1] - fields[0]) / np.linalg.norm(fields[0])
    assert relative < bound
    np.testing.assert_allclose(
        outputs[1]["stats/species_ionisation"],
        outputs[0]["stats/species_ionisation"],
        rtol=bound,
        atol=bound,
    )


def test_mixture_refuses_pressure_floor(material_grid):
    mixture, grid = material_grid
    with pytest.raises(DriverError, match="pressure_floor"):
        construct(mixture, grid, pressure_floor=0.0)


@pytest.mark.parametrize("option", ["plasma", "raman", "preionisation"])
def test_unknown_component_selection_fails_by_name(material_grid, option):
    mixture, grid = material_grid
    with pytest.raises((DriverError, MaterialError), match="misspelled"):
        construct(mixture, grid, **{option: {"misspelled": False}})


def test_preionisation_preserves_omitted_components(material_grid):
    _, grid = material_grid
    mixture, _, _, _, _ = mixture_case()
    both = {label: AnalyticRate() for label in mixture.labels}
    simulation = construct(
        mixture, grid, plasma=both, preionisation={mixture.labels[1]: 0.2}
    )
    assert isinstance(simulation.modes[0], MarcatiliMode)
    material = simulation.modes[0].mixture
    assert material is not None
    assert material.components[0].preionisation == mixture.components[0].preionisation
    assert material.components[1].preionisation == 0.2
    carried = {
        one.species: one.preionisation
        for one in simulation.responses
        if isinstance(one, PlasmaResponse)
    }
    assert carried == {
        mixture.labels[0]: mixture.components[0].preionisation,
        mixture.labels[1]: 0.2,
    }


def test_a_preionisation_no_plasma_channel_carries_is_refused_by_label(material_grid):
    # The components of this mixture carry 0.03 and 0.1; only the first has plasma.
    _, grid = material_grid
    mixture, _, _, _, _ = mixture_case()
    with pytest.raises(DriverError, match="declared for 'population-1', which has no"):
        construct(mixture, grid)
    zeroed, _, _, _, _ = mixture_case(preionisation=(0.03, 0.0))
    with pytest.raises(DriverError, match="declared for 'population-1', which has no"):
        construct(zeroed, grid, preionisation={zeroed.labels[1]: 0.2})
    assert construct(zeroed, grid).modes


@pytest.mark.parametrize("fraction", [-0.1, 1.1, np.nan])
def test_invalid_initial_fraction_is_refused(material_grid, fraction):
    mixture, grid = material_grid
    with pytest.raises(MaterialError, match="preionisation"):
        construct(mixture, grid, preionisation=fraction)


def test_per_label_rates_and_inactive_species_retain_order(material_grid):
    mixture, grid = material_grid
    simulation = construct(mixture, grid, plasma={mixture.labels[1]: AnalyticRate()})
    problem = simulation.translate()
    assert problem.responses.labels == mixture.labels
    assert tuple(channel.label for channel in problem.responses.plasma_channels) == (
        mixture.labels[1],
    )
    assert simulation.settings.stats is not None
    assert "pressure" not in simulation.profiles
    assert "pressure" not in simulation.settings.stats.names
    assert {
        "species_ionisation",
        "species_electron_density",
        "species_peak_ionisation_rate",
        "species_density",
        "species_ionisation_energy",
    } <= set(simulation.settings.stats.names)


def test_prebuilt_modes_keep_geometry_under_the_declared_material(material_grid):
    mixture, grid = material_grid
    original = construct(mixture, grid).modes[0]
    assert isinstance(original, MarcatiliMode)
    rotated = replace(original, phi=0.0, model="reduced")
    simulation = construct(mixture, grid, modes=[rotated])
    assert isinstance(simulation.modes[0], MarcatiliMode)
    assert simulation.modes[0].model == "reduced"
    assert simulation.modes[0].radius == original.radius
    assert simulation.modes[0].mixture is mixture


def test_fraction_mapping_uses_total_pressure(material_grid):
    _, grid = material_grid
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        simulation = Simulation.from_capillary(
            30e-6,
            grid.length,
            {"He": 0.75, "Ar": 0.25},
            2.0,
            lambda0=800e-9,
            grid=grid,
            thg=False,
            fwhm=10e-15,
            energy=1e-9,
            plasma=False,
            n_r=8,
            n_save=2,
            loss=False,
            cladding=1.45,
        )
        simulation.translate()
    assert isinstance(simulation.modes[0], MarcatiliMode)
    assert simulation.modes[0].mixture is not None
    reference = Mixture.fill({"He": 0.75, "Ar": 0.25}, 2.0, length=grid.length)
    np.testing.assert_array_equal(
        simulation.modes[0].mixture.density(np.array([0.0, grid.length])),
        reference.density(np.array([0.0, grid.length])),
    )


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu"])
def test_public_propagation_accepts_mixture_and_periodic_species(
    material_grid, backend
):
    mixture, grid = material_grid
    stats = StatsSpec(
        names=(
            "energy",
            "species_ionisation",
            "species_electron_density",
            "species_density",
            "species_ionisation_energy",
        ),
        periodic=("species_ionisation", "species_electron_density"),
    )
    output = prop_capillary(
        30e-6,
        grid.length,
        mixture,
        lambda0=800e-9,
        grid=grid,
        thg=False,
        fwhm=10e-15,
        energy=1e-9,
        plasma={mixture.labels[0]: AnalyticRate()},
        n_r=8,
        n_save=2,
        loss=False,
        cladding=1.45,
        backend=backend,
        ionisation_field_range=(1e7, 1e11),
        stats=stats,
        # Forty steps are far beyond the in-stage collar's limit under the default
        # Adams history (7.0e-7 m); split off, it is integrated exactly.
        absorber_scheme="split_step",
        schedule=FixedSchedule(np.full(40, grid.length / 40)),
    )
    assert output["stats/species_ionisation"].shape[1] == 2
    assert output["stats/species_density"].shape == (2, 2)
    assert "species_electron_density" in output.step_stat_names
    np.testing.assert_array_equal(
        species_ionisation(output), output["stats/species_ionisation"]
    )
    np.testing.assert_array_equal(
        species_ionisation_energy(output), output["stats/species_ionisation_energy"]
    )


def test_public_update_retains_rate_tables_and_the_field_range_setting(material_grid):
    mixture, grid = material_grid
    source = construct(mixture, grid)
    original = source.translate()
    changed = mixture.with_densities({mixture.labels[0]: Constant(1.2e24)})
    updated = source.with_mixture(changed)
    assert updated.settings is source.settings
    assert (
        updated.translate().responses.plasma_channels[0].rate
        is original.responses.plasma_channels[0].rate
    )
    # A density change preserves true-power input, not its internal vacuum units.
    np.testing.assert_array_equal(updated._field0, source._field0)
    assert not np.array_equal(
        updated.translate().initial.field0, original.initial.field0
    )


def test_changed_temperature_rebuilds_rotational_populations(material_grid, caplog):
    material, grid = material_grid
    first, second = material.components
    # solprop's former N2 constants, supplied (former_raman: dc4ca0d's registry).
    first = replace(first, species=replace(first.species, raman=FORMER_N2))
    material = Mixture.of(first, second)
    source = construct(material, grid, raman=True)
    original = source.translate().responses.raman_terms[0]
    with caplog.at_level("INFO", logger="solprop"):
        updated = source.with_mixture(replace(material, temperature=350.0))
    assert (
        updated.translate().responses.raman_terms[0].undamped is not original.undamped
    )
    assert "undamped Raman model" in caplog.text


def test_zero_supplied_mode_count_is_refused(material_grid):
    material, grid = material_grid
    with pytest.raises(DriverError, match="at least one mode"):
        construct(material, grid, modes=0)


def test_multimode_update_keeps_the_shared_material_and_transform(material_grid):
    material, grid = material_grid
    source = construct(material, grid, modes=2, n_r=16)
    original = source.translate()
    updated = source.with_mixture(
        material.with_densities({material.labels[0]: Constant(1.5e24)})
    )
    assert updated.translate().transform is original.transform
    assert len(updated.modes) == 2


def test_envelope_mixture_temporal_collar_matches_reference(material_grid):
    material, original = material_grid
    grid = EnvelopeGrid.from_limits(original.length, 800e-9, (400e-9, 2e-6), 80e-15)
    steps = COLLAR_STEPS
    fields = []
    for backend in ("numpy_ref", selected_backend()):
        simulation = construct(
            material,
            grid,
            plasma=False,
            ionisation_field_range=None,
            backend=backend,
            stats=StatsSpec(names=("energy",)),
            schedule=FixedSchedule(np.full(steps, grid.length / steps)),
        )
        evaluations = step_evaluations(simulation.translate())
        bound = (64 * grid.n_to + 128 * 8) * evaluations * steps * np.finfo(float).eps
        fields.append(simulation.run().fields("fft"))
    assert np.linalg.norm(fields[1] - fields[0]) / np.linalg.norm(fields[0]) < bound


def test_air_enum_and_string_names_select_the_same_material():
    assert is_air(Gas.Air) and is_air(" AIR ")
    assert not is_air(Gas.Ar)
