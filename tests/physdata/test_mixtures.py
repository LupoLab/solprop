# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Mixture algebra, independent material data and passive optical limits.

Bounds are declared from operation counts and the dilute square-root remainder;
none is chosen from a measured discrepancy.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from former_gas_laws import AR_KINGSTON, N2_BORZSONYI, O2_ZHANG
from numpy.testing import assert_allclose, assert_array_equal

from bounds import assert_within
from solprop import physdata
from solprop.constants import BAR, K_B, C
from solprop.exceptions import MaterialError, ModeError, TranslationError
from solprop.modes import MarcatiliMode
from solprop.modes.capillary import MarcatiliTerms
from solprop.physdata import (
    Component,
    Gas,
    IdealGas,
    IonisationData,
    Mixture,
    Species,
    TabulatedSusceptibility,
    density,
    gamma,
    species,
)
from solprop.physdata.selection import MaterialSelection
from solprop.physdata.species import OptmatGas
from solprop.profiles import Constant, DensityPath, FromCallable, Linear, Scaled, Sum
from solprop.spatial import SpatialMedium

EPS = np.finfo(float).eps
OMEGA = np.linspace(2.4e15, 3.0e15, 17)


def test_physdata_lazy_exports_keep_unknown_names_as_attribute_errors():
    assert physdata.Mixture is Mixture
    with pytest.raises(AttributeError, match="no_such_material"):
        physdata.__getattr__("no_such_material")


@pytest.fixture
def absorbing_species():
    """Synthetic smooth optical data with an exact cubic-spline derivative."""
    omega = np.linspace(2e15, 4e15, 21)
    refractivity = 1e-30 * (1 + (omega / 3e15) ** 2)
    sigma = np.full(omega.shape, 2e-24)
    model = TabulatedSusceptibility(
        omega,
        refractivity,
        sigma,
        (2 * np.pi * C / omega[-1], 2 * np.pi * C / omega[0]),
        citation="Synthetic polynomial and constant absorption for an analytic test",
    )
    return Species("synthetic absorber", model)


@pytest.mark.parametrize("gas", list(Gas))
def test_a_built_in_species_takes_optmats_material_models(gas):
    material = species(gas)
    assert material.susceptibility == gamma(gas)
    assert material.name == gas.value
    assert material.equation_of_state == OptmatGas(gas)
    assert material.equation_of_state.density(1.0, 294.0) == density(gas, 1.0, 294.0)
    assert species(material) is material
    assert material.raman is None
    # optmat's first ionisation energy, except for air, a composition.
    assert (material.ionisation is None) == (gas is Gas.Air)


def test_air_is_optmats_named_composition_at_partial_pressures():
    mixture = Mixture.air(6.0, temperature=300.0)
    assert mixture.labels == ("N2", "O2", "Ar")
    assert mixture.composition == "dry air (N2, O2, Ar)"
    assert mixture.fractions is not None
    assert_allclose(sum(mixture.fractions), 1.0, rtol=3 * EPS, atol=0)
    assert species("Air").gamma3 is None
    for component, fraction in zip(mixture.components, mixture.fractions, strict=True):
        expected = density(component.species.name, fraction * 6.0, 300.0)
        # A constant density spline is sampled at its first knot, exactly.
        assert component.density(np.array([0.0]))[0] == expected


def test_ideal_equation_of_state_and_inverse_keep_si_and_bar():
    eos = IdealGas()
    number = eos.density(2.0, 300.0)
    assert number == 2.0 * BAR / (K_B * 300.0)
    assert_allclose(eos.pressure(number, 300.0), 2.0, rtol=3 * EPS, atol=0)
    assert eos.density(0.0, 300.0) == 0.0
    with pytest.raises(MaterialError, match="density"):
        eos.pressure(-1.0, 300.0)


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_ideal_equation_of_state_rejects_nonphysical_states(value):
    eos = IdealGas()
    with pytest.raises(MaterialError, match="pressure"):
        eos.density(value, 300.0)
    with pytest.raises(MaterialError, match="temperature"):
        eos.density(1.0, value)
    with pytest.raises(MaterialError, match="temperature"):
        eos.pressure(1e24, value)


def test_absent_species_needs_no_optical_extrapolation(absorbing_species):
    mixture = Mixture.of(
        Component(species("He"), Constant(1e24)),
        Component(absorbing_species, Constant(0.0), label="absent"),
    )
    omega = np.array([4.2e15])
    mode = MarcatiliMode(125e-6, mixture=mixture)
    terms = mode.sampled_terms(omega)
    assert_array_equal(terms.core_susceptibility[1], np.zeros(1))
    assert_array_equal(terms.dcore_susceptibility_domega[1], np.zeros(1))
    assert np.all(np.isfinite(mode.neff(omega)))
    # A newly present population must again validate its spectral support.
    present = mixture.with_densities({"absent": 1e20})
    with pytest.raises(MaterialError, match="support"):
        MarcatiliMode(125e-6, mixture=present).neff(omega)


def test_sampled_mixture_terms_require_one_species_axis():
    with pytest.raises(ModeError, match="shape"):
        MarcatiliTerms(
            OMEGA,
            2.4,
            np.zeros((2, 2, OMEGA.size)),
            np.zeros((2, 2, OMEGA.size)),
            np.zeros(OMEGA.size, dtype=complex),
            np.zeros(OMEGA.size, dtype=complex),
        )


def test_tabulated_susceptibility_and_derivative_are_independently_analytic(
    absorbing_species,
):
    model = absorbing_species.susceptibility
    expected = 2e-30 * (1 + (OMEGA / 3e15) ** 2) + 1j * C * 2e-24 / OMEGA
    derivative = 4e-30 * OMEGA / 3e15**2 - 1j * C * 2e-24 / OMEGA**2
    # Polynomial reproduction costs the spline solve and Horner evaluations. With
    # 21 knots, 64 eps bounds those arithmetic operations, separately in each part.
    assert_allclose(model.susceptibility(OMEGA), expected, rtol=64 * EPS, atol=0)
    assert_allclose(
        model.dsusceptibility_domega(OMEGA).real, derivative.real, rtol=64 * EPS, atol=0
    )
    assert_allclose(
        model.dsusceptibility_domega(OMEGA).imag, derivative.imag, rtol=16 * EPS, atol=0
    )
    for omega in (0.0, 1.9e15, 4.1e15, np.inf):
        with pytest.raises(MaterialError, match="support"):
            model.susceptibility(omega)
    assert not model.omega.flags.writeable
    assert not model.refractivity.flags.writeable


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("omega", np.array([2e15, 2e15]), "increasing"),
        ("refractivity", np.ones(3), "shape"),
        ("absorption_cross_section", np.full(21, -1.0), "non-negative"),
        ("wavelength_range", (0.0, 1e-6), "positive"),
        ("wavelength_range", (1e-9, 1e-6), "beyond"),
    ],
)
def test_tabulated_data_rejects_invalid_input(absorbing_species, field, value, match):
    with pytest.raises(MaterialError, match=match):
        dataclasses.replace(absorbing_species.susceptibility, **{field: value})


def test_unavailable_custom_response_data_fails_by_species_name(absorbing_species):
    mixture = Mixture.of(Component(absorbing_species, Constant(1e24)))
    with pytest.raises(MaterialError, match="synthetic absorber"):
        mixture.gamma3_path()
    with pytest.raises(MaterialError, match="synthetic absorber"):
        species(absorbing_species, selection=MaterialSelection())
    with pytest.raises(MaterialError, match="positive"):
        IonisationData(0.0)


def test_composite_profiles_retain_breakpoints_and_derivatives():
    first = Linear(1.0, 3.0, 1.0)
    second = Linear(2.0, 0.0, 2.0)
    scaled = Scaled(first, 2.0)
    combined = Sum((scaled, second))
    z = np.linspace(-0.1, 2.1, 25)
    # A scale and a sum of the parts: two roundings, held to 4 eps.
    assert_allclose(combined(z), 2 * first(z) + second(z), rtol=4 * EPS, atol=0)
    assert_allclose(
        combined.derivative(z),
        2 * first.derivative(z) + second.derivative(z),
        rtol=4 * EPS,
        atol=0,
    )
    assert_array_equal(
        combined.breakpoints, np.unique(np.r_[first.breakpoints, second.breakpoints])
    )
    assert combined.z_span == (0.0, 2.0)
    assert combined.derivative_is_analytic
    assert not combined.is_constant
    assert Scaled(first, 0.0).is_constant
    assert Sum((Constant(1.0), Constant(2.0))).is_constant
    with pytest.raises(TranslationError, match="finite"):
        Scaled(first, np.inf)
    with pytest.raises(TranslationError, match="at least one"):
        Sum(())


def test_material_sampling_preserves_species_and_rejects_ambiguous_density():
    mixture = Mixture.fill({"He": 1.0}, 1.0)
    mode = MarcatiliMode(125e-6, mixture=mixture)
    sampled = mode.at_parameters({"density": 1e24})
    assert sampled.mixture is not None
    assert sampled.mixture["He"].species is mixture["He"].species
    assert sampled.density(np.array([0.0]))[0] == 1e24
    medium = SpatialMedium(mixture)
    frozen = medium.at_density(1e24)
    assert frozen.density(np.array([0.0]))[0] == 1e24
    with pytest.raises(MaterialError, match="unknown"):
        medium.at_parameters({"unknown": 0.0})
    with pytest.raises(MaterialError, match="carries"):
        SpatialMedium(mixture, pressure=2.0)
    air = Mixture.air(1.0)
    shared = SpatialMedium(air).at_parameters({"pressure": 2.0})
    # The same index reached by sampling and by refilling: the one-evaluation 16 eps.
    assert_within(
        shared.index(OMEGA),
        SpatialMedium(air.with_fill(pressure=2.0)).index(OMEGA),
        16 * EPS,
    )
    independent = SpatialMedium(air.with_densities({"N2": 1e24}))
    frozen_independent = independent.at_parameters({"density:N2": 2e24})
    assert isinstance(frozen_independent.material, Mixture)
    assert frozen_independent.material["N2"].density(np.array([0.0]))[0] == 2e24
    with pytest.raises(MaterialError, match="labelled"):
        independent.at_density(1e24)
    with pytest.raises(ModeError, match="every sampled species"):
        mode.neff_from_terms(mode.sampled_terms(OMEGA), 125e-6, np.array([1e24, 1e24]))


def test_sharp_pressure_interpolant_keeps_vacuum_and_endpoint_continuation(
    absorbing_species,
):
    # An interpolant may overshoot between declared non-negative boundary data.
    sharp = FromCallable(
        lambda z: 1.0 - 8.0 * z * (1.0 - z),
        (0.0, 1.0),
        np.array([0.0, 1.0]),
        slope=lambda z: 16.0 * z - 8.0,
    )
    path = DensityPath(absorbing_species, sharp, 300.0)
    positions = np.linspace(-0.2, 1.2, 101)
    assert np.all(path(positions) >= 0.0)
    assert path(np.array([0.5]))[0] == 0.0
    assert_array_equal(path(np.array([-0.2, 1.2])), path(np.array([0.0, 1.0])))
    assert_array_equal(path.derivative(np.array([-0.2, 0.5, 1.2])), np.zeros(3))
    zero = DensityPath(absorbing_species, Constant(0.0))
    assert_array_equal(zero(positions), np.zeros_like(positions))


@pytest.mark.parametrize("model", ["full", "reduced"])
@pytest.mark.parametrize("fill", [1.0, Linear(2.0, 0.1, 1.0)])
def test_one_component_mode_reduces_to_the_single_gas(model, fill):
    # The mixture route sums one population where the single gas evaluates it
    # directly: the same closed form, held to the one-evaluation budget of 16 eps
    # (accuracy.md section 7), normwise.
    old = MarcatiliMode(125e-6, "Ar", fill, cladding=1.45, model=model)
    mixture = Mixture.fill({"Ar": 1.0}, fill)
    new = MarcatiliMode(125e-6, mixture=mixture, cladding=1.45, model=model)
    assert tuple(old.parameter_paths) == tuple(new.parameter_paths)
    for z in (-0.1, 0.0, 0.35, 1.0, 1.1):
        assert_within(new.neff(OMEGA, z), old.neff(OMEGA, z), 16 * EPS)
        assert_within(new.dneff_domega(OMEGA, z), old.dneff_domega(OMEGA, z), 16 * EPS)


def test_split_zero_and_permutation_respect_population_order(absorbing_species):
    argon = species("Ar")
    first = Component(argon, Constant(2e25), "first")
    second = Component(argon, Constant(2e25), "second")
    split = Mixture.of(first, second)
    pure = Mixture.of(Component(argon, Constant(4e25)))
    z = np.array([0.0, 0.5])
    # Two halves summed against the whole: a few roundings, held to 4 eps.
    assert_allclose(
        split.susceptibility(OMEGA, z), pure.susceptibility(OMEGA, z), rtol=4 * EPS
    )
    # A zero-density population adds an exact zero: an explicit no-op.
    zero = Mixture.of(*split.components, Component(absorbing_species, Constant(0.0)))
    assert_array_equal(zero.susceptibility(OMEGA, z), split.susceptibility(OMEGA, z))
    air = Mixture.air(1.0)
    permuted = Mixture.of(*reversed(air.components))
    assert permuted.labels == tuple(reversed(air.labels))
    assert_allclose(
        permuted.susceptibility(OMEGA, z),
        air.susceptibility(OMEGA, z),
        rtol=6 * EPS,
        atol=0,
    )
    assert split.labels == ("first", "second")
    with pytest.raises(MaterialError, match="unique"):
        Mixture.of(first, first)


def test_absorption_first_order_limit_and_total_loss_controls(absorbing_species):
    rho = 1e23
    medium = Mixture.of(Component(absorbing_species, Constant(rho)))
    mode = MarcatiliMode(1.0, mixture=medium, cladding=1.45)
    empty = MarcatiliMode(1.0, cladding=1.45)
    alpha = 2 * OMEGA / C * (np.imag(mode.neff(OMEGA)) - np.imag(empty.neff(OMEGA)))
    expected = np.full(OMEGA.shape, rho * 2e-24)
    chi = medium.susceptibility(OMEGA, np.array([0.0]))[0]
    # sqrt(1+chi): the relative correction to Im(chi)/2 is bounded by |chi|
    # for |chi| < 1/2. The 1 m core contributes <1e-12 relative waveguide correction.
    bound = float(np.max(np.abs(chi))) + 1e-12 + 32 * EPS
    assert_allclose(alpha, expected, rtol=bound, atol=0)
    lossless = dataclasses.replace(mode, loss=False)
    assert_array_equal(np.imag(lossless.neff(OMEGA)), np.zeros(OMEGA.shape))
    assert_allclose(
        np.imag(dataclasses.replace(mode, loss=0.5).neff(OMEGA)),
        np.imag(mode.neff(OMEGA)) / 2,
        rtol=0,
        atol=0,
    )


def test_fill_updates_and_parameter_samples_share_material_data():
    pressure = Linear(3.0, 1.0, 1.0)
    mixture = Mixture.fill({"He": 0.7, "Ar": 0.3}, pressure)
    updated = mixture.with_fill({"He": 0.6, "Ar": 0.4}, 2.0)
    assert updated.labels == mixture.labels
    assert all(
        a.species is b.species
        for a, b in zip(mixture.components, updated.components, strict=True)
    )
    sampled = mixture.at_parameters({"pressure": 2.0})
    # The pressure path's midpoint is the sampled 2 bar to a rounding or two.
    assert_allclose(
        sampled.density(np.array([0.0]))[:, 0],
        mixture.density(np.array([0.5]))[:, 0],
        rtol=4 * EPS,
        atol=0,
    )
    independent = mixture.with_densities({"He": 0.0})
    assert independent.shared is None
    assert independent["Ar"] is mixture["Ar"]
    assert independent["He"].density(np.array([0.0]))[0] == 0.0
    assert json.loads(json.dumps(mixture.serialise()))["labels"] == ["He", "Ar"]
    with pytest.raises(MaterialError, match="length"):
        Mixture.fill({"He": 1.0}, (2.0, 1.0))
    fill = Mixture.fill({"He": 1.0}, (2.0, 1.0), length=0.5)
    assert fill.shared is not None
    assert fill.shared.z_span == (0.0, 0.5)


def test_spatial_single_species_and_mixture_parameters():
    fill = Linear(2.0, 1.0, 1.0)
    old = SpatialMedium("Ar", fill)
    new = SpatialMedium(Mixture.fill({"Ar": 1.0}, fill))
    # A one-gas mixture and the gas itself: the one-evaluation 16 eps, normwise.
    assert_within(new.index(OMEGA, 0.3), old.index(OMEGA, 0.3), 16 * EPS)
    assert_within(
        new.dindex_domega(OMEGA, 0.3), old.dindex_domega(OMEGA, 0.3), 16 * EPS
    )
    mixture = Mixture.fill({"He": 0.5, "Ar": 0.5}, fill)
    mode = MarcatiliMode(125e-6, mixture=mixture, cladding=1.45)
    assert tuple(mode.parameter_paths) == ("core_radius", "pressure")
    assert tuple(SpatialMedium(mixture).parameter_paths) == ("pressure",)
    frozen = mode.at_parameters({"pressure": 1.5})
    assert_within(frozen.neff(OMEGA), mode.neff(OMEGA, 0.5), 16 * EPS)
    independent = mixture.with_densities({})
    assert tuple(SpatialMedium(independent).parameter_paths) == (
        "density:He",
        "density:Ar",
    )
    with pytest.raises(ModeError, match="mixture"):
        MarcatiliMode(125e-6, "He", mixture=mixture)


# Luna's laws for its N2, O2 and Ar, the inputs of this witness: the published fits
# solprop's former registry carried (dc4ca0d), supplied as a caller's own data
# (tests/physdata/former_gas_laws.py), each over optmat's equation of state.
LUNA_AIR = {
    law.gas.value: Species(law.gas.value, law, OptmatGas(law.gas))
    for law in (N2_BORZSONYI, O2_ZHANG, AR_KINGSTON)
}


def test_luna_mixture_susceptibility_witness():
    witness = json.loads(
        (Path(__file__).parents[1] / "data/luna_mixture.json").read_text()
    )
    assert len(witness["provenance"]["commit"]) == 40
    assert witness["provenance"]["checkout_status"] == ""
    emitted = witness["emitted"]
    omega = 2 * np.pi * emitted["CONSTANTS"][0] / np.array(emitted["WAVELENGTH_M"])
    for case in range(1, 4):
        pressures = dict(
            zip(("N2", "O2", "Ar"), emitted[f"PRESSURE_{case}"], strict=True)
        )
        mixture = Mixture.partial_pressures(
            {LUNA_AIR[name]: pressure for name, pressure in pressures.items()},
            temperature=300.0,
        )
        chi = mixture.susceptibility(omega, np.array([0.0]))[0]
        index = np.array(emitted[f"INDEX_{case}"])
        # The faithful-model bound of tests/physdata/test_luna_reference.py is 1e-11
        # in refractivity: optmat's equations of state differ from Luna's CoolProp by
        # a constant ratio per gas, which cancels against each law's reference
        # density. Recovering
        # chi from Luna's stored index adds at most 4 eps / min|chi|, and the
        # three-component sum adds six rounded operations.
        bound = 1e-11 + 4 * EPS / float(np.min(np.abs(chi))) + 6 * EPS
        assert_allclose(chi, index**2 - 1.0, rtol=bound, atol=0)


@pytest.mark.parametrize("fractions", [{}, {"Ar": -1.0}, {"Ar": np.nan}, {"Ar": 0.9}])
def test_invalid_fill_fractions_are_rejected(fractions):
    with pytest.raises(MaterialError, match="fractions"):
        Mixture.fill(fractions, 1.0)


@pytest.mark.parametrize("pressure", [-1.0, np.nan, (1.0, 2.0, 3.0)])
def test_invalid_fill_pressures_are_rejected(pressure):
    with pytest.raises(MaterialError, match="pressure"):
        Mixture.fill({"Ar": 1.0}, pressure)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"label": "bad:label"}, "label"),
        ({"preionisation": -0.1}, "preionisation"),
        ({"preionisation": np.nan}, "preionisation"),
        ({"density": Constant(-1.0)}, "density"),
        ({"density": Constant(np.nan)}, "density"),
    ],
)
def test_component_rejects_invalid_state(options, match):
    arguments = {"density": Constant(0.0), **options}
    with pytest.raises(MaterialError, match=match):
        Component(species("Ar"), **arguments)


def test_shared_partial_pressures_total_density_and_kerr_composition():
    partial = Linear(1.0, 2.0, 0.2)
    # A caller's fixed gamma3 (test data, of the gases' order of magnitude) needs no
    # carrier; optmat's Kerr model is evaluated at one.
    nitrogen = Species("N2", gamma("N2"), OptmatGas("N2"), gamma3=9.4e-64)
    argon = Species("Ar", gamma("Ar"), OptmatGas("Ar"), gamma3=1.2e-63)
    mixture = Mixture.partial_pressures({nitrogen: partial, argon: partial})
    assert mixture.shared is not None
    positions = np.array([0.0, 0.1, 0.2])
    # Sums and products of two populations: a few roundings each, held to 4 eps.
    assert_allclose(
        mixture.shared(positions), 2 * partial(positions), rtol=4 * EPS, atol=0
    )
    densities = mixture.density(positions)
    assert_allclose(
        mixture.total_density(positions),
        densities[0] + densities[1],
        rtol=4 * EPS,
        atol=0,
    )
    assert_allclose(
        mixture.gamma3_path()(positions),
        densities[0] * mixture["N2"].species.gamma3
        + densities[1] * mixture["Ar"].species.gamma3,
        rtol=4 * EPS,
        atol=0,
    )
    sampled = mixture.at_parameters({"pressure": 0.0})
    assert_array_equal(sampled.total_density(positions), np.zeros(3))
    assert_array_equal(
        sampled.susceptibility(OMEGA, positions), np.zeros((3, OMEGA.size))
    )
    assert sampled.serialise()["shared_pressure"] is None
    staged = Mixture.fill({"Ar": 1.0}, ([0.0, 0.1, 0.2], [1.0, 1.5, 2.0]))
    assert_array_equal(staged.total_density(positions), staged.density(positions)[0])
    assert_array_equal(staged.with_fill().density(positions), staged.density(positions))


def test_mixture_configuration_and_updates_refuse_unknown_labels():
    component = Component(species("Ar"), Constant(1e25))
    mixture = Mixture.of(component)
    for action, match in (
        (Mixture.of, "at least one"),
        (lambda: Mixture.of(component, temperature=0.0), "temperature"),
        (lambda: Mixture((component,), fractions=(1.0,)), "fractions"),
        (lambda: mixture["missing"], "missing"),
        (lambda: mixture.with_densities({"missing": 0.0}), "missing"),
        (lambda: mixture.with_fill({"missing": 1.0}, 1.0), "exactly"),
        (mixture.with_fill, "fractions and pressure"),
        (lambda: mixture.at_parameters({"pressure": 1.0}), "shared fill"),
        (lambda: mixture.at_parameters({"invalid": 1.0}), "invalid"),
    ):
        with pytest.raises(MaterialError, match=match):
            action()
    independent = mixture.at_parameters({"density:Ar": 2e25})
    assert independent["Ar"].density(np.array([0.0]))[0] == 2e25
    fill = Mixture.fill({"Ar": 1.0}, 1.0)
    with pytest.raises(MaterialError, match="sampled pressure"):
        fill.at_parameters({"pressure": -1.0})
    with pytest.raises(MaterialError, match="shared fill"):
        fill.at_parameters({"pressure": 1.0, "density:Ar": 0.0})
    unspecified = Mixture((component,), shared=Constant(1.0))
    with pytest.raises(MaterialError, match="declared fill fractions"):
        unspecified.at_parameters({"pressure": 1.0})
    explicit = Mixture((component,), shared=Constant(1.0), fractions=(1.0,))
    frozen = explicit.at_parameters({"pressure": 2.0})
    assert frozen["Ar"].density(np.array([0.0]))[0] == density("Ar", 2.0)


def test_species_metadata_validation_and_inverse_eos(absorbing_species):
    for options, match in (({"name": " "}, "name"), ({"gamma3": np.nan}, "gamma3")):
        with pytest.raises(MaterialError, match=match):
            dataclasses.replace(absorbing_species, **options)
    with pytest.raises(MaterialError, match="finite"):
        dataclasses.replace(
            absorbing_species.susceptibility, refractivity=np.full(21, np.nan)
        )
    assert absorbing_species.susceptibility.source == "tabulated"
    material = species("Ar")
    number = material.equation_of_state.density(1.0, 300.0)
    # The nonlinear EOS inversion is the separately qualified pure-species routine.
    assert_allclose(
        material.equation_of_state.pressure(number, 300.0), 1.0, rtol=1e-10, atol=0
    )


def test_air_without_argon_is_the_same_fractions_renormalised():
    two = Mixture.air(1.0, argon=False)
    three = Mixture.air(1.0)
    assert two.labels == ("N2", "O2")
    assert two.composition == "dry air (N2, O2)"
    assert two.fractions is not None and three.fractions is not None
    # optmat's two named compositions: the same N2 : O2 ratio, renormalised.
    renormalised = np.array(three.fractions[:2]) / sum(three.fractions[:2])
    assert_allclose(two.fractions, renormalised, rtol=1e-12)
    assert_allclose(sum(two.fractions), 1.0, rtol=3 * EPS, atol=0)
    # Given fractions win over the switch.
    assert Mixture.air(1.0, fractions={"N2": 0.5, "O2": 0.5}, argon=False).labels == (
        "N2",
        "O2",
    )
