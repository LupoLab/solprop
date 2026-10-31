# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Per-label response selection and supplied, partial molecular models."""

import dataclasses
import logging

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mixture_cases import AnalyticRate, mixture_case
from solprop.exceptions import MaterialError
from solprop.nonlinear import responses_for
from solprop.nonlinear.optmat_raman import OptmatRamanResponse
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.raman import (
    RamanResponse,
    RotationalLines,
    response_for,
    undamped_dephasing,
)
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import (
    Component,
    IdealGas,
    Linewidth,
    Mixture,
    MolecularRaman,
    RotationalRaman,
    Species,
    species,
)
from solprop.physdata.raman_types import MolecularOptions
from solprop.profiles import Constant

# solprop's per-component response selection. Since SEP 3, Phase 4b every built-in gas
# is optmat's: plasma=True takes optmat's rate (OptmatRate) and a mixture of built-in
# gases one optmat Raman response for its whole composition; a caller's species keeps
# its own per-component molecular Raman model, built from the constants it supplies.

SYNTHETIC_ROTOR = RotationalRaman(
    b=199.0,
    d=5.74e-4,
    degeneracy_odd=1,
    degeneracy_even=2,
    polarisability_anisotropy=6.7e-31,
    linewidth=Linewidth(density=3.3e9),
    citation=(
        "synthetic test rotor: the constants of solprop's former N2 registry entry "
        "(dc4ca0d), not an oxygen or nitrogen dataset"
    ),
)
"""A caller's rotational constants for the synthetic single-term species below."""


def synthetic_rotor(name="synthetic rotor"):
    """A caller's species with a rotation-only molecular Raman model."""
    return dataclasses.replace(
        species("N2"), name=name, raman=MolecularRaman(name, SYNTHETIC_ROTOR, None)
    )


def test_air_defaults_select_optmat_rates_and_one_composition_raman(caplog):
    # The same policy as the single-gas capillary driver: optmat's rate for every
    # built-in gas, and one optmat Raman response for the whole composition, placed
    # at the first label (argon has no Raman response and is part of it).
    mixture = Mixture.air(1.0)
    with caplog.at_level(logging.INFO, logger="solprop"):
        responses = responses_for(mixture, plasma=True, raman=True, wavelength=800e-9)
    plasma = [r for r in responses if isinstance(r, PlasmaResponse)]
    assert [r.species for r in plasma] == ["N2", "O2", "Ar"]
    field = np.geomspace(1e8, 5e10, 100)
    for response, name in zip(plasma, ("N2", "O2", "Ar"), strict=True):
        assert isinstance(response.rate, OptmatRate)
        assert response.rate.species_identity == name
        assert_array_equal(response.rate(field), OptmatRate(name, 800e-9)(field))
    raman = [r for r in responses if isinstance(r, RamanResponse)]
    assert len(raman) == 1
    assert isinstance(raman[0], OptmatRamanResponse)
    assert raman[0].species == "N2"


@pytest.mark.parametrize(
    "selection",
    [{"empty": True}, ["empty"], {"empty": MolecularOptions(vibration=False)}],
)
def test_explicit_selection_of_a_species_without_terms_names_the_missing_data(
    selection,
):
    # A registry-shaped model with neither term and a note saying why. No registry
    # species has this shape (oxygen now has both terms), so it is synthetic.
    nitrogen = species("N2")
    empty = MolecularRaman("synthetic empty", None, None, "no data was tabulated")
    custom = dataclasses.replace(nitrogen, name="synthetic empty", raman=empty)
    mixture = Mixture.of(Component(custom, Constant(1e24), "empty"))
    with pytest.raises(MaterialError, match=r"empty.*no data was tabulated"):
        responses_for(mixture, raman=selection, wavelength=800e-9)


@pytest.mark.parametrize("option", ["plasma", "raman"])
def test_misspelled_label_is_refused(option):
    with pytest.raises(MaterialError, match="misspelled"):
        if option == "plasma":
            responses_for(Mixture.air(1.0), plasma={"misspelled": True})
        else:
            responses_for(Mixture.air(1.0), raman={"misspelled": True})


def test_custom_rotation_only_model_runs_without_inventing_oxygen_constants():
    # A labelled synthetic test model of a caller's constants (SYNTHETIC_ROTOR).
    custom = synthetic_rotor()
    component = Component(
        custom, Constant(1e24), "rotor", dephasing=undamped_dephasing()
    )
    mixture = Mixture.of(component)
    selected = responses_for(
        mixture, raman={"rotor": MolecularOptions(vibration=False)}, wavelength=800e-9
    )
    raman = next(r for r in selected if isinstance(r, RamanResponse))
    assert len(raman.models) == 1
    assert raman.models[0].dephasing is component.dephasing
    assert raman.tensor_terms[0].contraction == "rank2"
    assert len(response_for(custom).models) == 1
    with pytest.raises(MaterialError, match=r"synthetic rotor.*vibrational"):
        responses_for(
            mixture,
            raman={"rotor": MolecularOptions(rotation=False)},
            wavelength=800e-9,
        )


def test_per_label_rates_and_preionisation_use_the_component_population():
    mixture = mixture_case()[0]
    rate = AnalyticRate()
    responses = responses_for(mixture, plasma={"population-1": rate}, kerr=False)
    assert len(responses) == 1
    assert isinstance(responses[0], PlasmaResponse)
    assert responses[0].rate is rate
    assert responses[0].species == "population-1"
    assert responses[0].preionisation == mixture.components[1].preionisation


def test_bare_rate_requires_the_named_species_and_a_named_rate():
    with pytest.raises(MaterialError, match="bare plasma rate"):
        responses_for(
            Mixture.air(1.0), plasma=OptmatRate("N2", 800e-9), wavelength=800e-9
        )
    with pytest.raises(MaterialError, match="bare plasma rate"):
        responses_for(mixture_case()[0], plasma=AnalyticRate())
    single = Mixture.fill({"N2": 1.0}, 1.0)
    rate = OptmatRate("N2", 800e-9)
    assert (
        next(
            r
            for r in responses_for(single, plasma=rate, wavelength=800e-9)
            if isinstance(r, PlasmaResponse)
        ).rate
        is rate
    )


@pytest.mark.parametrize("kind", ["kerr", "plasma", "raman"])
def test_missing_custom_data_names_the_species_even_at_zero_density(kind):
    optical = mixture_case()[0].components[0].species.susceptibility
    mixture = Mixture.of(Component(Species("without data", optical), Constant(0.0)))
    with pytest.raises(MaterialError, match="without data"):
        responses_for(
            mixture,
            kerr=kind == "kerr",
            plasma=kind == "plasma",
            raman={"without data": True} if kind == "raman" else False,
        )


def test_default_options_take_the_terms_a_species_has_at_the_mixture_temperature():
    # MolecularOptions() means every term the species has, only a term set to True
    # can be refused, and an unset temperature is the mixture's, not room temperature.
    rotor = synthetic_rotor()
    mixture = Mixture.of(Component(rotor, Constant(1e24), "rotor"), temperature=350.0)
    for selection in (True, {"rotor": MolecularOptions()}, {"rotor": True}):
        raman = responses_for(mixture, kerr=False, raman=selection)[0]
        assert isinstance(raman, RamanResponse) and len(raman.models) == 1
        assert isinstance(raman.models[0], RotationalLines)
        assert raman.models[0].temperature == 350.0
    assert len(response_for(rotor, MolecularOptions(temperature=350.0)).models) == 1
    with pytest.raises(MaterialError, match=r"synthetic rotor.*vibrational"):
        responses_for(
            mixture, kerr=False, raman={"rotor": MolecularOptions(vibration=True)}
        )
    explicit = responses_for(
        mixture, kerr=False, raman={"rotor": MolecularOptions(temperature=295.0)}
    )[0]
    assert isinstance(explicit, RamanResponse)
    assert isinstance(explicit.models[0], RotationalLines)
    assert explicit.models[0].temperature == 295.0


@pytest.mark.parametrize("name", ["Ar", "N2"])
def test_default_rate_needs_optmats_potential_but_not_its_citation(name):
    # Citation text is provenance: a relabelled built-in gas keeps optmat's rate. A
    # changed potential makes the species a caller's, whose rate must be given
    # explicitly (SEP 3, D1); solprop builds no rate from supplied data.
    material = species(name)
    assert material.ionisation is not None
    relabelled = dataclasses.replace(
        material,
        ionisation=dataclasses.replace(material.ionisation, citation="provenance"),
    )
    changed = dataclasses.replace(
        material,
        ionisation=dataclasses.replace(
            material.ionisation, potential=1.1 * material.ionisation.potential
        ),
    )
    response = responses_for(
        Mixture.of(Component(relabelled, Constant(1e24))),
        kerr=False,
        plasma=True,
        wavelength=800e-9,
    )[0]
    assert isinstance(response, PlasmaResponse)
    assert isinstance(response.rate, OptmatRate)
    field = np.geomspace(1e8, 1e11, 32)
    assert_array_equal(response.rate(field), OptmatRate(name, 800e-9)(field))
    with pytest.raises(MaterialError, match="custom species"):
        responses_for(
            Mixture.of(Component(changed, Constant(1e24))),
            kerr=False,
            plasma=True,
            wavelength=800e-9,
        )


def test_default_atomic_rate_requires_a_wavelength():
    with pytest.raises(MaterialError, match=r"Ar.*carrier wavelength"):
        responses_for(Mixture.fill({"Ar": 1.0}, 1.0), kerr=False, plasma=True)


def test_kerr_component_labels_are_in_mixture_order():
    mixture = Mixture.air(1.0)
    responses = responses_for(mixture, wavelength=800e-9)
    assert all(isinstance(r, KerrResponse) for r in responses)
    assert tuple(r.species for r in responses) == mixture.labels


def test_single_component_traits_keep_the_original_profile_names():
    responses = responses_for(
        Mixture.fill({"N2": 1.0}, 1.0),
        plasma=True,
        raman=True,
        wavelength=800e-9,
    )
    for response in responses:
        assert response.species is None
        assert "density" in response.traits.z_profiles
        assert all(":" not in name for name in response.traits.z_profiles)


def test_missing_custom_response_data_is_named():
    material = dataclasses.replace(
        mixture_case()[0].components[0].species, ionisation=None, raman=None
    )
    with pytest.raises(MaterialError, match="has no ionisation data"):
        _ = PlasmaResponse(material, AnalyticRate()).potential
    with pytest.raises(MaterialError, match="has no Raman response data"):
        response_for(material)


@pytest.mark.parametrize("name", ["supplied atom", "Air"])
def test_default_rate_refuses_a_species_optmat_has_no_rate_for(name):
    # A caller's species (named apart from every built-in gas) or air, a composition
    # rather than one species: their rates must be given by label (SEP 3, D1).
    material = dataclasses.replace(species("Ar"), name=name)
    mixture = Mixture.of(Component(material, Constant(1e24)))
    with pytest.raises(MaterialError, match="custom species"):
        responses_for(mixture, kerr=False, plasma=True, wavelength=800e-9)


def test_an_ideal_eos_does_not_remove_optmats_rate():
    material = dataclasses.replace(species("Ar"), equation_of_state=IdealGas())
    mixture = Mixture.of(Component(material, Constant(1e24)))
    response = responses_for(mixture, kerr=False, plasma=True, wavelength=800e-9)[0]
    assert isinstance(response, PlasmaResponse)
    assert isinstance(response.rate, OptmatRate)
    reference = OptmatRate("Ar", 800e-9)
    field = np.geomspace(1e8, 1e11, 32)
    assert response.rate.parameters == reference.parameters
    assert_array_equal(response.rate(field), reference(field))
