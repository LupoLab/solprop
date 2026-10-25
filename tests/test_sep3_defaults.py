# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 3 after Phase 4b: optmat is the only source of solprop's built-in materials.

Phase 4a made an omitted selection ``MaterialSelection()`` and kept a temporary
``"registry"`` value per family; Phase 4b deleted it with solprop's own data. These
tests pin what the default resolves to, that the deleted names and the registry value
are gone, and the decisions of Phase 4a that outlive the registry (Q14, Q17, Q26).
The pre-deletion witnesses and the attributed change of each one were historical
records, no longer kept beside the tests; the registry runs that reproduced them bit for
bit were last run at solprop ``8d116c8``.
"""

import warnings
from typing import Any, cast

import pytest

import solprop.nonlinear.ionisation
import solprop.physdata
import solprop.physdata.selection
from solprop.driver.simulation import Simulation, SimulationSettings
from solprop.exceptions import MaterialError, MaterialStatusWarning, SolpropWarning
from solprop.grid import FieldGrid, FreeGrid
from solprop.modes.guide import ModalGuide
from solprop.modes.material import IndexMaterial
from solprop.modes.stepindex import StepIndexMode
from solprop.nonlinear.gnlse import problem_from
from solprop.nonlinear.mixtures import responses_for
from solprop.nonlinear.optmat_raman import OptmatRamanResponse
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import (
    Mixture,
    bk7,
    fused_silica,
    gamma3,
    glass,
    magnesium_fluoride,
    sapphire,
    silver,
    species,
)
from solprop.physdata.gases import Gas
from solprop.physdata.glasses import OptmatGlass, SilicaBundle
from solprop.physdata.selection import MaterialSelection
from solprop.physdata.sellmeier import (
    OptmatSusceptibility,
    PoleSumSusceptibility,
    ReferenceState,
)
from solprop.physdata.species import OptmatGas, Species, built_in_population
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController

# ------------------------------------------------------------------ the selection


@pytest.mark.parametrize(
    ("module", "name"),
    [
        (solprop.physdata.selection, "REGISTRY"),
        (solprop.physdata.selection, "REGISTRY_SELECTION"),
        (solprop.physdata.selection, "FAMILIES"),
        (solprop.physdata.selection, "default_selection"),
        (solprop.physdata.selection, "takes_optmat"),
        (solprop.physdata.selection, "is_registry"),
        (solprop.physdata, "RegistryGas"),
        (solprop.physdata, "sources"),
        (solprop.physdata, "gamma3_sources"),
        (solprop.physdata, "raman_parameters"),
        (solprop.physdata, "silica_raman"),
        (solprop.physdata, "quantum_numbers"),
        (solprop.physdata, "fused_silica_sources"),
        (solprop.nonlinear.ionisation, "ADK"),
        (solprop.nonlinear.ionisation, "PPT"),
        (solprop.nonlinear.ionisation, "IsotropicMOADK"),
    ],
)
def test_the_deleted_names_are_gone(module, name):
    """Phase 4b deleted the registry, its handle and solprop's rate formulas."""
    assert not hasattr(module, name)
    assert name not in module.__all__


def test_registry_is_no_longer_a_value_of_any_family():
    with pytest.raises(MaterialError, match="electronic"):
        MaterialSelection(electronic="registry")
    # A model family takes the name as an optmat model's, which optmat has not.
    with pytest.raises(MaterialError, match="no linear model 'registry'"):
        species("Ar", selection=MaterialSelection(linear="registry"))


def test_the_electronic_policies_still_refuse_other_names():
    with pytest.raises(MaterialError, match="electronic"):
        MaterialSelection(electronic="kerr")


# --------------------------------------------------------------------- the defaults


def test_a_species_is_optmat_and_takes_no_source():
    argon = species("Ar")
    assert isinstance(argon.susceptibility, OptmatSusceptibility)
    assert isinstance(argon.equation_of_state, OptmatGas)
    assert argon.gamma3 is None and argon.kerr is not None
    assert argon.raman is None
    assert argon.ionisation is not None
    with pytest.raises(TypeError, match="source"):
        cast("Any", species)("Ar", source="kingston")


def test_a_built_in_population_is_one_species_resolved():
    assert built_in_population(species("N2"))
    law = PoleSumSusceptibility(
        gas=Gas.N2,
        source="supplied",
        reference_state=ReferenceState(pressure_bar=1.0, temperature=273.15),
        wavelength_range=(400e-9, 1e-6),
        citation="a caller's own two-pole fit",
        b=(5.8e-4,),
        c=(7.4e-3,),
    )
    assert not built_in_population(Species("N2", law))


def test_air_is_the_named_composition():
    assert Mixture.air(1.0).composition == "dry air (N2, O2, Ar)"
    assert Mixture.air(1.0, argon=False).composition == "dry air (N2, O2)"


def test_the_solid_factories_take_optmat():
    assert isinstance(fused_silica(), OptmatGlass)
    assert fused_silica().source == "fit/SiO2"
    assert bk7().source == "schott_2023/sellmeier"
    assert silver().source == "table/Ag"
    assert glass("CaF2").source == "daimon_2002/sellmeier_20C"


@pytest.mark.parametrize(
    "build",
    [magnesium_fluoride, sapphire, lambda: glass("MgF2"), lambda: glass("sapphire")],
)
def test_a_uniaxial_crystal_without_its_ray_is_refused_by_name(build):
    with pytest.raises(MaterialError, match="uniaxial"):
        build()


def test_a_uniaxial_crystals_named_model_is_its_ray():
    assert sapphire(axis="o").source == "weber_2003/sellmeier_o"
    with pytest.raises(MaterialError, match="ray"):
        sapphire(axis="e", model="weber_2003/sellmeier_o")


def test_the_gnlse_fraction_with_raman_on_is_the_bundles():
    problem = problem_from([0.0, 0.0, -1e-26], omega_ref=2.3e15, gamma=0.1, raman=True)
    assert problem.fraction_origin == "bundle"
    assert problem.raman_fraction == SilicaBundle.resolve().raman_fraction
    assert f"{problem.raman_fraction:.4f}" == "0.2188"
    off = problem_from([0.0, 0.0, -1e-26], omega_ref=2.3e15, gamma=0.1)
    assert off.raman_fraction == 0.0 and off.fraction_origin == "raman-off"
    held = problem_from(
        [0.0, 0.0, -1e-26], omega_ref=2.3e15, gamma=0.1, raman_fraction=0.18
    )
    assert held.raman_fraction == 0.18 and held.fraction_origin == "user"


def test_the_default_plasma_rate_is_optmats():
    fill = Mixture.fill({"N2": 1.0}, 1.0)
    (kerr, plasma) = cast("Any", responses_for)(fill, plasma=True, lambda0=800e-9)
    assert isinstance(kerr, KerrResponse)
    assert isinstance(plasma.rate, OptmatRate)


# ----------------------------------------------------------- Q26: value equality


def test_two_default_silica_adapters_are_one_material():
    first, second = cast("Any", fused_silica()), cast("Any", fused_silica())
    assert first is not second
    assert first == second and hash(first) == hash(second)
    transparent = first.as_transparent()
    assert transparent != first
    assert transparent == second.as_transparent()
    assert first.as_transparent(absorption_bound=50.0) != transparent
    assert first != fused_silica(model="malitson_1965/sellmeier")
    assert first != "fit/SiO2"
    assert len({first, second, transparent}) == 2


def test_two_default_cores_share_one_step_index_guide():
    """Q26: two channels built separately with ``core=None`` are one guide."""
    channels = tuple(
        StepIndexMode(
            2e-6, cladding=1.4, profile_wavelength=1e-6, kind=cast("Any", kind), n=n
        )
        for kind, n in (("HE", 1), ("TE", 0))
    )
    assert channels[0]._core is not channels[1]._core
    assert channels[0]._core == channels[1]._core
    guide = ModalGuide.from_step_index(
        channels, core=IndexMaterial(channels[0]._core), cladding=IndexMaterial(1.4)
    )
    assert len(guide.mode_collection.modes) == 2


def test_gas_adapters_compare_by_their_digest():
    argon = OptmatSusceptibility.resolve("Ar")
    assert argon == OptmatSusceptibility.resolve("Ar")
    assert hash(argon) == hash(OptmatSusceptibility.resolve("Ar"))
    narrow = cast("Any", argon.model).with_parameters(
        intervals=((120e-9, float("inf")),)
    )
    assert argon != OptmatSusceptibility.resolve("Ar", narrow)
    assert argon != OptmatSusceptibility.resolve("Kr")
    assert argon != 1.0


def test_silica_bundles_compare_by_their_digest():
    assert SilicaBundle.resolve() == SilicaBundle.resolve()
    assert hash(SilicaBundle.resolve()) == hash(SilicaBundle.resolve())
    assert SilicaBundle.resolve() != SilicaBundle.resolve(tensor=True)
    assert SilicaBundle.resolve() != "default"


# ---------------------------------------------------------------- Q14: free space


def _free(responses, materials=None) -> Any:
    """Build a small Cartesian argon run, translating nothing."""
    grid = FieldGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15)
    space = FreeGrid(3e-4, 12, 4e-4, 16, window_factor=0)
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
        backend="numpy_ref",
        ionisation_field_range=(1e7, 1e11),
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 80e-6
    )
    return Simulation.from_free(
        grid,
        space,
        SpatialMedium("Ar", 1.0),
        responses,
        pulse,
        settings=settings,
        materials=materials,
    )


def test_a_free_kerr_response_is_the_callers():
    """Q9's by-value substitution went with the registry: a coefficient is as given."""
    coefficient = gamma3("Ar", wavelength=800e-9)
    assert coefficient == species("Ar").kerr_coefficient(800e-9)
    (kerr,) = _free([KerrResponse(coefficient, thg=False)]).responses
    assert kerr.chi3 == coefficient and kerr.thg is False
    (given,) = _free([KerrResponse(1.5e-51, thg=False)]).responses
    assert given.chi3 == 1.5e-51


def test_a_free_route_binds_an_optmat_rate_to_the_input_peak():
    """Q14: the free-space rate's energy per event is taken at the input's peak."""
    rate = OptmatRate("Ar", 800e-9)
    assert rate.reference_field is None
    simulation = _free([PlasmaResponse("Ar", rate)])
    bound = simulation.responses[0].rate
    assert isinstance(bound, OptmatRate)
    assert bound.reference_field is not None and bound.reference_field > 0
    assert bound.field_range == (1e7, 1e11)
    kept = _free([PlasmaResponse("Ar", rate.bound(1e10))])
    assert kept.responses[0].rate.reference_field == 1e10


# --------------------------------------------------------- Q17: N2O, provisional


def _n2o_capillary(selection):
    return Simulation.from_capillary(
        50e-6,
        4e-4,
        "N2O",
        1.0,
        lambda0=800e-9,
        wavelength_limits=(400e-9, 2e-6),
        trange=200e-15,
        fwhm=30e-15,
        energy=1e-6,
        plasma=False,
        n_r=8,
        n_save=2,
        materials=selection,
    )


def test_n2o_raman_stays_off_by_default_with_its_warning():
    with pytest.warns(SolpropWarning, match="provisionally"):
        simulation = _n2o_capillary(None)
    assert not any(isinstance(one, OptmatRamanResponse) for one in simulation.responses)


def test_the_provisional_policy_turns_n2os_default_raman_on_once():
    """Q17: ``raman=None`` under ``"provisional"`` takes N2O's provisional bands."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulation = _n2o_capillary(MaterialSelection(raman="provisional"))
    statuses = [
        one for one in caught if issubclass(one.category, MaterialStatusWarning)
    ]
    assert len(statuses) == 1
    assert not any("provisionally" in str(one.message) for one in caught)
    assert any(isinstance(one, OptmatRamanResponse) for one in simulation.responses)
    names = {record.name for record in simulation.material_manifest().records}
    assert any("N2O" in name and "raman" in name for name in names), names
