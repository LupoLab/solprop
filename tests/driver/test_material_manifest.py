# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run's material manifest: what the drivers resolve under a selection, and record.

Every run here is built and translated, or at most takes ten fixed steps on
``numpy_ref``: what is tested is what the run carries and records, not its physics.
"""

import dataclasses
import math
import warnings
from functools import partial
from typing import Any

import numpy as np
import pytest
from optmat.gas.models.registry import default_model

from former_raman import FORMER_SILICA_TENSOR
from modal_provider_cases import (
    GENERAL_CHI3,
    GENERAL_Z_END,
    general_grid,
    general_guide,
    grin_grid,
    grin_provider,
)
from solprop.boundaries import absorber
from solprop.constants import C
from solprop.driver import Simulation, SimulationSettings, prop_capillary
from solprop.driver.executors import Local
from solprop.driver.material_output import material_manifest
from solprop.driver.scan import Scan, run_scan
from solprop.driver.simulation import _selected_guide
from solprop.exceptions import DiluteScalingWarning, MaterialRangeWarning, OutputError
from solprop.grid import FieldGrid, FreeGrid, RadialGrid
from solprop.modes import PlanarGrid, SlabMode, StepIndexMode
from solprop.modes.guide import ModalGuide
from solprop.modes.material import IndexMaterial, MaterialKerr, MaterialRegion
from solprop.nonlinear.raman import SilicaTensorResponse
from solprop.output import open_file
from solprop.output.hdf5 import check_cache
from solprop.physdata import Gas, Mixture, fused_silica, gamma, species
from solprop.physdata.glasses import (
    OptmatGlass,
    SellmeierGlass,
    SilicaBundle,
    lossless_index,
)
from solprop.physdata.mixtures import DILUTE_SCALING_THRESHOLD
from solprop.physdata.selection import MaterialSelection
from solprop.physdata.sellmeier import OptmatSusceptibility
from solprop.physdata.species import OptmatGas
from solprop.profiles import Linear
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, PIController

# optmat's models warn beyond their qualified band on these windows; the test of
# optmat's once-per-run warnings records every warning itself.
pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    # The fixtures here launch noise-free into time windows kept tight for speed, which
    # hold more than a millionth of the energy in their guards or collar: the boundary
    # monitor warns (SEP 21), and the window's edge is not what this module tests.
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
]

SELECTION = MaterialSelection()
LENGTH = 4e-4
STEPS = 10

# A caller's own silica: Malitson's 1965 Sellmeier coefficients (I. H. Malitson,
# J. Opt. Soc. Am. 55, 1205 (1965)), supplied here as user data, never an optmat model.
CALLERS_SILICA = SellmeierGlass(
    name="SiO2",
    source="caller",
    wavelength_range=(0.21e-6, 3.71e-6),
    citation="I. H. Malitson, J. Opt. Soc. Am. 55, 1205 (1965), supplied by the test",
    poles=((0.6961663, 0.0684043), (0.4079426, 0.1162414), (0.8974794, 9.896161)),
)


def capillary_options(
    materials: MaterialSelection | None = SELECTION, **changes: object
) -> dict[str, Any]:
    """Keywords of a ten-step argon capillary on numpy_ref.

    The weak in-stage collar keeps ten fixed steps of so short a guide inside its
    stability limit, as for SEP 3's baseline witnesses.
    """
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (300e-9, 3e-6), 100e-15)
    return {
        "radius": 50e-6,
        "length": LENGTH,
        "gas": "Ar",
        "pressure": 1.0,
        "wavelength": 800e-9,
        "grid": grid,
        "absorber": absorber(grid, boundary_length=1.0),
        "fwhm": 30e-15,
        "energy": 1e-6,
        "plasma": False,
        "n_r": 8,
        "schedule": FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        "n_save": 2,
        "backend": "numpy_ref",
        "materials": materials,
    } | changes


def capillary(
    materials: MaterialSelection | None = SELECTION, **changes: object
) -> Any:
    """The capillary simulation of `capillary_options`, loosely typed for reading."""
    options = capillary_options(materials, **changes)
    return Simulation.from_capillary(
        options.pop("radius"),
        options.pop("length"),
        options.pop("gas"),
        **{"noise": None, **options},
    )


def names(manifest):
    """The names of a manifest's models."""
    return sorted(record.name for record in manifest.records)


def test_a_selected_capillary_records_its_gas_and_its_recommended_wall():
    simulation = capillary()
    manifest = simulation.metadata().materials
    assert names(manifest) == [
        "eos/Ar",
        "fit/Ar",
        "fit/SiO2",
        "hyperpolarizability/Ar",
        "transport/Ar",
    ]
    # cladding=None resolved to optmat's recommended silica, the gas through optmat.
    assert isinstance(simulation.modes[0].cladding, OptmatGlass)
    assert manifest.species[0]["label"] == "Ar"
    (solid,) = manifest.solids
    assert (solid["material"], solid["model"]) == ("SiO2", "fit/SiO2")
    assert solid["axis"] == ""
    assert solid["transparent"] is False
    (check,) = manifest.dilute_scaling
    assert check["measured"] and abs(check["relative_error"]) < 1e-3
    assert manifest.policies["electronic"] == "recommended"
    assert manifest.policies["linear_density_scaling"]["warning_threshold"] == 1e-2
    assert manifest.selection == SELECTION.to_dict()
    assert manifest.optmat_version == default_model("Ar").identity.optmat_version
    assert simulation.metadata().versions["optmat"] == manifest.optmat_version


def test_a_plasma_response_keeps_the_selected_species_and_lists_only_what_acted():
    # Q45: the plasma response carries the fill's species as the selection resolved
    # it, so optmat's recommended argon index, which nothing evaluates, is not listed.
    linear = "borzsonyi_2008/sellmeier_Ar"
    simulation = capillary(materials=MaterialSelection(linear=linear), plasma=True)
    manifest = simulation.metadata().materials
    assert names(manifest) == [
        linear,
        "eos/Ar",
        "fit/SiO2",
        "hyperpolarizability/Ar",
        "rate/default/Ar@800nm",
        "transport/Ar",
    ]
    (plasma,) = (one for one in simulation.responses if hasattr(one, "rate"))
    selected = species("Ar", selection=MaterialSelection(linear=linear))
    assert plasma.gas.susceptibility == selected.susceptibility
    assert plasma.gas.ionisation.potential == plasma.rate.plasma_energy


def test_the_absent_keyword_is_the_default_selection():
    # Since SEP 3's Phase 4a; Phase 4b deleted the registry selection beside it.
    options = capillary_options(materials=SELECTION)
    del options["materials"]
    absent = Simulation.from_capillary(
        options.pop("radius"),
        options.pop("length"),
        options.pop("gas"),
        **{"noise": None, **options},
    )
    assert (
        absent.metadata().configuration_hash()
        == capillary(materials=SELECTION).metadata().configuration_hash()
    )


def test_air_records_its_named_composition():
    manifest = capillary(gas="air").metadata().materials
    assert manifest.compositions == (
        {
            "name": "dry air (N2, O2)",
            "labels": ["N2", "O2"],
            "fractions": manifest.compositions[0]["fractions"],
        },
    )
    assert sum(manifest.compositions[0]["fractions"]) == pytest.approx(1.0, rel=1e-15)
    assert [entry["label"] for entry in manifest.species] == ["N2", "O2"]


def test_a_fraction_mapping_resolves_every_name_under_the_selection():
    simulation = capillary(gas={"He": 0.5, "Ar": 0.5}, plasma=False)
    labels = [entry["label"] for entry in simulation.metadata().materials.species]
    assert labels == ["Ar", "He"]
    for component in simulation.configuration["medium"].components:
        assert isinstance(component.species.susceptibility, OptmatSusceptibility)


def test_a_supplied_cladding_is_taken_as_it_is_and_cited():
    supplied = CALLERS_SILICA
    simulation = capillary(cladding=supplied)
    assert simulation.modes[0].cladding is supplied
    manifest = simulation.metadata().materials
    assert manifest.solids == ()
    # The caller's glass is cited.
    assert supplied.citation in [entry["citation"] for entry in manifest.citations]


def test_each_material_warning_is_raised_once_per_run():
    # A window whose grid reaches below argon's qualified interval (107.5 nm): the
    # construction, the translation and the record each evaluate the gas.
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (100e-9, 3e-6), 100e-15)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulation = capillary(grid=grid, absorber=absorber(grid, boundary_length=1.0))
        simulation.translate()
        manifest = simulation.metadata().materials
    ranges = [w for w in caught if issubclass(w.category, MaterialRangeWarning)]
    assert len(ranges) == 1
    assert "fit/Ar" in str(ranges[0].message)
    (entry,) = manifest.ranges
    assert entry["model"] == "fit/Ar"
    assert entry["outside"] and "below its shortest" in entry["outside"][0]


def test_a_gnlse_run_records_its_selection_and_changes_nothing_else():
    # A GNLSE run resolves no material, so another selection moves only its record.
    options = {
        "wavelength": 835e-9,
        "wavelength_limits": (500e-9, 2e-6),
        "trange": 1e-12,
        "width": 100e-15,
        "peak_power": 9.8,
        "shape": "sech",
        "n_save": 3,
    }
    plain = Simulation.from_gnlse(
        0.11,
        0.01,
        [0.0, 0.0, -1.183e-26],
        materials=MaterialSelection(electronic="carrier-kerr"),
        **{"noise": None, **options},
    )
    selected = Simulation.from_gnlse(
        0.11,
        0.01,
        [0.0, 0.0, -1.183e-26],
        materials=SELECTION,
        **{"noise": None, **options},
    )
    manifest = selected.metadata().materials
    assert not manifest.empty and manifest.records == ()
    assert plain.metadata().differences(selected.metadata()) == (
        "materials.policies.electronic",
    )
    np.testing.assert_array_equal(
        plain.translate().initial.field0, selected.translate().initial.field0
    )


@pytest.fixture
def free_case():
    grid = FieldGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 80e-6
    )
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
        backend="numpy_ref",
    )
    return grid, pulse, settings


def test_free_space_resolves_a_named_gas_medium_into_an_optmat_fill(free_case):
    grid, pulse, settings = free_case
    space = FreeGrid(3e-4, 12, 4e-4, 16, collar_fraction=0)
    simulation: Any = Simulation.from_free(
        grid,
        space,
        SpatialMedium("Ar", 2.0),
        [],
        pulse,
        settings=settings,
        materials=SELECTION,
        noise=None,
    )
    medium = simulation.medium.material
    assert isinstance(medium, Mixture)
    assert isinstance(medium.components[0].species.susceptibility, OptmatSusceptibility)
    simulation.translate()
    manifest = simulation.metadata().materials
    assert names(manifest) == [
        "eos/Ar",
        "fit/Ar",
        "hyperpolarizability/Ar",
        "transport/Ar",
    ]
    # The absent keyword resolves the same way: there is no other gas.
    plain: Any = Simulation.from_free(
        grid, space, SpatialMedium("Ar", 2.0), [], pulse, settings=settings, noise=None
    )
    assert isinstance(plain.medium.material, Mixture)
    assert plain.metadata().materials.hashed() == manifest.hashed()


def test_a_supplied_susceptibility_is_kept_and_a_glass_taken_as_it_is(free_case):
    grid, pulse, settings = free_case
    space = RadialGrid(3e-4, 8)
    # optmat's Borzsonyi form, not the recommended fit: a caller's choice.
    other = gamma(Gas.Ar, model="borzsonyi_2008/sellmeier_Ar")
    kept: Any = Simulation.from_radial(
        grid,
        space,
        SpatialMedium("Ar", Linear(1.0, 2.0, grid.length), susceptibility=other),
        [],
        pulse,
        settings=settings,
        materials=SELECTION,
        noise=None,
    )
    (component,) = kept.medium.material.components
    assert component.species.susceptibility == other
    assert type(component.species.equation_of_state).__name__ == "OptmatGas"
    window = fused_silica(model="recommended")
    glass: Any = Simulation.from_radial(
        grid,
        space,
        SpatialMedium(window),
        [],
        pulse,
        settings=settings,
        materials=SELECTION,
        noise=None,
    )
    assert glass.medium.material is window
    assert names(glass.metadata().materials) == ["fit/SiO2"]


def test_a_planar_named_gas_fill_resolves_beside_a_selection(free_case):
    # SEP 3, Q5: the slab's named gas takes optmat's single-species
    # susceptibility, no mixture, and its default cladding optmat's silica.
    grid, _, settings = free_case
    modes = [SlabMode(63.5e-6, "Ar", 1.0, order=order) for order in (1, 3)]
    space = PlanarGrid(modes, 1e-3, 8)
    start = np.zeros((space.n_columns, 1, grid.n_omega), complex)
    simulation: Any = Simulation.from_planar(
        grid, space, [], start, settings=settings, materials=SELECTION
    )
    resolved = simulation.spatial_grid.modes
    assert all(mode.mixture is None and mode.gas == "Ar" for mode in resolved)
    assert isinstance(resolved[0].susceptibility, OptmatSusceptibility)
    assert resolved[1].susceptibility is resolved[0].susceptibility
    assert isinstance(resolved[0].cladding, OptmatGlass)
    assert resolved[1].cladding is resolved[0].cladding
    assert simulation.medium.susceptibility is resolved[0].susceptibility
    # The slab's named gas is optmat's whole species since SEP 3's Phase 4a: its
    # density path carries optmat's equation of state too (Q27).
    assert names(simulation.metadata().materials) == [
        "eos/Ar",
        "fit/Ar",
        "fit/SiO2",
        "hyperpolarizability/Ar",
        "transport/Ar",
    ]
    # The gas's index is optmat's, at the slab's own density.
    omega = np.array([2 * np.pi * C / 800e-9])
    np.testing.assert_array_equal(
        resolved[0].susceptibility.susceptibility(omega),
        species("Ar", selection=SELECTION).susceptibility.susceptibility(omega),
    )
    assert resolved[0].density(np.zeros(1)) == modes[0].density(np.zeros(1))
    simulation.translate()


def test_a_planar_fill_is_carried_as_one_species_with_its_dilute_check(free_case):
    # Q27: the slab's fill is one species, not a mixture; its equation of state gives
    # the density and the manifest records it with its dilute-scaling check, once
    # for the two modes that share it. Argon at 60 bar is above the 1 % bound.
    grid, _, settings = free_case
    linear = "borzsonyi_2008/sellmeier_Ar"
    cases = ((SELECTION, "fit/Ar"), (MaterialSelection(linear=linear), linear))
    for selection, law in cases:
        modes = [SlabMode(63.5e-6, "Ar", 60.0, order=order) for order in (1, 3)]
        space = PlanarGrid(modes, 1e-3, 8)
        start = np.zeros((space.n_columns, 1, grid.n_omega), complex)
        with pytest.warns(DiluteScalingWarning, match="Ar"):
            simulation: Any = Simulation.from_planar(
                grid, space, [], start, settings=settings, materials=selection
            )
            manifest = simulation.metadata().materials
        resolved = simulation.spatial_grid.modes[0]
        assert resolved.mixture is None
        assert resolved.fill_species.susceptibility is resolved.susceptibility
        assert resolved.fill_species.equation_of_state == OptmatGas("Ar")
        # Only what acted: the selected law, never the recommended one beside it.
        assert names(manifest) == sorted(
            [law, "eos/Ar", "fit/SiO2", "hyperpolarizability/Ar", "transport/Ar"]
        )
        (entry,) = manifest.species
        assert entry["label"] == "Ar" and len(entry["records"]) == 4
        (check,) = manifest.dilute_scaling
        assert check["label"] == "Ar" and check["measured"]
        assert check["relative_error"] < -DILUTE_SCALING_THRESHOLD
        assert check["density"] == pytest.approx(
            float(resolved.density(np.zeros(1))[0]), rel=1e-12
        )


def test_a_planar_fill_and_cladding_given_explicitly_are_kept(free_case):
    grid, _, settings = free_case
    wall = fused_silica()
    kept = PlanarGrid(
        [SlabMode(63.5e-6, "Ar", 1.0, susceptibility=gamma(Gas.Ar), cladding=wall)],
        1e-3,
        8,
    )
    simulation: Any = Simulation.from_planar(
        grid,
        kept,
        [],
        np.zeros((kept.n_columns, 1, grid.n_omega), complex),
        settings=settings,
        materials=SELECTION,
    )
    # optmat's recommended susceptibility given explicitly is the built-in gas, and
    # resolves; a declared cladding is the caller's.
    assert isinstance(
        simulation.spatial_grid.modes[0].susceptibility, OptmatSusceptibility
    )
    assert simulation.spatial_grid.modes[0].cladding is wall
    vacuum = PlanarGrid([SlabMode(63.5e-6, cladding=wall)], 1e-3, 8)
    untouched: Any = Simulation.from_planar(
        grid,
        vacuum,
        [],
        np.zeros((vacuum.n_columns, 1, grid.n_omega), complex),
        settings=settings,
        materials=SELECTION,
    )
    assert untouched.spatial_grid is vacuum
    assert untouched.materials is SELECTION
    own: Any = species("Ar", selection=SELECTION).susceptibility
    supplied = PlanarGrid(
        [SlabMode(63.5e-6, "Ar", 1.0, susceptibility=own, cladding=wall)], 1e-3, 8
    )
    taken: Any = Simulation.from_planar(
        grid,
        supplied,
        [],
        np.zeros((supplied.n_columns, 1, grid.n_omega), complex),
        settings=settings,
        materials=SELECTION,
    )
    assert taken.spatial_grid is supplied
    # Another optmat law supplied by the caller is not the built-in gas: it is kept,
    # whatever the selection says.
    law = gamma(Gas.Ar, model="borzsonyi_2008/sellmeier_Ar")
    other = PlanarGrid(
        [SlabMode(63.5e-6, "Ar", 1.0, susceptibility=law, cladding=wall)], 1e-3, 8
    )
    kept_law: Any = Simulation.from_planar(
        grid,
        other,
        [],
        np.zeros((other.n_columns, 1, grid.n_omega), complex),
        settings=settings,
        materials=SELECTION,
    )
    assert kept_law.spatial_grid is other


def test_a_planar_grid_below_its_node_bound_warns_once(free_case):
    grid, _, settings = free_case
    with pytest.warns(UserWarning, match="PlanarGrid n_x") as seen:
        space = PlanarGrid([SlabMode(63.5e-6, "Ar", 1.0, order=9)], 1e-3, 8, n_x=40)
        Simulation.from_planar(
            grid,
            space,
            [],
            np.zeros((space.n_columns, 1, grid.n_omega), complex),
            settings=settings,
            materials=SELECTION,
        )
    assert len(seen) == 1


def test_a_waveguide_fill_resolves_and_its_supplied_modes_keep_their_cladding():
    supplied = capillary(cladding=CALLERS_SILICA).modes
    options = capillary_options()
    for name in ("radius", "n_r", "gas", "pressure", "length"):
        options.pop(name)
    simulation: Any = Simulation.from_waveguide(
        supplied, LENGTH, "Ar", 1.0, **{"noise": None, **options}
    )
    assert simulation.modes[0].cladding is supplied[0].cladding
    assert isinstance(
        simulation.modes[0].mixture.components[0].species.susceptibility,
        OptmatSusceptibility,
    )
    assert "fit/SiO2" not in names(simulation.metadata().materials)


def test_a_step_index_core_and_a_material_region_are_collected():
    silica: Any = fused_silica(model="recommended")
    core = silica.as_transparent()
    mode = StepIndexMode(2e-6, core=core, cladding=1.0, profile_wavelength=1e-6)
    region = MaterialRegion("core", 0.0, 2e-6, IndexMaterial(core))
    omega = 2 * np.pi * C / np.array([800e-9, 1e-6])
    manifest = material_manifest((mode, [region]), selection=None, omega=omega)
    (solid,) = manifest.solids
    assert solid["transparent"] is True and solid["absorption_bound"] == 100.0
    assert names(manifest) == ["fit/SiO2"]
    # The walk visits the shared adapter once, through both holders.
    assert len(manifest.records) == 1
    # Nothing optmat and no selection: the empty manifest, whatever else is there.
    legacy = StepIndexMode(2e-6, core=1.45, cladding=1.0, profile_wavelength=1e-6)
    assert material_manifest((legacy,), selection=None, omega=omega).empty


def test_resolution_does_not_depend_on_what_was_resolved_before():
    # Scans resolve a selection at each point: the manifest of one point must not
    # depend on the order in which materials were resolved in the process.
    species("Kr", selection=SELECTION)
    first = capillary().material_manifest()
    species("Ne", selection=SELECTION)
    fused_silica(model="recommended")
    second = capillary().material_manifest()
    assert first.hashed() == second.hashed()


@pytest.mark.medium
def test_ten_steps_write_the_manifest_and_a_changed_identity_is_refused(tmp_path):
    path = tmp_path / "argon.h5"
    options = capillary_options()
    prop_capillary(
        options.pop("radius"),
        options.pop("length"),
        options.pop("gas"),
        output=path,
        progress="none",
        **{"noise": None, **options},
    )
    stored: Any = open_file(path).metadata
    assert stored.schema_version == 2
    assert names(stored.materials) == [
        "eos/Ar",
        "fit/Ar",
        "fit/SiO2",
        "hyperpolarizability/Ar",
        "transport/Ar",
    ]
    assert check_cache(path, capillary().metadata())
    override = default_model("Ar").with_parameters(intervals=((120e-9, math.inf),))
    changed = capillary(materials=MaterialSelection(linear=override)).metadata()
    with pytest.raises(OutputError, match=r"materials\.records") as refused:
        check_cache(path, changed)
    assert "the material manifest differs" in str(refused.value)


def scan_point(point, linear: object = "recommended"):
    """A scan point: the capillary at the point's energy, under a linear choice."""
    return capillary(
        materials=MaterialSelection(linear=linear), energy=1e-6 * (point.index + 1)
    )


@pytest.mark.medium
def test_scan_reuse_is_refused_when_a_points_identity_changes(tmp_path):
    scan = Scan("materials", base_seed=5, energy=[1.0])
    options = {"output": tmp_path / "scan.h5", "executor": Local(), "progress": "none"}
    run_scan(scan, scan_point, **options)
    # The same selection reuses the completed point.
    assert run_scan(scan, scan_point, **options).point(0)["status"] == "done"
    override = default_model("Ar").with_parameters(intervals=((120e-9, math.inf),))
    refusal = r"different configuration.*materials\.records"
    with pytest.raises(OutputError, match=refusal):
        run_scan(scan, partial(scan_point, linear=override), **options)


def test_the_walk_reaches_through_objects_that_are_not_dataclasses():
    # A Simulation keeps its parts as plain attributes; its modes carry the optmat
    # wall and the gas, which the walk finds through it.
    simulation = capillary()
    manifest = material_manifest(
        (simulation,), selection=None, omega=simulation.grid.omega
    )
    assert names(manifest) == [
        "eos/Ar",
        "fit/Ar",
        "fit/SiO2",
        "hyperpolarizability/Ar",
        "transport/Ar",
    ]


def test_a_planar_runs_slab_cladding_is_collected(free_case):
    grid, _, settings = free_case
    wall = fused_silica(model="recommended")
    space = PlanarGrid([SlabMode(63.5e-6, cladding=wall)], 1e-3, 8)
    start = np.zeros((space.n_columns, 1, grid.n_omega), complex)
    for materials in (None, SELECTION):
        simulation = Simulation.from_planar(
            grid, space, [], start, settings=settings, materials=materials
        )
        manifest = simulation.material_manifest()
        assert names(manifest) == ["fit/SiO2"]
        assert manifest.solids[0]["material"] == "SiO2"


@dataclasses.dataclass(frozen=True, eq=False)
class CallersStepIndex(StepIndexMode):
    """A caller's subclass of a solprop mode, defined outside solprop."""


class CallersMode:
    """A caller's own mode: no dataclass and no solprop base, only attributes."""

    def __init__(self, cladding):
        self.cladding = cladding

    def index_at(self, omega):
        """A bound method the run might hold."""
        return self.cladding.index(omega)


def cladding_of(glass):
    """A function a partial is built from."""
    return glass


@pytest.mark.parametrize(
    "holder",
    [
        lambda glass: CallersStepIndex(
            2e-6, core=glass, cladding=1.0, profile_wavelength=1e-6
        ),
        CallersMode,
        lambda glass: partial(cladding_of, glass=glass),
        lambda glass: CallersMode(glass).index_at,
    ],
    ids=["subclass", "user-mode", "partial", "bound-method"],
)
def test_the_walk_finds_optmat_objects_in_a_callers_objects(holder):
    silica: Any = fused_silica(model="recommended")
    glass = silica.as_transparent()
    omega = 2 * np.pi * C / np.array([800e-9])
    manifest = material_manifest((holder(glass),), selection=None, omega=omega)
    assert names(manifest) == ["fit/SiO2"]


def test_the_walk_stops_at_closures_and_at_optmat_itself():
    # Documented limits: a closure's captures are not walked, and optmat's own
    # objects are recorded through solprop's adapters, never walked.
    glass = fused_silica(model="recommended")
    omega = 2 * np.pi * C / np.array([800e-9])
    closure = lambda: glass  # noqa: E731 - the closure is what is tested
    assert material_manifest((closure,), selection=None, omega=omega).empty
    assert material_manifest((default_model("Ar"),), selection=None, omega=omega).empty


def test_a_derived_run_keeps_its_own_log_from_its_parents():
    parent = capillary()
    parent.translate()
    parent._material_log.seen.add(("probe", "parent"))
    child = parent.with_input(parent.translate().initial.field0[0])
    assert child._material_log is not parent._material_log
    assert ("probe", "parent") in child._material_log.seen
    child._material_log.seen.add(("probe", "child"))
    assert ("probe", "child") not in parent._material_log.seen


# ------------------------------------------- the general modal route (Q3 and Q8)


def general(guide, grid, materials: MaterialSelection | None = SELECTION) -> Any:
    """A general modal run of `guide`, a tenth of a millimetre in two steps."""
    return Simulation.from_waveguide(
        (),
        GENERAL_Z_END,
        provider=guide,
        grid=grid,
        wavelength=835e-9,
        fwhm=20e-15,
        energy=1e-12,
        shape="gauss",
        save_planes=[GENERAL_Z_END / 10],
        backend="numpy_ref",
        dz_max=GENERAL_Z_END / 20,
        dz_init=GENERAL_Z_END / 20,
        materials=materials,
        noise=None,
    )


@pytest.mark.medium
def test_a_general_guide_on_an_absorbing_optmat_core_is_solved_transparent():
    # SEP 3, Q3: the step-index provider, its regions' laws and the vector fields
    # take the transparent counterpart; nothing refuses, and the run records it.
    silica: Any = fused_silica(model="recommended")
    grid = general_grid()
    guide: Any = general_guide(grid=grid, indices=(silica, 1.0))
    counterpart = lossless_index(silica)
    assert guide.regions[0].material.index is counterpart
    assert guide.field_provider.modes[0].core is counterpart
    simulation: Any = general(guide, grid)
    simulation.translate()
    (solid,) = simulation.material_manifest().solids
    assert solid["transparent"] is True and solid["absorption_bound"] == 100.0
    assert solid["transparency"]["largest_dropped_extinction"] > 0
    assert solid["transparency"]["bound_exceeded"] is False


@pytest.mark.medium
def test_a_default_tensor_raman_response_resolves_to_optmats_bundle():
    # SEP 3, Q8: a region's SilicaTensorResponse() under a selection is optmat's
    # tensor bundle, with an identity; its kernels are the declared ones to 4e-16.
    grid = general_grid()
    declared = SilicaTensorResponse()
    guide = general_guide(grid=grid, raman=declared)
    simulation: Any = general(guide, grid)
    resolved = simulation.guide.regions[0].raman
    assert isinstance(resolved.parameters, SilicaBundle)
    assert resolved.parameters.name == "silica_bundle/tensor"
    assert simulation.guide.field_provider.regions == simulation.guide.regions
    assert simulation.guide.sampling is guide.sampling
    assert "silica_bundle/tensor" in names(simulation.metadata().materials)
    identity = simulation.metadata().modal["guide"]["regions"][0]["raman"]
    assert identity == {
        "bundle": "silica_bundle/tensor",
        "digest": resolved.parameters.digest,
        "fraction": resolved.fraction,
    }
    t = np.arange(4000) * 1e-16
    np.testing.assert_allclose(
        resolved.kernels(t),
        declared.kernels(t),
        rtol=0,
        atol=4e-16 * np.max(np.abs(declared.kernels(t))),
    )
    assert resolved.fraction == declared.fraction
    simulation.translate()
    # A caller's response with other parameters is the run's: solprop's former
    # Lin-Agrawal parameters (former_raman, dc4ca0d's registry) with another fraction.
    other = SilicaTensorResponse(
        parameters=dataclasses.replace(FORMER_SILICA_TENSOR, fraction=0.2)
    )
    own = general_guide(grid=grid, raman=other)
    assert general(own, grid).guide is own
    linear = general_guide(grid=grid)
    assert general(linear, grid).guide is linear
    # A provider of the caller's own kind cannot be rebuilt, and is taken as it is.
    wrapped = dataclasses.replace(
        guide, field_provider=Delegating(guide.field_provider)
    )
    assert _selected_guide(wrapped, SELECTION) is wrapped


class Delegating:
    """A caller's own provider, of no shipped kind, answering as another one."""

    def __init__(self, inner):
        self._inner = inner

    def node_rule(self, omega, z=0.0, *, n_r=None, n_theta=None):
        return self._inner.node_rule(omega, z, n_r=n_r, n_theta=n_theta)

    def solve(self, omega, z, *, nodes):
        return self._inner.solve(omega, z, nodes=nodes)

    @property
    def regions(self):
        return self._inner.regions

    @property
    def z_breakpoints(self):
        return self._inner.z_breakpoints

    @property
    def omega_domain(self):
        return self._inner.omega_domain

    @property
    def z_domain(self):
        return self._inner.z_domain

    @property
    def identity(self):
        return self._inner.identity


@pytest.mark.medium
def test_a_graded_index_cores_default_tensor_response_resolves_too():
    grid = grin_grid()
    provider = dataclasses.replace(
        grin_provider(),
        core_responses=(MaterialKerr(chi3=GENERAL_CHI3), SilicaTensorResponse()),
    )
    active = grid.omega[grid.active_mask]
    lo, hi = float(active[0]), float(active[-1])
    guide = ModalGuide.from_provider(
        provider,
        np.linspace(0.99 * lo, 1.01 * hi, 5),
        nodes=provider.node_rule(0.99 * lo, n_r=2),
    )
    resolved: Any = _selected_guide(guide, SELECTION)
    kerr, raman = resolved.field_provider.core_responses
    assert kerr == provider.core_responses[0]
    assert isinstance(raman.parameters, SilicaBundle)
    assert resolved.regions == resolved.field_provider.regions
    assert resolved.sampling is guide.sampling
