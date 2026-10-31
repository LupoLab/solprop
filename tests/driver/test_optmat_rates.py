# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""optmat's ionisation rates in a run: selection, reference field, record, diagnostic.

Every run here is built and translated, or takes ten fixed steps on ``numpy_ref``.
The adapter is compared with direct calls of the optmat model it carries, never with
optmat's values, so that a later optmat release changes both sides alike.
"""

import dataclasses
import json
from typing import Any, cast

import numpy as np
import pytest

from rate_oracles import adk_oracle
from solprop.boundaries import absorber
from solprop.driver import Simulation, SimulationSettings
from solprop.driver.simulation import (
    _capillary_input_peak,
    _check_rate_inputs,
    default_field_range,
)
from solprop.exceptions import MaterialError, MaterialRangeWarning
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid
from solprop.nonlinear import responses_for
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import PlasmaResponse
from solprop.output import open_file
from solprop.output.hdf5 import check_cache
from solprop.output.schema import PEAK_FIELDS_ANNOTATION
from solprop.physdata import Mixture, species
from solprop.physdata.selection import MaterialManifest, MaterialSelection
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import default_spec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.envelope_plasma import cycle_moments

# optmat's gas models warn beyond their qualified band on these windows.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:.*qualified intervals:solprop.exceptions.MaterialRangeWarning",
        "ignore:.*is fitted over:solprop.exceptions.MaterialRangeWarning",
    ),
    # The fixtures here launch noise-free into time windows kept tight for speed, which
    # hold more than a millionth of the energy in their guards or collar: the boundary
    # monitor warns (SEP 21), and the window's edge is not what this module tests.
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
]

SELECTION = MaterialSelection()
LENGTH = 4e-4
STEPS = 10
WAVELENGTH = 800e-9


def options(
    materials: MaterialSelection | None = SELECTION, **changes: Any
) -> dict[str, Any]:
    """Keywords of a ten-step argon capillary with plasma on numpy_ref."""
    envelope = bool(changes.pop("envelope", False))
    kind = EnvelopeGrid if envelope else FieldGrid
    grid = kind.from_limits(LENGTH, WAVELENGTH, (300e-9, 3e-6), 100e-15)
    return {
        "radius": 50e-6,
        "length": LENGTH,
        "gas": "Ar",
        "pressure": 1.0,
        "wavelength": WAVELENGTH,
        "grid": grid,
        "absorber": absorber(grid, boundary_length=1.0),
        "fwhm": 30e-15,
        "energy": 100e-6,
        "plasma": True,
        "envelope": envelope,
        "n_r": 8,
        "schedule": FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        "n_save": 2,
        "backend": "numpy_ref",
        "materials": materials,
    } | changes


def capillary(materials: MaterialSelection | None = SELECTION, **changes: Any) -> Any:
    """The capillary of `options`."""
    given = options(materials, **changes)
    return Simulation.from_capillary(
        given.pop("radius"),
        given.pop("length"),
        given.pop("gas"),
        **{"noise": None, **given},
    )


def plasma_of(simulation) -> Any:
    """The run's one plasma response."""
    (one,) = [
        response
        for response in simulation.responses
        if isinstance(response, PlasmaResponse | EnvelopePlasmaResponse)
    ]
    return one


def input_peak(simulation) -> float:
    """The input's peak field, as the driver finds it."""
    mode_set = simulation.modes
    from solprop.modes.collection import ModeCollection  # noqa: PLC0415

    collection = ModeCollection(list(mode_set))
    peak, circular = _capillary_input_peak(
        simulation._field0, collection, simulation.grid, simulation.settings.quadrature
    )
    assert circular < 1e-12  # one mode, one scalar pulse: linear everywhere
    return peak


# ------------------------------------------------------------- the selection


def test_responses_for_takes_optmats_rate_under_a_selection():
    mixture = Mixture.fill({"Ar": 0.5, "N2": 0.5}, 1.0, selection=SELECTION)
    responses: Any = responses_for(
        mixture, kerr=False, plasma=True, wavelength=WAVELENGTH, selection=SELECTION
    )
    rates = {response.species: response.rate for response in responses}
    assert {label: type(rate) for label, rate in rates.items()} == {
        "Ar": OptmatRate,
        "N2": OptmatRate,
    }
    assert rates["N2"].material_records[0].name.startswith("molecular_rate/")
    # Each response pays its own rate's energy.
    for response in responses:
        assert response.potential == response.rate.ionisation_potential


def test_a_custom_species_under_a_selection_needs_an_explicit_rate():
    probe = dataclasses.replace(species("Ar", selection=SELECTION), name="X")
    mixture = Mixture.of(*Mixture.fill({"Ar": 1.0}, 1.0).components)
    custom = Mixture.of(dataclasses.replace(mixture.components[0], species=probe))
    with pytest.raises(MaterialError, match="custom species"):
        responses_for(
            custom, kerr=False, plasma=True, wavelength=WAVELENGTH, selection=SELECTION
        )
    from optmat.gas.ionisation_rates import IonisationRateModel  # noqa: PLC0415

    model = IonisationRateModel.from_parameters(
        "X",
        WAVELENGTH,
        ionization_energy=2.5e-18,
        orbital_l=1,
        electrons=1.0,
        static_constants=(9.0,),
        citation="a test species",
    )
    (response,) = cast("Any", responses_for)(
        custom,
        kerr=False,
        plasma={"Ar": OptmatRate(model)},
        wavelength=WAVELENGTH,
        selection=SELECTION,
    )
    assert response.potential == 2.5e-18


def test_a_bare_optmat_rate_names_its_species():
    mixture = Mixture.fill({"Ar": 1.0}, 1.0, selection=SELECTION)
    (response,) = cast("Any", responses_for)(
        mixture, kerr=False, plasma=OptmatRate("Ar", WAVELENGTH), wavelength=WAVELENGTH
    )
    assert isinstance(response.rate, OptmatRate)
    with pytest.raises(MaterialError, match="names 'N2'"):
        responses_for(
            mixture,
            kerr=False,
            plasma=OptmatRate("N2", WAVELENGTH),
            wavelength=WAVELENGTH,
        )


# -------------------------------------------------------- the reference field


@pytest.mark.parametrize("envelope", [False, True])
def test_the_capillary_binds_the_rate_to_the_inputs_peak_field(envelope):
    simulation = capillary(envelope=envelope)
    response = plasma_of(simulation)
    rate = response.rate
    assert isinstance(rate, OptmatRate)
    peak = input_peak(simulation)
    # The peak that sets the default table range is the reference field.
    assert simulation.settings.ionisation_field_range == default_field_range(peak)
    assert rate.reference_field == peak
    assert rate.field_range == default_field_range(peak)
    assert rate.ionisation_potential == float(
        rate.optmat_model.energy_per_event(rate.effective_reference_field)
    )
    assert response.potential == rate.ionisation_potential
    channel = simulation.translate().responses.population_channels[0]
    assert channel.ionisation_potential == rate.ionisation_potential


def test_a_given_table_range_keeps_the_inputs_peak_as_the_reference():
    simulation = capillary(ionisation_field_range=(1e9, 2e10))
    rate = plasma_of(simulation).rate
    assert rate.reference_field == input_peak(simulation)
    # The peak lies above the given table, so the energy is taken at its top.
    assert rate.effective_reference_field == 2e10
    assert simulation.settings.ionisation_field_range == (1e9, 2e10)


def test_an_explicit_rate_keeps_its_own_reference_field():
    rate = OptmatRate("Ar", WAVELENGTH, reference_field=1e10)
    simulation = capillary(plasma=rate)
    assert plasma_of(simulation).rate is rate
    assert plasma_of(simulation).potential == rate.ionisation_potential


def test_without_a_selection_the_rate_is_optmats_too():
    # materials=None is MaterialSelection(): there is no other rate (SEP 3 Phase 4b).
    simulation = capillary(materials=None)
    rate = plasma_of(simulation).rate
    assert isinstance(rate, OptmatRate)
    assert (
        simulation.material_manifest().hashed()
        == capillary().material_manifest().hashed()
    )


def test_an_input_above_the_maximum_field_warns_once():
    rate = OptmatRate("Ar", WAVELENGTH)
    with pytest.warns(MaterialRangeWarning, match="exceeds the maximum field") as seen:
        simulation = capillary(energy=1e-3)
        simulation.translate()
        simulation.metadata()
    assert len([one for one in seen if "exceeds the maximum" in str(one.message)]) == 1
    assert input_peak(simulation) > rate.maximum_field


def test_a_new_input_is_checked_and_keeps_the_reference_field():
    simulation = capillary()
    reference = plasma_of(simulation).rate.reference_field
    stronger = GaussianPulse(wavelength=WAVELENGTH, fwhm=30e-15, energy=1e-3)
    with pytest.warns(MaterialRangeWarning, match="exceeds the maximum field"):
        updated = simulation.with_input(stronger)
    assert plasma_of(updated).rate.reference_field == reference


def test_an_elliptical_free_space_input_warns_where_the_keldysh_parameter_is_large():
    grid = FieldGrid.from_limits(1e-3, WAVELENGTH, (600e-9, 1.2e-6), 150e-15)
    space = FreeGrid(3e-4, 12, 4e-4, 16, window_factor=0)
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
        backend="numpy_ref",
        ionisation_field_range=(1e7, 1e11),
    )
    temporal = GaussianPulse(wavelength=WAVELENGTH, fwhm=30e-15, energy=3e-9)
    plasma = [PlasmaResponse("Ar", OptmatRate("Ar", WAVELENGTH))]
    medium = SpatialMedium("Ar", 1.0)
    circular = SpatialPulse(temporal, 80e-6, jones=(1.0, 1.0j))
    with pytest.warns(MaterialRangeWarning, match="not linearly polarised"):
        Simulation.from_free(
            grid, space, medium, plasma, circular, settings=settings, noise=None
        )
    linear = SpatialPulse(temporal, 80e-6, jones=(1.0, 0.0))
    Simulation.from_free(
        grid, space, medium, plasma, linear, settings=settings, noise=None
    )
    array = linear.build(grid, space, medium=medium)
    Simulation.from_free(
        grid, space, medium, plasma, array, settings=settings, noise=None
    )


def test_a_route_without_a_transverse_rule_is_not_checked():
    simulation = capillary()
    simulation.settings = dataclasses.replace(simulation.settings, quadrature=None)
    _check_rate_inputs(simulation, simulation.initial)


def test_a_new_fill_keeps_the_runs_reference_field():
    simulation = capillary()
    original = plasma_of(simulation)
    mixture = simulation.modes[0].mixture
    updated = simulation.with_mixture(mixture.with_fill(pressure=2.0))
    response = plasma_of(updated)
    assert response.potential == original.rate.ionisation_potential


# --------------------------------------------------------------- the record


def test_the_manifest_records_each_rates_policies():
    simulation = capillary()
    manifest = simulation.material_manifest()
    rate = plasma_of(simulation).rate
    (entry,) = manifest.policies["ionisation"]
    assert entry == json.loads(json.dumps(rate.manifest_entry()))
    assert entry["threshold"]["kind"] == "negligible-rate"
    assert entry["energy_loss"]["reference_field"] == rate.reference_field
    record = rate.material_records[0]
    assert record in manifest.records
    hashed = manifest.hashed()
    assert hashed is not None
    assert hashed["policies"]["ionisation"] == [entry]
    # Another reference field is another computation.
    other = capillary(energy=50e-6).material_manifest().hashed()
    assert other is not None
    assert other["policies"]["ionisation"] != hashed["policies"]["ionisation"]


@pytest.mark.medium
@pytest.mark.parametrize("envelope", [False, True])
def test_ten_steps_report_the_largest_field_each_population_saw(envelope, tmp_path):
    simulation = capillary(envelope=envelope)
    path = tmp_path / "argon.h5"
    simulation.run(path, progress="none")
    rate = plasma_of(simulation).rate
    (report,) = simulation._peak_fields
    stored: Any
    with open_file(path) as stored:
        recorded = np.max(stored["stats/species_peak_ionisation_rate"])
        stored_manifest = stored.metadata.materials
        stored_hash = stored.metadata.configuration_hash()
        annotated = stored.annotations[PEAK_FIELDS_ANNOTATION]
    assert report.peak_rate == recorded
    envelope_plasma = simulation.translate().responses.envelope_plasma
    expected = (
        (lambda e: float(cycle_moments(e, 0.0, envelope_plasma.channels[0].cycle)[0]))
        if envelope
        else (lambda e: float(rate(e)))
    )
    assert expected(report.largest_field) == pytest.approx(recorded, rel=1e-9)
    assert not report.reached_maximum
    assert report.constant_energy == rate.ionisation_potential
    manifest = simulation.material_manifest()
    assert manifest.peak_fields == (json.loads(json.dumps(report.to_dict())),)
    round_trip = MaterialManifest.from_dict(manifest.to_dict())
    assert round_trip.peak_fields == manifest.peak_fields
    # The file's record, written before the run, carries the inputs, not the result.
    assert stored_manifest.policies["ionisation"] == manifest.policies["ionisation"]
    assert stored_manifest.peak_fields == ()
    # The result is the file's post-run annotation (SEP 3, Q13), outside the record:
    # the stored digest is the run's, and the file still resumes as this run.
    assert annotated == [json.loads(json.dumps(report.to_dict()))]
    assert stored_hash == simulation.metadata().configuration_hash()
    assert check_cache(path, simulation.metadata())
    # The input stays below the maximum field, and the table's last knot at it is
    # not a field beyond it.
    assert not any(
        "maximum field" in part
        for entry in manifest.ranges
        for part in entry["outside"]
    )


@pytest.mark.medium
def test_a_run_that_reaches_the_maximum_field_warns_once():
    # Without its static correction argon's model ends at the barrier-suppression
    # field, which this input's peak passes; the run itself stays a gentle one.
    from optmat.gas.ionisation_rates import ionisation_rate_model  # noqa: PLC0415

    model = ionisation_rate_model("Ar", WAVELENGTH).with_parameters(
        static_correction="none", static_constants=()
    )
    selection = MaterialSelection(ionisation=model)
    # Its own steps: the plasma at twice the energy outgrows ten fixed ones under
    # the default Adams history, which a replay cannot reject.
    with pytest.warns(MaterialRangeWarning) as seen:
        simulation = capillary(selection, energy=200e-6, schedule=None)
        output = simulation.run(progress="none")
    texts = [str(one.message) for one in seen]
    assert sum("exceeds the maximum field" in text for text in texts) == 1
    assert sum("reached the maximum field" in text for text in texts) == 1
    (report,) = simulation._peak_fields
    rate = plasma_of(simulation).rate
    assert input_peak(simulation) > rate.maximum_field
    assert report.reached_maximum
    assert report.largest_field == rate.maximum_field
    assert output.n_saved == 2


@pytest.mark.medium
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_a_run_without_peak_rates_reports_nothing():
    spec = default_spec("multimode")
    simulation = capillary(stats=spec)
    simulation.run(progress="none")
    assert simulation._peak_fields == ()
    # One population's total peak rate serves as well as its species row.
    total = dataclasses.replace(spec, names=(*spec.names, "peak_ionisation_rate"))
    single = capillary(stats=total)
    output = single.run(progress="none")
    (report,) = single._peak_fields
    assert report.peak_rate == np.max(output["stats/peak_ionisation_rate"])
    # An in-memory output carries the annotation as a file does.
    assert output.annotations[PEAK_FIELDS_ANNOTATION] == [
        json.loads(json.dumps(report.to_dict()))
    ]
    # A caller's rate (the test-only ADK oracle) has no optmat field range to report.
    callers = capillary(plasma=adk_oracle("Ar"))
    unannotated = callers.run(progress="none")
    assert callers._peak_fields == ()
    assert PEAK_FIELDS_ANNOTATION not in unannotated.annotations
