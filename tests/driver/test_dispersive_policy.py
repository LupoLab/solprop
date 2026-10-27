# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The dispersive electronic policies through the driver and the run record (SEP 6).

``MaterialSelection.electronic`` picks the response: the capillary entry point, its
mixture route and its scans build SEP 6's dispersive response under a dispersive policy;
the constant-coefficient routes refuse one by name; and the manifest records the law,
the support record and the regions, which the resume key covers.
"""

import math
from typing import Any

import numpy as np
import pytest

from solprop.driver import Simulation, SimulationSettings
from solprop.driver.material_output import MIXTURE_FILTER_RULE
from solprop.exceptions import DriverError, OutputError
from solprop.grid import FieldGrid, FreeGrid
from solprop.nonlinear.responses import DispersiveKerrResponse, KerrResponse
from solprop.output import HDF5Output
from solprop.output.hdf5 import check_cache
from solprop.output.schema import SavedPlane
from solprop.physdata import MaterialSelection, Mixture
from solprop.physdata.nonlinear_data import OptmatKerr
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)

LENGTH = 4e-4


def capillary(policy: str = "gas-polynomial", gas: Any = "Ar", **overrides: Any):
    """Return a short capillary under an electronic policy, without propagating."""
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (190e-9, 3e-6), 100e-15)
    arguments: dict[str, Any] = {
        "lambda0": 800e-9,
        "grid": grid,
        "fwhm": 30e-15,
        "energy": 1e-6,
        "plasma": False,
        "raman": False,
        "n_save": 2,
        "backend": "numpy_ref",
        "materials": MaterialSelection(electronic=policy),
    } | overrides
    return Simulation.from_capillary(50e-6, LENGTH, gas, 1.0, **arguments)


@pytest.mark.parametrize(
    ("policy", "evaluator"),
    [("gas-polynomial", "polynomial"), ("symmetric-kerr", "symmetric")],
)
def test_a_single_gas_capillary_takes_the_dispersive_response(policy, evaluator):
    simulation = capillary(policy)
    (response,) = simulation.responses
    assert isinstance(response, DispersiveKerrResponse)
    assert response.evaluator == evaluator
    assert response.thg and response.species is None
    assert response.model == OptmatKerr("Ar").dispersive_model()
    assert isinstance(capillary("carrier-kerr").responses[0], KerrResponse)


def test_a_mixture_and_its_scans_take_one_response_per_population():
    simulation = capillary(gas={"He": 0.5, "Ar": 0.5}, thg=False)
    electronic = [
        one for one in simulation.responses if isinstance(one, DispersiveKerrResponse)
    ]
    assert [(one.species, one.model.material) for one in electronic] == [
        ("He", "He"),
        ("Ar", "Ar"),
    ]
    assert all(not one.thg for one in electronic)
    replaced = simulation.with_mixture(Mixture.fill({"He": 0.2, "Ar": 0.8}, 1.0))
    assert [
        one.species
        for one in replaced.responses
        if isinstance(one, DispersiveKerrResponse)
    ] == ["He", "Ar"]


def test_air_resolves_through_its_components():
    """Air has no hyperpolarizability of its own: each population carries its own."""
    simulation = capillary(gas="Air", raman=True)
    electronic = [
        one for one in simulation.responses if isinstance(one, DispersiveKerrResponse)
    ]
    assert len(electronic) >= 2
    assert all(one.model.material == one.species for one in electronic)
    leaf = simulation.translate().responses.dispersive_kerr
    assert leaf is not None
    assert leaf.labels == tuple(one.species for one in electronic)


def test_the_gnlse_refuses_a_dispersive_policy_by_name():
    with pytest.raises(DriverError, match=r"'gas-polynomial'.*GNLSE"):
        Simulation.from_gnlse(
            0.11,
            0.01,
            [0.0, 0.0, -1.183e-26],
            lambda0=835e-9,
            wavelength_limits=(500e-9, 2e-6),
            trange=1e-12,
            width=100e-15,
            peak_power=9.8,
            pulseshape="sech",
            n_save=2,
            materials=MaterialSelection(electronic="gas-polynomial"),
            noise=None,
        )


def test_a_constant_kerr_response_beside_a_dispersive_policy_is_refused():
    """A caller's responses are taken as they are, but not one law beside another."""
    grid = FieldGrid.from_limits(1e-4, 800e-9, (600e-9, 1.2e-6), 150e-15)
    space = FreeGrid(3e-4, 9, 4e-4, 12, window_factor=0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 8e-5
    )
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1e-4],
        dz_max=1e-4,
        dz_init=1e-5,
    )
    with pytest.raises(DriverError, match="constant KerrResponse"):
        Simulation.from_free(
            grid,
            space,
            SpatialMedium("Ar", 1.0),
            [KerrResponse(1e-51)],
            pulse,
            settings=settings,
            materials=MaterialSelection(electronic="symmetric-kerr"),
            noise=None,
        )


def test_the_manifest_records_the_law_the_support_and_the_regions():
    simulation = capillary("symmetric-kerr")
    manifest = simulation.material_manifest()
    policy: Any = manifest.policies["electronic_response"]  # pyright: ignore[reportOptionalSubscript]
    model = OptmatKerr("Ar").dispersive_model()
    assert policy["law"] == "symmetric-kerr"
    assert policy["evaluator"] == "symmetric" and policy["sector"] == "full"
    assert policy["tensor"] == "isotropic-kleinman"
    (species,) = policy["species"]
    assert species["support_digest"] == model.support_digest
    (entry,) = manifest.electronic
    assert entry["digest"] == model.model_digest
    assert entry["leg_bound"] == model.leg_bound
    assert entry["band_edge_linear"] == model.band_edge_linear
    regions: Any = entry["regions"]
    # The grid's active band reaches 1.1 omega_max, 172.7 nm, past argon's 189.4 nm
    # edge: all three regions.
    assert regions["bounded"] is not None and regions["best_effort"] is not None
    assert math.isclose(regions["band"][0], 190e-9 / 1.1, rel_tol=1e-12)
    assert model.support_digest in {record.digest for record in manifest.records}
    assert "carrier_wavelength" not in entry


def test_a_symmetric_mixture_records_its_filter_and_its_composition():
    """The mixture filter, its rule and each population's density, in the policy.

    A pure gas and the polynomial, which keep their own operators, record none, so
    their records are what they were.
    """
    manifest = capillary(
        "symmetric-kerr", gas={"He": 0.5, "Ar": 0.5}
    ).material_manifest()
    policy: Any = manifest.policies["electronic_response"]  # pyright: ignore[reportOptionalSubscript]
    (recorded,) = policy["mixture_filters"]
    assert recorded["rule"] == MIXTURE_FILTER_RULE
    assert recorded["region"] is None
    composition = recorded["composition"]
    assert [one["label"] for one in composition] == ["He", "Ar"]
    assert all(len(one["density"]) == 64 for one in composition)
    assert composition[0]["density"] != composition[1]["density"]
    assert [entry["filter"] for entry in manifest.electronic] == [
        {"kind": "mixture", "labels": ["He", "Ar"]}
    ] * 2
    for one in (
        capillary("symmetric-kerr").material_manifest(),
        capillary("gas-polynomial", gas={"He": 0.5, "Ar": 0.5}).material_manifest(),
    ):
        law: Any = one.policies["electronic_response"]  # pyright: ignore[reportOptionalSubscript]
        assert "mixture_filters" not in law
        assert all("filter" not in entry for entry in one.electronic)


def test_changing_a_symmetric_mixtures_composition_changes_its_filter_record():
    """The composition the filter is built from enters the resume key."""
    records = [
        capillary("symmetric-kerr", gas=gas).material_manifest().hashed()
        for gas in ({"He": 0.5, "Ar": 0.5}, {"He": 0.2, "Ar": 0.8})
    ]
    filters = [
        record["policies"]["electronic_response"]["mixture_filters"]  # pyright: ignore[reportOptionalSubscript, reportIndexIssue]
        for record in records
    ]
    assert filters[0] != filters[1]


def written(path, record):
    handler = HDF5Output(path)
    handler.open(record)
    handler.save(SavedPlane(z=0.0, field=np.ones((1, 1, record.grid.n_omega), complex)))
    handler.close()
    return path


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"policy": "carrier-kerr"}, {"policy": "gas-polynomial"}),
        ({"policy": "gas-polynomial"}, {"policy": "symmetric-kerr"}),
        ({"policy": "gas-polynomial"}, {"policy": "gas-polynomial", "thg": False}),
    ],
    ids=["carrier-to-dispersive", "evaluator", "sector"],
)
def test_changing_the_electronic_law_changes_the_resume_key(tmp_path, first, second):
    one = capillary(**first).metadata()
    two = capillary(**second).metadata()
    assert one.configuration_hash() != two.configuration_hash()
    path = written(tmp_path / "run.h5", one)
    assert check_cache(path, capillary(**first).metadata())
    with pytest.raises(OutputError, match=r"materials\.policies\.electronic"):
        check_cache(path, two)
