# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 3, Phase 1b at the drivers: the carrier Kerr coefficient and the silica bundle.

A capillary under a selection takes each optmat species' Kerr coefficient at the
grid's reference wavelength and records it, with the THG discrepancy; a GNLSE run
takes silica's Raman fraction and shape from the bundle and records the fraction it
held back and where it came from. Runs are built and translated, or take at most ten
fixed steps on ``numpy_ref``.
"""

from typing import Any

import numpy as np
import pytest

from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.exceptions import DriverError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import species
from solprop.physdata.glasses import SilicaBundle
from solprop.physdata.nonlinear_data import OptmatKerr
from solprop.physdata.selection import MaterialSelection
from solprop.stepper import FixedSchedule

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)

SELECTION = MaterialSelection()
LENGTH = 4e-4
STEPS = 10
CARRIER = 800e-9


def capillary(
    materials: MaterialSelection | None = SELECTION, gas: Any = "Ar", **changes: Any
) -> Any:
    """A short argon capillary, built and not run."""
    grid = changes.pop("grid", None) or FieldGrid.from_limits(
        LENGTH, CARRIER, (300e-9, 3e-6), 100e-15
    )
    options: dict[str, Any] = {
        "lambda0": CARRIER,
        "grid": grid,
        "absorber": absorber(grid, boundary_length=1.0),
        "fwhm": 30e-15,
        "energy": 1e-6,
        "plasma": False,
        "n_r": 8,
        "n_save": 2,
        "backend": "numpy_ref",
        "materials": materials,
    } | changes
    return Simulation.from_capillary(
        50e-6, LENGTH, gas, 1.0, **{"noise": None, **options}
    )


def kerr_terms(simulation):
    return [one for one in simulation.responses if isinstance(one, KerrResponse)]


def test_a_single_gas_takes_optmats_coefficient_at_the_carrier():
    (kerr,) = kerr_terms(capillary())
    assert kerr.chi3 == OptmatKerr("Ar").gamma3(CARRIER)
    # The species carries optmat's law, no fixed coefficient (SEP 3 Phase 4b).
    assert species("Ar").gamma3 is None


def test_a_mixture_takes_each_populations_coefficient_at_the_carrier():
    terms = kerr_terms(capillary(gas={"He": 0.5, "Ar": 0.5}))
    assert [one.species for one in terms] == ["He", "Ar"]
    assert [one.chi3 for one in terms] == [
        OptmatKerr("He").gamma3(CARRIER),
        OptmatKerr("Ar").gamma3(CARRIER),
    ]


def test_the_manifest_records_the_law_the_carrier_and_the_thg_discrepancy():
    manifest = capillary().material_manifest()
    assert manifest.policies["electronic_response"] == {
        "law": "carrier-kerr",
        "carrier_wavelength": CARRIER,
    }
    (entry,) = manifest.electronic
    assert entry["label"] == "Ar" and entry["model"] == "hyperpolarizability/Ar"
    assert entry["gamma3"] == OptmatKerr("Ar").gamma3(CARRIER)
    assert entry["thg_ratio"] == pytest.approx(1.177, abs=5e-4)
    document = manifest.to_dict()
    assert document is not None
    assert document["electronic"] == [dict(entry)]
    hashed = manifest.hashed()
    assert hashed is not None and "electronic" not in hashed
    assert "hyperpolarizability/Ar" in str(hashed["records"])


def test_another_carrier_is_another_policy():
    grid = FieldGrid.from_limits(LENGTH, 1030e-9, (300e-9, 3e-6), 100e-15)
    other = capillary(grid=grid, lambda0=1030e-9).material_manifest()
    first = capillary().material_manifest()
    assert other.policies["electronic_response"]["carrier_wavelength"] == 1030e-9
    assert first.hashed() != other.hashed()


def test_a_carrier_that_is_not_the_grids_is_refused():
    with pytest.raises(DriverError, match=r"carrier_wavelength is 1\.03e-06 m"):
        capillary(materials=MaterialSelection(carrier_wavelength=1030e-9))
    # The same carrier, given explicitly, is accepted.
    capillary(materials=MaterialSelection(carrier_wavelength=CARRIER))
    with pytest.raises(DriverError, match=r"lambda0 is 1\.03e-06 m"):
        capillary(lambda0=1030e-9)


def test_a_gnlse_lambda0_that_is_not_the_grids_carrier_is_refused():
    grid = EnvelopeGrid.from_limits(1e-3, 1030e-9, (700e-9, 2e-6), 1e-12)
    common: dict[str, Any] = {
        "grid": grid,
        "wavelength_limits": None,
        "trange": None,
        "materials": SELECTION,
    }
    with pytest.raises(DriverError, match=r"lambda0 is 8e-07 m"):
        gnlse(lambda0=800e-9, **common)
    policy = gnlse(lambda0=1030e-9, **common).material_manifest().policies
    assert policy["gnlse_raman"]["carrier_wavelength"] == pytest.approx(
        1030e-9, rel=1e-12
    )


GNLSE: dict[str, Any] = {
    "lambda0": 835e-9,
    "wavelength_limits": (500e-9, 2e-6),
    "trange": 1e-12,
    "width": 50e-15,
    "peak_power": 1e3,
    "pulseshape": "sech",
    "shock": False,
    "schedule": FixedSchedule(np.full(STEPS, 1e-3 / STEPS)),
    # No collar: ten fixed steps are refused beside the default one, and these tests
    # are about the material policy, not boundaries.
    "absorber": False,
    "n_save": 2,
    "backend": "numpy_ref",
}


def gnlse(**changes: Any) -> Any:
    options: dict[str, Any] = GNLSE | changes
    return Simulation.from_gnlse(
        options.pop("gamma", 0.11),
        1e-3,
        options.pop("betas", [0.0, 0.0, -1.183e-26]),
        **{"noise": None, **options},
    )


def test_a_gnlse_run_records_the_held_back_fraction_and_its_origin():
    bundle = SilicaBundle.resolve()
    on = gnlse(raman=True, materials=SELECTION).material_manifest()
    assert on.policies["gnlse_raman"] == {
        "fraction": bundle.raman_fraction,
        "origin": "bundle",
        "raman": True,
        "bundle": "silica_bundle/default",
        "n2_origin": "user",
        "carrier_wavelength": pytest.approx(835e-9, rel=1e-12),
    }
    assert sorted(record.name for record in on.records) == [
        "kerr/SiO2",
        "silica_bundle/default",
        "silica_raman/hollenbeck_2002",
    ]
    off = gnlse(materials=SELECTION).material_manifest()
    assert off.policies["gnlse_raman"]["origin"] == "raman-off"
    assert off.records == ()
    held = gnlse(raman_fraction=0.18, materials=SELECTION).material_manifest()
    assert held.policies["gnlse_raman"]["fraction"] == 0.18
    assert held.policies["gnlse_raman"]["origin"] == "user"
    other = gnlse(raman=True, materials=MaterialSelection(raman="blow_1989"))
    assert other.material_manifest().hashed() != on.hashed()


def test_n2_recommended_needs_a_fibre_area_and_records_its_origin():
    grid = EnvelopeGrid.from_limits(1e-3, 1030e-9, (700e-9, 2e-6), 1e-12)
    fibre = SpecifiedMode.from_taylor([0.0, 0.0, 2e-26], grid.omega0, aeff=80e-12)
    simulation = gnlse(
        gamma=None,
        betas=None,
        n2="recommended",
        fibre=fibre,
        grid=grid,
        lambda0=1030e-9,
        wavelength_limits=None,
        trange=None,
        materials=SELECTION,
    )
    mapping = simulation.configuration["gnlse"]
    assert mapping.n2 == SilicaBundle.resolve().n2(1030e-9)
    policy = simulation.material_manifest().policies["gnlse_raman"]
    assert policy["n2_origin"] == "bundle" and policy["origin"] == "raman-off"


@pytest.mark.medium
def test_ten_gnlse_steps_with_the_bundle_match_the_same_fraction_given():
    # Both runs take the bundle's Hollenbeck shape (solprop's own shape is deleted,
    # SEP 3 Phase 4b); the bundle's fraction given as the caller's differs only in
    # its recorded origin, so the fields agree to roundoff amplified over ten steps
    # of a strongly nonlinear pulse.
    bundle = SilicaBundle.resolve()
    selected = gnlse(raman=True, materials=SELECTION).run()
    given = gnlse(raman=True, raman_fraction=bundle.raman_fraction).run()
    final, reference = selected.fields("fft")[-1], given.fields("fft")[-1]
    assert np.all(np.isfinite(final))
    np.testing.assert_allclose(
        final, reference, rtol=0, atol=1e-12 * np.max(abs(reference))
    )
    stored: Any = selected.metadata
    assert stored.materials.policies["gnlse_raman"]["origin"] == "bundle"
