# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The material manifest: canonical form, its hashed part, the run log, compositions."""

import dataclasses
import math
from typing import Any

import numpy as np
import pytest
from optmat.gas.models.registry import default_model

from solprop.constants import C
from solprop.exceptions import MaterialError, MaterialRangeWarning
from solprop.physdata import Mixture, _optmat, fused_silica
from solprop.physdata.selection import (
    MaterialLog,
    MaterialManifest,
    MaterialRecord,
    MaterialSelection,
)
from solprop.physdata.sellmeier import OptmatSusceptibility

ARGON = default_model("Ar")
NARROWED = ARGON.with_parameters(intervals=((120e-9, math.inf),))


def record_of(model):
    """The plain record of an optmat model."""
    return _optmat.record(model)


def manifest(**changes) -> Any:
    """A small non-empty manifest, with fields replaced, loosely typed for reading."""
    fields = {
        "optmat_version": "0.1.1",
        "selection": MaterialSelection().to_dict(),
        "records": (record_of(ARGON), record_of(NARROWED)),
        "species": ({"label": "Ar", "species": "Ar", "records": ["x"]},),
        "solids": (
            {
                "material": "SiO2",
                "axis": "",
                "model": "fit/SiO2",
                "digest": "d" * 64,
                "transparent": True,
                "absorption_bound": 100.0,
                "transparency": {"largest_dropped_absorption": 1.0},
            },
        ),
        "policies": {"electronic": "recommended", "energy_loss": "peak-weighted"},
        "dilute_scaling": ({"label": "Ar", "relative_error": -2e-4},),
        "ranges": ({"material": "Ar", "outside": ["80 nm to 107.5 nm"]},),
        "citations": ({"object": "SellmeierGlass", "name": "x", "citation": "y"},),
    }
    return MaterialManifest(**(fields | changes))


def omega_of(wavelength):
    """Angular frequencies of vacuum wavelengths in m."""
    return 2 * np.pi * C / np.asarray(wavelength)


def test_the_empty_manifest_hashes_to_none_and_round_trips():
    empty = MaterialManifest()
    assert empty.empty
    assert empty.hashed() is None and empty.to_dict() is None
    assert MaterialManifest.from_dict(None) == empty
    # Records alone, or a selection alone, make a manifest non-empty.
    assert not MaterialManifest(records=(record_of(ARGON),)).empty
    assert not MaterialManifest(selection=MaterialSelection().to_dict()).empty


def test_records_are_deduplicated_by_digest_and_canonically_ordered():
    one = manifest(records=(record_of(NARROWED), record_of(ARGON), record_of(ARGON)))
    two = manifest(records=(record_of(ARGON), record_of(NARROWED)))
    assert one == two
    assert len(one.records) == 2
    assert one.records == tuple(
        sorted(one.records, key=lambda r: (r.material, r.kind, r.name, r.digest))
    )
    assert one.hashed() == two.hashed()


def test_entries_are_deduplicated_and_sorted_whatever_their_order():
    entries = ({"label": "b"}, {"label": "a"}, {"label": "b"})
    left = manifest(species=entries)
    right = manifest(species=tuple(reversed(entries)))
    assert left.species == right.species == ({"label": "a"}, {"label": "b"})


def test_records_sharing_a_name_are_keyed_apart_by_number():
    # A curated model and its override share their material, kind and name.
    keys = sorted(manifest().hashed()["records"])
    assert keys == ["Ar:gas_refractive:fit/Ar#1", "Ar:gas_refractive:fit/Ar#2"]


def test_the_whole_manifest_round_trips_through_plain_values():
    original = manifest()
    back: Any = MaterialManifest.from_dict(original.to_dict())
    assert back == original
    assert back.hashed() == original.hashed()


def test_a_malformed_stored_record_is_refused_by_name():
    document = manifest().to_dict()
    key = next(iter(document["records"]))
    del document["records"][key]["digest"]
    with pytest.raises(MaterialError, match="material record is malformed"):
        MaterialManifest.from_dict(document)


@pytest.mark.parametrize(
    "changes",
    [
        {"dilute_scaling": ({"label": "Ar", "relative_error": -3e-4},)},
        {"ranges": ()},
        {"citations": ()},
        {"selection": MaterialSelection(linear="fit/Ar").to_dict()},
    ],
    ids=["dilute", "ranges", "citations", "selection-names"],
)
def test_measurements_and_selection_names_stay_outside_the_hashed_part(changes):
    # What the selection resolved to is in the records; naming the same model
    # "recommended" or by its name must hash alike.
    assert manifest(**changes).hashed() == manifest().hashed()
    assert manifest(**changes).to_dict() != manifest().to_dict()


def test_a_solids_measured_transparency_is_recorded_but_not_hashed():
    solid = dict(manifest().solids[0])
    measured = manifest(solids=(solid | {"transparency": {"x": 2.0}},))
    assert measured.hashed() == manifest().hashed()
    rebound = manifest(solids=(solid | {"absorption_bound": 50.0},))
    assert rebound.hashed() != manifest().hashed()
    # An absorbing solid has no bound to hash.
    opaque = solid | {"transparent": False, "absorption_bound": None}
    hashed = manifest(solids=(opaque,)).hashed()["solids"]
    assert "absorption_bound" not in hashed["SiO2:-:fit/SiO2"]


@pytest.mark.parametrize(
    "changes",
    [
        {"optmat_version": "0.1.2"},
        {"records": (record_of(ARGON),)},
        {"policies": {"electronic": "carrier-kerr", "energy_loss": "peak-weighted"}},
        {"compositions": ({"name": "dry air (N2, O2)"},)},
    ],
    ids=["version", "identity", "policy", "composition"],
)
def test_identities_policies_and_the_version_are_hashed(changes):
    assert manifest(**changes).hashed() != manifest().hashed()


def test_a_run_log_keeps_warnings_once_across_reentries():
    log = MaterialLog()
    with pytest.warns(MaterialRangeWarning) as caught:
        for _ in range(2):
            with _optmat.material_context(log) as inside:
                assert inside is log and _optmat.open_log() is log
                _optmat.warn_once(("x",), "once", MaterialRangeWarning)
                # Re-entering the same log while it is open shares it.
                with _optmat.material_context(log):
                    _optmat.warn_once(("x",), "once", MaterialRangeWarning)
    assert len(caught) == 1
    assert ("x",) in log.seen
    assert _optmat.open_log() is None


def test_a_run_log_resumed_inside_another_context_records_into_both():
    run = MaterialLog()
    argon = OptmatSusceptibility.resolve("Ar")
    with (
        pytest.warns(MaterialRangeWarning, match="below its shortest") as caught,
        _optmat.material_context() as outer_log,
    ):
        with _optmat.material_context(run):
            argon.susceptibility(omega_of([80e-9, 800e-9]))
        # The outer context has seen the warning; the run's log is not resumed.
        argon.susceptibility(omega_of([90e-9, 800e-9]))
    assert len(caught) == 1
    assert run.range_entries() == (
        {
            "material": "Ar",
            "model": "fit/Ar",
            "digest": ARGON.identity.digest,
            "outside": ["80 nm to 107.5 nm, below its shortest qualified wavelength"],
        },
    )
    # The outer log unites both evaluations' bands.
    outer: Any = outer_log.range_entries()[0]
    assert len(outer["outside"]) == 2


def test_a_range_without_a_known_band_is_recorded_without_parts():
    log = MaterialLog()
    with _optmat.material_context(log):
        _optmat._record_range(ARGON, "Ar", None)
    entry: Any = log.range_entries()[0]
    assert entry["outside"] == []
    # Outside every context there is nothing to record into.
    _optmat._record_range(ARGON, "Ar", None)


def test_air_under_a_selection_records_its_named_composition():
    selected = Mixture.air(1.0, selection=MaterialSelection())
    assert selected.composition == "dry air (N2, O2, Ar)"
    assert selected.serialise()["composition"] == "dry air (N2, O2, Ar)"
    pair = Mixture.air(1.0, argon=False, selection=MaterialSelection())
    assert pair.composition == "dry air (N2, O2)"
    assert Mixture.air(1.0).composition == "dry air (N2, O2, Ar)"  # the default
    # Fractions given by the caller name no composition.
    given = Mixture.air(
        1.0, fractions={"N2": 0.8, "O2": 0.2}, selection=MaterialSelection()
    )
    assert given.composition is None
    # A mixture derived with other densities does not keep it.
    assert selected.with_densities({"N2": 1e25}).composition is None


def test_a_named_composition_needs_fill_fractions():
    independent = Mixture.air(1.0).with_densities({"N2": 1e25})
    with pytest.raises(MaterialError, match="needs the fill fractions"):
        dataclasses.replace(independent, composition="dry air (N2, O2, Ar)")


def test_a_record_built_from_a_solid_matches_its_adapter():
    silica: Any = fused_silica(model="recommended")
    entry = silica.material_records[0]
    assert isinstance(entry, MaterialRecord)
    assert MaterialManifest(records=(entry,)).records == (entry,)
