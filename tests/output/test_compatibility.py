# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The resume comparison: classes, tolerance, neutral values and normalisation (SEP 42).

Structural (P) and rounding (R) class tests of :mod:`solprop.output.compatibility`, on
records built by hand where a rule is clearest that way and on a real route's record
where the rule is about what a route records.
"""

import copy
import json
import math

import h5py
import numpy as np
import pytest

from solprop.exceptions import OutputError
from solprop.output import compatibility
from solprop.output.compatibility import (
    RECORD_RTOL,
    Neutral,
    PathClass,
    classify,
    compare_records,
    record_paths,
)
from solprop.output.hdf5 import _stored_stat_shapes
from solprop.output.schema import RunMetadata, _example_metadata, serialise
from solprop.stepper.multistep import ComparisonMultistep


def array(values, dtype="float64"):
    """Return the record of an array, as serialise writes it."""
    return serialise(np.asarray(values, dtype=dtype))


def ulps(value, count):
    """Return `value` moved by `count` units in the last place."""
    for _ in range(count):
        value = math.nextafter(value, math.inf)
    return value


def paths(changes):
    return [change.path for change in changes]


# ------------------------------------------------------------------- tolerance


def test_a_dispersion_change_at_a_small_scale_is_refused():
    # β3 doubled and β4's sign flipped: a scale taken from the array's largest
    # element (β2) would let both through, which is why the test is per element.
    stored = {"configuration": {"betas": array([-2.0e-26, 1.0e-40, 1.0e-55])}}
    requested = {"configuration": {"betas": array([-2.0e-26, 2.0e-40, -1.0e-55])}}
    comparison = compare_records(stored, requested)
    assert not comparison.continues
    [change] = comparison.refused
    assert change.path == ("configuration", "betas")
    assert change.elements == 2


@pytest.mark.parametrize("name", ["lambda0", "trange", "energy", "fwhm"])
def test_a_few_ulps_of_a_scalar_agree(name):
    value = {"lambda0": 800e-9, "trange": 400e-15, "energy": 50e-6, "fwhm": 30e-15}[
        name
    ]
    stored = {"configuration": {name: value}}
    requested = {"configuration": {name: ulps(value, 4)}}
    assert compare_records(stored, requested).identical


def test_a_difference_above_the_tolerance_is_seen():
    stored = {"configuration": {"energy": 1.0}}
    requested = {"configuration": {"energy": 1.0 + 10 * RECORD_RTOL}}
    [change] = compare_records(stored, requested).refused
    assert (
        change.describe() == f"configuration.energy 1.0 -> {1.0 + 10 * RECORD_RTOL!r}"
    )


def test_a_named_tail_agrees_where_its_per_element_test_would_not():
    # An absorber rate's tail of rounding: the small element moves by far more than
    # its own size, but by less than the tolerance of the array's largest.
    rate_one = array([1.0, 1e-20])
    rate_two = array([1.0, 3e-20])
    tail = ("configuration", "absorber", "spectral_rate")
    other = ("configuration", "absorber", "band_mask")

    def record(rate, key):
        return {"configuration": {"absorber": {key[-1]: rate}}}

    assert compare_records(record(rate_one, tail), record(rate_two, tail)).identical
    assert not compare_records(
        record(rate_one, other), record(rate_two, other)
    ).continues


def test_nan_infinity_complex_and_integer_cases():
    nan, inf = math.nan, math.inf
    same = {"values": array([1.0, nan, inf, -inf])}
    assert compare_records(same, copy.deepcopy(same)).identical
    moved = {"values": array([1.0, 1.0, inf, -inf])}
    assert not compare_records(same, moved).continues
    flipped = {"values": array([1.0, nan, -inf, -inf])}
    assert not compare_records(same, flipped).continues
    # Complex values by modulus, and a complex scalar.
    one = {
        "field": array([1 + 1j, 1e-3j], dtype="complex128"),
        "z": {"__complex__": [1.0, 2.0]},
    }
    two = {
        "field": array([1 + 1j + 1e-14, 1e-3j], dtype="complex128"),
        "z": {"__complex__": [1.0, ulps(2.0, 2)]},
    }
    assert compare_records(one, two).identical
    # An integer and a float compare as numbers; two integers must be equal.
    assert compare_records({"pressure": 1}, {"pressure": 1.0}).identical
    assert not compare_records({"n_r": 16}, {"n_r": 17}).continues
    assert not compare_records(
        {"n": array([1, 2], "int64")}, {"n": array([1, 3], "int64")}
    ).continues
    # A boolean is not a number.
    assert not compare_records({"flag": True}, {"flag": 1}).continues


def test_shape_and_dtype_changes_are_differences():
    assert not compare_records({"a": array([1.0, 2.0])}, {"a": array([1.0])}).continues
    assert not compare_records(
        {"a": array([1.0], "float64")}, {"a": array([1.0], "float32")}
    ).continues
    assert not compare_records({"a": [1, "x"]}, {"a": [1, "x", "y"]}).continues


def test_exact_paths_take_no_tolerance():
    stored = {
        "grid": {"length": 0.1},
        "configuration": {"save_planes": array([0.05, 0.1])},
    }
    requested = copy.deepcopy(stored)
    requested["grid"]["length"] = 0.1 + 1e-13 * 0.1
    comparison = compare_records(stored, requested)
    assert paths(comparison.layout) == [("grid", "length")]
    requested = copy.deepcopy(stored)
    requested["configuration"]["save_planes"] = array([0.05, ulps(0.1, 1)])
    assert paths(compare_records(stored, requested).layout) == [
        ("configuration", "save_planes")
    ]


# ---------------------------------------------------------------------- classes


def kind(path):
    """Return the class of a path, compared where no row matches."""
    row = classify(path)
    return PathClass.COMPARED if row is None else row.kind


def test_rows_cover_their_subtrees():
    assert kind(("spatial", "axes", 0)) is PathClass.LAYOUT
    assert kind(("materials", "records", "x", "sources", 0)) is PathClass.IGNORED
    assert kind(("stepper", "limits", "dz_max")) is PathClass.NUMERICS
    assert kind(("grid", "dt")) is PathClass.LAYOUT
    # An unclassified path is compared.
    assert classify(("configuration", "medium", "pressure")) is None


def test_the_mode_count_and_the_noise_presence_are_layout_and_mode_data_is_not():
    mode = {"label": {"family": "HE", "radial_order": 1}, "kind": "solprop.Mode"}
    other = copy.deepcopy(mode)
    other["label"]["radial_order"] = 2
    comparison = compare_records({"modes": [mode]}, {"modes": [other]})
    assert paths(comparison.layout) == [("modes", 0, "label", "radial_order")]
    comparison = compare_records({"modes": [mode]}, {"modes": [mode, mode]})
    assert paths(comparison.layout) == [("modes",)]
    kind = copy.deepcopy(mode)
    kind["kind"] = "solprop.OtherMode"
    assert paths(compare_records({"modes": [mode]}, {"modes": [kind]}).refused) == [
        ("modes", 0, "kind")
    ]
    noise = {"model": "input", "seed": 1, "spectrum": array([1.0])}
    assert paths(compare_records({"noise": noise}, {"noise": None}).layout) == [
        ("noise",)
    ]


def test_stepper_changes_are_numerics_recorded_as_plain_json():
    stored = {"stepper": {"controller": {"rtol": 1e-6}, "limits": {"dz_max": math.inf}}}
    requested = {"stepper": {"controller": {"rtol": 1e-5}, "limits": {"dz_max": 1e-3}}}
    comparison = compare_records(stored, requested)
    assert comparison.continues
    assert not comparison.identical
    record = comparison.numerics_record()
    assert record == {
        "stepper.controller.rtol": [1e-6, 1e-5],
        "stepper.limits.dz_max": ["inf", 1e-3],
    }
    json.dumps(record, allow_nan=False)
    assert "the numerics changed: stepper: controller.rtol" in comparison.summary()


def test_a_launch_realisation_is_ignored_only_where_compared_parameters_made_it():
    pulse = {"__class__": "solprop.pulses.GaussianPulse", "energy": 1e-6}
    one = {"configuration": {"pulse": pulse, "initial": array([1.0, 2.0])}}
    two = {"configuration": {"pulse": pulse, "initial": array([1.0, 2.5])}}
    assert compare_records(one, two).identical
    # An array launch has no other record: the array is compared.
    one = {"configuration": {"initial": array([1.0, 2.0])}}
    two = {"configuration": {"initial": array([1.0, 2.5])}}
    assert paths(compare_records(one, two).refused) == [("configuration", "initial")]
    # A changed pulse parameter is refused even though the stored field supersedes it.
    changed = {"__class__": "solprop.pulses.GaussianPulse", "energy": 2e-6}
    one = {"configuration": {"pulse": pulse, "initial": array([1.0])}}
    two = {"configuration": {"pulse": changed, "initial": array([2.0])}}
    assert paths(compare_records(one, two).refused) == [
        ("configuration", "pulse", "energy")
    ]


def test_a_noise_spectrum_is_ignored_only_for_seeded_input_noise():
    def noise(model, seed, value):
        return {"noise": {"model": model, "seed": seed, "spectrum": array([value])}}

    assert compare_records(noise("input", 3, 1.0), noise("input", 3, 2.0)).identical
    refused = compare_records(
        noise("input", None, 1.0), noise("input", None, 2.0)
    ).refused
    assert paths(refused) == [("noise", "spectrum")]
    refused = compare_records(
        noise("modified", 3, 1.0), noise("modified", 3, 2.0)
    ).refused
    assert paths(refused) == [("noise", "spectrum")]
    # A different seed is refused by the seed.
    refused = compare_records(noise("input", 3, 1.0), noise("input", 4, 2.0)).refused
    assert paths(refused) == [("noise", "seed")]


def test_a_callable_says_why_it_differs():
    one = {
        "configuration": {
            "loss": {"__callable__": "<function f at 0x1>", "reproducible": False}
        }
    }
    two = {
        "configuration": {
            "loss": {"__callable__": "<function f at 0x2>", "reproducible": False}
        }
    }
    summary = compare_records(one, two).summary()
    assert "configuration.loss is recorded by its repr" in summary
    assert "pass an object that carries its data" in summary


# ------------------------------------------------------------------ normalisation


def manifest(name, digest, *, solid_model="sellmeier/SiO2", status="validated"):
    record = {
        "name": name,
        "kind": "gas_refractive",
        "material": "Ar",
        "digest": digest,
        "status": status,
        "status_reason": "",
        "optmat_version": "0.9.1",
        "modified": [],
        "base": "",
        "sources": ["source_one"],
        "reference_temperature": None,
    }
    solid = {
        "material": "SiO2",
        "axis": "o",
        "model": solid_model,
        "digest": "s1",
        "transparent": False,
    }
    return {
        "materials": {
            "optmat_version": "0.9.1",
            "records": {f"Ar:gas_refractive:{name}": record},
            "solids": {f"SiO2:o:{solid_model}": solid},
            "policies": {},
        }
    }


def test_a_renamed_model_with_the_same_numbers_is_the_same_model():
    stored = manifest("fit/Ar", "d1")
    renamed = manifest("fit2/Ar", "d1", solid_model="fit/SiO2", status="provisional")
    renamed["materials"]["optmat_version"] = "0.9.2"
    assert compare_records(stored, renamed).identical


def test_a_model_whose_numbers_changed_is_refused_by_name():
    comparison = compare_records(manifest("fit/Ar", "d1"), manifest("fit/Ar", "d2"))
    assert not comparison.continues
    summary = comparison.summary()
    assert "optmat model Ar gas_refractive (digest d2) only in this run" in summary
    assert (
        "optmat model Ar gas_refractive (digest d1) only in the stored run" in summary
    )


def test_optmat_objects_are_reduced_to_a_marker():
    one = {
        "configuration": {"glass": {"__class__": "optmat.models.Sellmeier", "b": 1.0}}
    }
    two = {
        "configuration": {"glass": {"__class__": "optmat.models.Sellmeier", "b": 2.0}}
    }
    assert compare_records(one, two).identical
    three = {"configuration": {"glass": {"__class__": "solprop.Glass", "b": 1.0}}}
    assert not compare_records(one, three).continues


# --------------------------------------------------------------- absent and neutral


def with_neutral(monkeypatch, *entries):
    parsed = tuple((compatibility._pattern(entry.pattern), entry) for entry in entries)
    monkeypatch.setattr(compatibility, "_NEUTRAL_PARSED", parsed)


def test_an_absent_path_is_a_difference_unless_it_has_a_neutral_value(monkeypatch):
    stored = {"stepper": {"limits": {"dz_max": 1.0}}}
    requested = {"stepper": {"limits": {"dz_max": 1.0, "new_cap": 2.0}}}
    [change] = compare_records(stored, requested).numerics
    assert change.absent == "stored"
    assert change.describe() == "stepper.limits.new_cap (absent from the stored record)"
    with_neutral(monkeypatch, Neutral("stepper.limits.new_cap", 2.0, "0.2.0"))
    comparison = compare_records(stored, requested)
    assert comparison.identical
    assert comparison.neutral == (("stepper", "limits", "new_cap"),)
    # A neutral value is compared with the present one like any other value.
    requested["stepper"]["limits"]["new_cap"] = 3.0
    assert paths(compare_records(stored, requested).numerics) == [
        ("stepper", "limits", "new_cap")
    ]


def test_a_stored_record_is_read_as_written(monkeypatch):
    metadata = _example_metadata()
    document = json.loads(metadata.to_json())
    # A field this release does not know, and one it lacks with a default.
    document["grid"]["future_field"] = 1.0
    del document["stepper"]["limits"]["stretch"]
    stored = RunMetadata.from_json(json.dumps(document))
    # The record digests to its own text, not to what today's code would write.
    grid = stored.hashed()["grid"]
    assert isinstance(grid, dict) and grid["future_field"] == 1.0
    assert stored.configuration_hash() != metadata.configuration_hash()
    assert stored.grid == metadata.grid
    comparison = stored.compare(metadata)
    assert paths(comparison.layout) == [("grid", "future_field")]
    assert paths(comparison.numerics) == [("stepper", "limits", "stretch")]
    # A required field lacking with a neutral value reads as that value.
    del document["grid"]["n_t"]
    with pytest.raises(OutputError, match=r"lacks grid\.n_t"):
        RunMetadata.from_json(json.dumps(document))
    with_neutral(monkeypatch, Neutral("grid.n_t", metadata.grid.n_t, "0.2.0"))
    assert RunMetadata.from_json(json.dumps(document)).grid.n_t == metadata.grid.n_t


def test_a_round_trip_keeps_the_digest():
    metadata = _example_metadata()
    back = RunMetadata.from_json(metadata.to_json())
    assert back.configuration_hash() == metadata.configuration_hash()
    assert back.compare(metadata).identical
    assert metadata.compare(back).identical


def test_every_path_of_a_record_is_walked_once():
    record = _example_metadata().hashed()
    walked = list(record_paths(record))
    assert len(walked) == len(set(walked))
    assert ("configuration", "gamma") in walked


# ------------------------------------------------------------------ edges of the walk


def test_patterns_match_as_written():
    assert repr(compatibility.ANY) == "*"
    assert repr(compatibility.ANY_DEPTH) == "**"
    pattern = compatibility._pattern("**.citation")
    assert compatibility._full_matches(pattern, ("a", "b", "citation"))
    assert not compatibility._full_matches(pattern, ())
    assert not compatibility._full_matches(compatibility._pattern("a.b"), ("a",))
    assert not compatibility._prefix_matches(compatibility._pattern("a.b"), ("a",))


def test_a_moved_class_is_renamed_before_the_walk(monkeypatch):
    monkeypatch.setattr(compatibility, "CLASS_RENAMES", {"old.Kerr": "new.Kerr"})
    one = {"configuration": {"kerr": {"__class__": "old.Kerr", "chi3": 1.0}}}
    two = {"configuration": {"kerr": {"__class__": "new.Kerr", "chi3": 1.0}}}
    assert compare_records(one, two).identical


def test_manifest_entries_without_a_digest_key_or_solids_are_walked_as_they_are():
    def record(records):
        return {"materials": {"records": records, "policies": {}}}

    # Two models with the same numbers under one identity are both kept.
    twins = {
        "a": {"material": "Ar", "kind": "k", "digest": "d", "name": "a"},
        "b": {"material": "Ar", "kind": "k", "digest": "d", "name": "b"},
    }
    assert compare_records(record(twins), record(twins)).identical
    assert compare_records(record({"x": "odd"}), record({"x": "odd"})).identical
    assert compare_records(record(None), record(None)).identical
    # A record lacking an ignored field differs in nothing that matters.
    full = dict(twins["a"], sources=["s"])
    assert compare_records(record({"a": full}), record({"a": twins["a"]})).identical


def test_malformed_numbers_compare_as_text():
    bad_complex = {"z": {"__complex__": [1.0]}}
    assert compare_records(bad_complex, copy.deepcopy(bad_complex)).identical
    ragged = {"values": [[1.0], [1.0, 2.0]]}
    assert compare_records(ragged, copy.deepcopy(ragged)).identical
    mismatched = {
        "a": {
            "dtype": "complex128",
            "shape": [2],
            "__ndarray__": [1.0, 2.0],
            "__imaginary__": [0.0],
        }
    }
    assert compare_records(mismatched, copy.deepcopy(mismatched)).identical
    changed = copy.deepcopy(mismatched)
    changed["a"]["__ndarray__"] = [1.0, 3.0]
    assert not compare_records(mismatched, changed).continues


def test_changes_are_summarised_by_what_they_hold():
    enum = {"__enum__": "solprop.Kind", "value": "a"}
    digest = {
        "__class__": "solprop.output.schema.ArrayDigest",
        "sha256": "0" * 64,
        "dtype": "float64",
        "shape": [3],
    }
    other_digest = dict(digest, sha256="1" * 64)
    one = {"configuration": {"kind": enum, "big": digest, "obj": {"__class__": "x.Y"}}}
    two = {
        "configuration": {
            "kind": dict(enum, value="b"),
            "big": other_digest,
            "obj": 3,
        }
    }
    summary = compare_records(one, two).summary()
    assert "kind 'a' -> 'b'" in summary
    assert "big 'digest 000000000000' -> 'digest 111111111111'" in summary
    assert "obj '<x.Y>' -> 3" in summary


def test_the_numerics_record_writes_every_value_as_plain_json():
    one = {"stepper": {"schedule": array([1.0, 2.0]), "z": {"__complex__": [1.0, 2.0]}}}
    two = {"stepper": {"schedule": array([1.0, 2.5]), "z": {"__complex__": [1.0, 3.0]}}}
    record = compare_records(one, two).numerics_record()
    assert record["stepper.schedule"] == ["shape [2]", "shape [2]"]
    assert record["stepper.z"] == ["(1+2j)", "(1+3j)"]
    json.dumps(record, allow_nan=False)


def test_a_stored_record_lacking_stepper_entries_or_materials(monkeypatch):
    metadata = _example_metadata()
    document = json.loads(metadata.to_json())
    del document["materials"]
    del document["stepper"]["method"]
    stored = RunMetadata.from_json(json.dumps(document))
    assert "materials" not in stored.hashed()
    assert stored.stepper.method is None
    del document["stepper"]["tableau"]
    with pytest.raises(OutputError, match=r"lacks stepper\.tableau"):
        RunMetadata.from_json(json.dumps(document))
    with_neutral(monkeypatch, Neutral("stepper.tableau", "DOPRI5", "0.2.0"))
    assert RunMetadata.from_json(json.dumps(document)).stepper.tableau == "DOPRI5"


def test_a_comparison_variant_keys_its_history_on_its_whole_representation():
    method = ComparisonMultistep(corrector="pece", order=4)
    assert method.history_key == repr(method)


def test_a_file_without_statistics_reports_none(tmp_path):
    path = tmp_path / "bare.h5"
    with h5py.File(path, "w"):
        pass
    assert _stored_stat_shapes(path) == {}
