# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# The document layer is structural: every comparison is exact (class P), arrays read
# from data files included, since a .npy file stores the bits.

import pickle
import re
import tomllib

import numpy as np
import pytest

from solprop.recipe._errors import RecipeError
from solprop.recipe._format import (
    EXECUTION_KEYS,
    FORMAT,
    MAX_ARRAY_LENGTH,
    REFUSED_BUILD_KEYS,
    TABLES,
    DataFile,
    decode_array,
    decode_complex,
    decode_data_file,
    decode_integer,
    decode_none,
    dump_text,
    encode_complex,
    encode_data_file,
    encode_integer,
    encode_none,
    load_array,
    parse_text,
    read_data_file,
    resolve_data_file,
    sha256_digest,
    upgrade,
    validate_document,
)
from solprop.recipe._quantity import parse_quantity, phase_unit

EXAMPLE = """\
[recipe]
format = 1
route = "capillary"
title = "RDW in a helium capillary"
solprop = "0.2.0"

[build]
radius = "125 um"
length = "1 m"
mode_average = true
gas = "He"
pressure = ["0 bar", "1.2 bar"]   # a gradient, as the keyword accepts
temperature = "293.15 K"
plasma = true
wavelength = "800 nm"
fwhm = "7.5 fs"
energy = "275 uJ"
wavelength_limits = ["100 nm", "1500 nm"]
trange = "200 fs"

[execution]
fields = "planes"
field_planes = [0, -1]
"""

SEED = 2**127 + 12345  # a drawn 128-bit noise seed


def full_document():
    """A document with every table and each value form, already encoded."""
    return {
        "recipe": {"format": 1, "route": "capillary", "title": "full"},
        "build": {
            "radius": "125 um",
            "pressure": ["0 bar", "1.2 bar"],
            "plasma": {"He": "ADK", "Ar": True},
            "pulses": [
                {"type": "GaussianPulse", "wavelength": "800 nm", "energy": "1 mJ"},
                {"type": "GaussianPulse", "wavelength": "400 nm", "energy": "10 uJ"},
            ],
            "launch": {"HE1,1:even": {"share": 0.75}},
            "noise": "none",
            "seed": encode_integer(SEED),
            "index": encode_complex(1.45 + 2e-6j),
            "betas": {"file": "betas.npy", "sha256": 64 * "0"},
            "phases": ["0 rad", "10 fs", "-20 fs^2"],
        },
        "execution": {"fields": "planes", "field_planes": [0, -1]},
        "pins": {
            "pressure": {
                "target": "zdw",
                "wavelength": "800 nm",
                "search": ["0.1 bar", "10 bar"],
                "root": "lowest",
            },
        },
        "scan": {
            "name": "energy-scan",
            "base_seed": encode_integer(SEED),
            "pulses[1].energy": {"start": "1 uJ", "stop": "10 uJ", "num": 4},
            'launch["HE1,1:even"].share': [0.25, 0.5],
        },
    }


# --- reading the document -------------------------------------------------------------


def test_parse_text_reads_the_format_example():
    document = parse_text(EXAMPLE)
    assert list(document) == ["recipe", "build", "execution"]
    assert document["recipe"]["route"] == "capillary"
    assert document["build"]["pressure"] == ["0 bar", "1.2 bar"]
    assert document["execution"] == {"fields": "planes", "field_planes": [0, -1]}


def test_parse_text_refuses_text_that_is_not_toml():
    with pytest.raises(RecipeError, match="not valid TOML"):
        parse_text("[recipe\nformat = 1")


@pytest.mark.parametrize(
    ("name", "suggestion"), [("builds", "build"), ("pin", "pins"), ("scans", "scan")]
)
def test_unknown_top_level_name_is_refused_with_the_nearest(name, suggestion):
    text = f'[recipe]\nformat = 1\nroute = "gnlse"\n[{name}]\nx = 1\n'
    with pytest.raises(RecipeError, match=f"unknown key '{name}'.*'{suggestion}'"):
        parse_text(text)


def test_unknown_top_level_name_without_a_near_one_lists_them():
    with pytest.raises(RecipeError, match="expected one of build, execution, pins"):
        parse_text('zzz = 1\n[recipe]\nformat = 1\nroute = "gnlse"\n')


def test_unknown_execution_and_recipe_keys_are_refused_with_the_nearest():
    with pytest.raises(RecipeError, match=r"'field' in \[execution\].*'fields'"):
        parse_text('[recipe]\nformat = 1\nroute = "g"\n[execution]\nfield = "all"\n')
    with pytest.raises(RecipeError, match=r"'titel' in \[recipe\].*'title'"):
        parse_text('[recipe]\nformat = 1\nroute = "g"\ntitel = "x"\n')


@pytest.mark.parametrize("key", EXECUTION_KEYS)
def test_execution_keys_are_accepted(key):
    document = validate_document(
        {"recipe": {"format": 1, "route": "gnlse"}, "execution": {key: "none"}}
    )
    assert document["execution"] == {key: "none"}


@pytest.mark.parametrize("key", REFUSED_BUILD_KEYS)
def test_refused_build_keys_are_named(key):
    document = {"recipe": {"format": 1, "route": "gnlse"}, "build": {key: "x"}}
    with pytest.raises(
        RecipeError, match=f"may not hold '{key}'.*command line"
    ) as caught:
        validate_document(document)
    assert caught.value.path == key


@pytest.mark.parametrize(
    ("recipe", "match"),
    [
        ({"route": "gnlse"}, "needs format"),
        ({"format": 1}, "needs route"),
        ({"format": True, "route": "gnlse"}, "format is an integer"),
        ({"format": 1.0, "route": "gnlse"}, "format is an integer"),
        ({"format": "1", "route": "gnlse"}, "format is an integer"),
        ({"format": 0, "route": "gnlse"}, "at least 1"),
        ({"format": 1, "route": 3}, "route is a string"),
        ({"format": 1, "route": "gnlse", "solprop": 0.2}, "solprop is a string"),
    ],
)
def test_recipe_table_is_checked(recipe, match):
    with pytest.raises(RecipeError, match=match):
        validate_document({"recipe": recipe})


def test_missing_or_malformed_tables_are_refused():
    with pytest.raises(RecipeError, match="needs a \\[recipe\\] table"):
        validate_document({"build": {}})
    with pytest.raises(RecipeError, match="\\[recipe\\] is a table"):
        validate_document({"recipe": 1})
    with pytest.raises(RecipeError, match="\\[build\\] is a table"):
        validate_document({"recipe": {"format": 1, "route": "g"}, "build": [1]})


def test_newer_format_is_refused_naming_it():
    # Refused as newer before anything a newer format might add is judged unknown.
    text = '[recipe]\nformat = 2\nroute = "gnlse"\nnew = 1\n[newtable]\nx = 1\n'
    with pytest.raises(RecipeError, match=r"format 2, newer.*format 1"):
        parse_text(text)


def test_upgrade_is_the_identity_for_format_one():
    document = full_document()
    upgraded = upgrade(document)
    assert upgraded == document
    assert upgraded is not document
    assert upgraded["recipe"] is not document["recipe"]
    assert FORMAT == 1
    with pytest.raises(RecipeError, match="cannot upgrade"):
        upgrade({"recipe": {"format": 2, "route": "g"}})


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("seed = 170141183460469231731687303715884105727", "outside 64 bits"),
        ("when = 1979-05-27", "is a date"),
        ("at = 07:32:00", "is a time"),
        ("x = [{a = 1979-05-27T07:32:00Z}]", "\\[build\\] x\\[0\\].a is a datetime"),
        ('launch."HE1,1" = {at = 07:32:00}', "\\[build\\] launch\\['HE1,1'\\].at is a"),
    ],
)
def test_values_that_are_not_recipe_values_are_refused(text, match):
    with pytest.raises(RecipeError, match=match):
        parse_text(f'[recipe]\nformat = 1\nroute = "g"\n[build]\n{text}\n')


def test_keys_that_are_not_strings_are_refused():
    document = {"recipe": {"format": 1, "route": "g"}, "build": {"plasma": {1: "x"}}}
    with pytest.raises(RecipeError, match=r"a key in \[build\] plasma is not a string"):
        validate_document(document)


def test_values_at_the_64_bit_limits_are_read():
    text = (
        '[recipe]\nformat = 1\nroute = "g"\n[build]\n'
        "low = -9223372036854775808\nhigh = 9223372036854775807\n"
    )
    assert parse_text(text)["build"] == {"low": -(2**63), "high": 2**63 - 1}


# --- writing the document -------------------------------------------------------------


def test_dump_text_round_trips_every_form():
    document = full_document()
    text = dump_text(document)
    assert parse_text(text) == document
    assert tomllib.loads(text) == document  # any TOML reader, not only ours


def test_dump_text_round_trips_the_format_example():
    document = parse_text(EXAMPLE)
    assert parse_text(dump_text(document)) == document


def test_dump_text_writes_the_tables_in_a_fixed_order():
    document = full_document()
    shuffled = {name: document[name] for name in reversed(TABLES)}
    shuffled["recipe"] = {"title": "full", "route": "capillary", "format": 1}
    text = dump_text(shuffled)
    assert text == dump_text(document)
    # The first header of each table, in order; [pins] holds only sub-tables, so
    # its first header is [pins.pressure].
    lines = text.splitlines()
    first = [
        next(i for i, line in enumerate(lines) if re.match(rf"\[{name}[].]", line))
        for name in TABLES
    ]
    assert first == sorted(first)
    assert text.startswith(
        '[recipe]\nformat = 1\nroute = "capillary"\ntitle = "full"\n'
    )


def test_dump_text_quotes_path_keys():
    text = dump_text(full_document())
    assert '[scan."pulses[1].energy"]' in text
    assert '"launch[\\"HE1,1:even\\"].share" = [' in text
    assert '[build.launch."HE1,1:even"]' in text


@pytest.mark.parametrize(
    ("value", "match"),
    [(SEED, "outside 64 bits"), (1 + 2j, "is a complex"), (None, "is a NoneType")],
)
def test_dump_text_refuses_unencoded_values(value, match):
    document = {"recipe": {"format": 1, "route": "g"}, "build": {"x": value}}
    with pytest.raises(RecipeError, match=match):
        dump_text(document)


def test_dump_text_refuses_what_parsing_refuses():
    with pytest.raises(RecipeError, match="may not hold 'output'"):
        dump_text({"recipe": {"format": 1, "route": "g"}, "build": {"output": "a.h5"}})


# --- value forms ----------------------------------------------------------------------


def test_none_spelling():
    assert decode_none("none") is None
    assert decode_none("None") == "None"
    assert decode_none(0) == 0
    assert encode_none(None) == "none"
    assert encode_none(0.0) == 0.0


@pytest.mark.parametrize("value", [1.45 + 2e-6j, -0.0 + 0j, complex(3, -4), 1e-300j])
def test_complex_round_trip(value):
    encoded = encode_complex(value)
    decoded = decode_complex(encoded)
    assert decoded == value
    document = {"recipe": {"format": 1, "route": "g"}, "build": {"z": encoded}}
    assert decode_complex(parse_text(dump_text(document))["build"]["z"]) == value


def test_complex_accepts_integer_parts():
    assert decode_complex({"re": 1, "im": 0}) == 1 + 0j


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ({"re": 1.0}, "is a table"),
        ({"re": 1.0, "im": 0.0, "x": 0.0}, "is a table"),
        (1.45, "is a table"),
        ({"re": True, "im": 0.0}, "real part True"),
        ({"re": 1.0, "im": "0"}, "imaginary part '0'"),
    ],
)
def test_complex_refusals(value, match):
    with pytest.raises(RecipeError, match=match) as caught:
        decode_complex(value, path="index")
    assert caught.value.path == "index"


def test_integer_128_bit_seed_as_decimal_string():
    encoded = encode_integer(SEED)
    assert encoded == str(SEED)
    assert decode_integer(encoded) == SEED
    document = {"recipe": {"format": 1, "route": "g"}, "build": {"seed": encoded}}
    assert decode_integer(parse_text(dump_text(document))["build"]["seed"]) == SEED


@pytest.mark.parametrize(
    ("value", "encoded"),
    [(42, 42), (2**63 - 1, 2**63 - 1), (-(2**63), -(2**63)), (2**63, str(2**63))],
)
def test_integer_encoding_at_the_64_bit_limits(value, encoded):
    assert encode_integer(value) == encoded
    assert decode_integer(encoded) == value


@pytest.mark.parametrize(
    ("value", "expected"), [(7, 7), ("-12", -12), ("+12", 12), (" 12 ", 12)]
)
def test_integer_forms(value, expected):
    assert decode_integer(value) == expected


@pytest.mark.parametrize("value", [3.0, True, "1.5", "0x10", "1e3", "abc", "", None])
def test_integer_refusals(value):
    with pytest.raises(RecipeError, match="expected an integer") as caught:
        decode_integer(value, path="seed")
    assert caught.value.path == "seed"


def test_array_elements_take_their_unit_by_index():
    phases = decode_array(
        ["0 rad", "10 fs", "-20 fs^2"],
        lambda value, index, path: parse_quantity(value, phase_unit(index), path=path),
        path="phases",
    )
    assert phases == [0.0, 10e-15, -20e-30]


def test_array_element_error_names_the_element_path():
    def element(value, index, path):
        return parse_quantity(value, phase_unit(index), path=path)

    with pytest.raises(RecipeError) as caught:
        decode_array(["0 rad", "10 fs^2"], element, path="phases")
    assert caught.value.path == "phases[1]"


def test_array_at_the_limit_is_held_inline_and_over_it_is_refused():
    def identity(value, index, path):
        return value

    assert MAX_ARRAY_LENGTH == 10_000
    at_limit = list(range(MAX_ARRAY_LENGTH))
    assert decode_array(at_limit, identity) == at_limit
    with pytest.raises(RecipeError, match=r"10001 elements.*\.npy") as caught:
        decode_array([*at_limit, 0], identity, path="save_planes")
    assert caught.value.path == "save_planes"


def test_array_refusals():
    def identity(value, index, path):
        return value

    with pytest.raises(RecipeError, match="expected an array"):
        decode_array(1.0, identity, path="betas")
    with pytest.raises(RecipeError, match="one-dimensional") as caught:
        decode_array([1.0, [2.0]], identity, path="betas")
    assert caught.value.path == "betas[1]"


# --- data files -----------------------------------------------------------------------


def write_npy(path, array, *, allow_pickle=False):
    np.save(path, array, allow_pickle=allow_pickle)
    return DataFile(path.name, sha256_digest(path.read_bytes()))


def test_data_file_reference_forms():
    reference = DataFile("table.npy", 64 * "A")
    assert reference.sha256 == 64 * "a"
    assert decode_data_file(encode_data_file(reference)) == reference
    assert sha256_digest(b"") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


@pytest.mark.parametrize(
    ("file", "digest", "match"),
    [
        ("", 64 * "0", "needs a file name"),
        ("/etc/table.npy", 64 * "0", "relative to the recipe"),
        ("C:\\data\\table.npy", 64 * "0", "relative to the recipe"),
        ("\\\\server\\share\\t.npy", 64 * "0", "relative to the recipe"),
        ("table.npy", 63 * "0", "64 hexadecimal digits"),
        ("table.npy", 64 * "g", "64 hexadecimal digits"),
    ],
)
def test_data_file_reference_refusals(file, digest, match):
    with pytest.raises(RecipeError, match=match):
        DataFile(file, digest)


@pytest.mark.parametrize(
    "value",
    [
        {"file": "t.npy"},
        {"file": "t.npy", "sha256": 64 * "0", "extra": 1},
        {"file": 1, "sha256": 64 * "0"},
        {"file": "t.npy", "sha256": 0},
        "t.npy",
    ],
)
def test_decode_data_file_refuses_other_tables(value):
    with pytest.raises(RecipeError, match="is a table") as caught:
        decode_data_file(value, path="betas")
    assert caught.value.path == "betas"


def test_decode_data_file_bad_digest_names_the_path():
    with pytest.raises(RecipeError, match="64 hexadecimal") as caught:
        decode_data_file({"file": "t.npy", "sha256": "abc"}, path="betas")
    assert caught.value.path == "betas"


@pytest.mark.parametrize(
    "array",
    [
        np.linspace(-1.0, 1.0, 7),
        np.arange(12, dtype=np.int32).reshape(3, 4),
        np.array([1 + 2j, -3j]),
        np.array([True, False]),
    ],
)
def test_load_array_reads_the_exact_array(tmp_path, array):
    reference = write_npy(tmp_path / "table.npy", array)
    loaded = load_array(reference, tmp_path)
    assert loaded.dtype == array.dtype
    assert np.array_equal(loaded, array)


def test_load_array_refuses_a_changed_file(tmp_path):
    reference = write_npy(tmp_path / "table.npy", np.zeros(3))
    np.save(tmp_path / "table.npy", np.ones(3))
    with pytest.raises(RecipeError, match=r"has SHA-256 \w+, not the") as caught:
        load_array(reference, tmp_path, path="betas")
    assert caught.value.path == "betas"


def test_load_array_refuses_a_pickled_array(tmp_path):
    reference = write_npy(
        tmp_path / "objects.npy", np.array([{"a": 1}], dtype=object), allow_pickle=True
    )
    with pytest.raises(RecipeError, match="pickling disabled"):
        load_array(reference, tmp_path)


def test_load_array_refuses_other_files(tmp_path):
    archive = tmp_path / "tables.npz"
    np.savez(archive, a=np.zeros(2))
    with pytest.raises(RecipeError, match=r"not a \.npy file"):
        load_array(
            DataFile(archive.name, sha256_digest(archive.read_bytes())), tmp_path
        )
    text = tmp_path / "table.csv"
    text.write_text("1,2,3\n", encoding="utf-8")
    with pytest.raises(RecipeError, match=r"not a \.npy file"):
        load_array(DataFile(text.name, sha256_digest(text.read_bytes())), tmp_path)
    strings = write_npy(tmp_path / "strings.npy", np.array(["a", "bc"]))
    with pytest.raises(RecipeError, match="holds <U2, not numbers"):
        load_array(strings, tmp_path)
    truncated = tmp_path / "truncated.npy"
    np.save(truncated, np.zeros(100))
    truncated.write_bytes(truncated.read_bytes()[:-8])
    reference = DataFile(truncated.name, sha256_digest(truncated.read_bytes()))
    with pytest.raises(RecipeError, match="cannot load"):
        load_array(reference, tmp_path)


def test_data_file_must_exist_and_be_a_file(tmp_path):
    with pytest.raises(RecipeError, match="cannot find the data file"):
        resolve_data_file(DataFile("missing.npy", 64 * "0"), tmp_path)
    (tmp_path / "folder").mkdir()
    with pytest.raises(RecipeError, match="is not a file"):
        resolve_data_file(DataFile("folder", 64 * "0"), tmp_path)


def test_data_file_outside_the_directory_needs_the_callers_allowance(tmp_path):
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir()
    reference = write_npy(tmp_path / "outside.npy", np.arange(3.0))
    escaping = DataFile("../outside.npy", reference.sha256)
    with pytest.raises(RecipeError, match="outside the recipe's directory"):
        load_array(escaping, recipe_dir)
    loaded = load_array(escaping, recipe_dir, allow_outside=True)
    assert np.array_equal(loaded, np.arange(3.0))


def test_data_file_symlink_escaping_the_directory_is_refused(tmp_path):
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir()
    reference = write_npy(tmp_path / "outside.npy", np.arange(3.0))
    link = recipe_dir / "inside.npy"
    try:
        link.symlink_to(tmp_path / "outside.npy")
    except OSError as error:
        # Windows creates symlinks only with Developer Mode or elevation.
        pytest.skip(f"this account cannot create symlinks: {error}")
    linked = DataFile("inside.npy", reference.sha256)
    with pytest.raises(RecipeError, match="outside the recipe's directory"):
        read_data_file(linked, recipe_dir)
    assert read_data_file(linked, recipe_dir, allow_outside=True) == (
        (tmp_path / "outside.npy").read_bytes()
    )


def test_data_file_symlink_within_the_directory_is_read(tmp_path):
    reference = write_npy(tmp_path / "table.npy", np.arange(4.0))
    (tmp_path / "data").mkdir()
    try:
        (tmp_path / "data" / "alias.npy").symlink_to(tmp_path / "table.npy")
    except OSError as error:
        # Windows creates symlinks only with Developer Mode or elevation.
        pytest.skip(f"this account cannot create symlinks: {error}")
    loaded = load_array(DataFile("data/alias.npy", reference.sha256), tmp_path)
    assert np.array_equal(loaded, np.arange(4.0))


# --- the error ------------------------------------------------------------------------


def test_recipe_error_pickles_with_its_path():
    error = RecipeError("a pressure cannot be negative", path="pressure")
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is RecipeError
    assert restored.path == "pressure"
    assert restored.message == error.message
    assert str(restored) == str(error)
    plain = pickle.loads(pickle.dumps(RecipeError("bad")))
    assert plain.path is None
    assert str(plain) == "bad"
