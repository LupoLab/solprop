# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Paths are structural: every comparison is exact (class P).

import pytest

from solprop.recipe._errors import RecipeError
from solprop.recipe._paths import (
    Attribute,
    Index,
    Key,
    format_path,
    get_path,
    parse_path,
    set_path,
)


def build_table():
    return {
        "pressure": "1.2 bar",
        "pulses": [{"energy": "1 uJ"}, {"energy": "10 uJ", "fwhm": "30 fs"}],
        "launch": {"HE1,1:even": {"share": 0.5}},
        "plasma": {"He": True},
    }


@pytest.mark.parametrize(
    ("text", "steps"),
    [
        ("pressure", (Attribute("pressure"),)),
        ("pulses[1].energy", (Attribute("pulses"), Index(1), Attribute("energy"))),
        (
            'launch["HE1,1:even"].share',
            (Attribute("launch"), Key("HE1,1:even"), Attribute("share")),
        ),
        ("betas[10]", (Attribute("betas"), Index(10))),
        ("plasma.He", (Attribute("plasma"), Attribute("He"))),
        ('plasma["He"]', (Attribute("plasma"), Key("He"))),
        ('m["a\\"b"]', (Attribute("m"), Key('a"b'))),
        ('m["back\\\\slash"]', (Attribute("m"), Key("back\\slash"))),
        ('m["µ.[0]"]', (Attribute("m"), Key("µ.[0]"))),
        ('m[""]', (Attribute("m"), Key(""))),
        ("_x[0][1]", (Attribute("_x"), Index(0), Index(1))),
    ],
)
def test_path_parses_and_formats_back_canonically(text, steps):
    assert parse_path(text) == steps
    assert format_path(steps) == text
    assert parse_path(format_path(steps)) == steps


def test_path_attribute_and_key_differ_in_text_not_in_the_tree():
    table = build_table()
    assert parse_path("plasma.He") != parse_path('plasma["He"]')
    assert get_path(table, "plasma.He") is get_path(table, 'plasma["He"]')


def test_path_key_escapes_are_written_canonically():
    # A JSON escape that need not be one is read, and written plainly.
    assert format_path(parse_path('m["\\u0041"]')) == 'm["A"]'


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("", "starts with a keyword"),
        ("1x", "column 1: a path starts with a keyword"),
        ("[0]", "starts with a keyword"),
        ("a.", "column 3: expected a name"),
        ("a..b", "expected a name"),
        ("a[", "column 2: expected"),
        ("a[01]", "leading zero"),
        ("a[-1]", "no sign"),
        ("a[x]", "expected"),
        ('a["open', "expected"),
        ("a b", "column 2"),
        ("a.b c", "expected"),
        ('a["\\q"]', "not a JSON string"),
    ],
)
def test_parse_path_refuses_malformed_text(text, match):
    with pytest.raises(RecipeError, match=match):
        parse_path(text)


@pytest.mark.parametrize(
    ("steps", "match"),
    [
        ((), "starts with a keyword"),
        ((Index(0),), "starts with a keyword"),
        ((Key("a"),), "starts with a keyword"),
        ((Attribute("a"), Attribute("b c")), "not a keyword name"),
        ((Attribute("a"), Index(-1)), "at least 0"),
    ],
)
def test_format_path_refuses_invalid_steps(steps, match):
    with pytest.raises(RecipeError, match=match):
        format_path(steps)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("pressure", "1.2 bar"),
        ("pulses[1].energy", "10 uJ"),
        ('launch["HE1,1:even"].share', 0.5),
        ((Attribute("pulses"), Index(0), Attribute("energy")), "1 uJ"),
    ],
)
def test_get_path(path, value):
    assert get_path(build_table(), path) == value


def test_get_path_returns_the_table_itself():
    table = build_table()
    assert get_path(table, "pulses[1]") is table["pulses"][1]


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("radius", "radius: there is no keyword 'radius'"),
        ("pulses[2]", "pulses[2]: pulses has 2 elements, so no [2]"),
        ("pulses[1].shape", "pulses[1].shape: pulses[1] has no .shape"),
        ("pulses.energy", "pulses.energy: pulses is not a table, so it has no .energy"),
        ("pressure[0]", "pressure[0]: pressure is not an array, so it has no [0]"),
        ('launch["HE1,1:odd"]', 'launch["HE1,1:odd"]: launch has no ["HE1,1:odd"]'),
        ('pulses["energy"]', 'pulses is not a table, so it has no ["energy"]'),
    ],
)
def test_get_path_names_the_path_and_the_failing_step(path, message):
    with pytest.raises(RecipeError) as caught:
        get_path(build_table(), path)
    assert message in str(caught.value)
    assert caught.value.path == path


def test_set_path_returns_a_new_table_and_shares_the_rest():
    table = build_table()
    updated = set_path(table, "pulses[1].energy", "20 uJ")
    assert get_path(updated, "pulses[1]") == {"energy": "20 uJ", "fwhm": "30 fs"}
    assert table == build_table()
    assert updated["pulses"] is not table["pulses"]
    assert get_path(updated, "pulses[0]") is table["pulses"][0]
    assert updated["launch"] is table["launch"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("pressure", "1.5 bar"),
        ("radius", "125 um"),
        ('launch["HE1,1:even"].share', 0.25),
        ('launch["HE1,1:odd"]', {"share": 0.5}),
        ("pulses[0]", {"energy": "3 uJ"}),
        ("pulses[1].shape", "sech"),
    ],
)
def test_set_path_then_get_path(path, value):
    updated = set_path(build_table(), path, value)
    assert get_path(updated, path) == value
    assert get_path(updated, "pulses[0].energy") == (
        "3 uJ" if path == "pulses[0]" else "1 uJ"
    )


@pytest.mark.parametrize(
    ("path", "match"),
    [
        ("pulses[2]", "has 2 elements, so no"),
        ("pulses[2].energy", "has 2 elements, so no"),
        ("radius.value", "there is no keyword 'radius'"),
        ('launch["HE1,1:odd"].share', 'launch has no \\["HE1,1:odd"\\]'),
        ("pressure.value", "pressure is not a table"),
        ("pulses.energy", "pulses is not a table"),
        ("plasma[0]", "plasma is not an array"),
    ],
)
def test_set_path_refuses_a_missing_step(path, match):
    table = build_table()
    with pytest.raises(RecipeError, match=match) as caught:
        set_path(table, path, 1.0)
    assert caught.value.path == path
    assert table == build_table()
